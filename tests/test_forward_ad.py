"""First-order derivatives under JIT, batching, linearization and checkpointing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import (
    Cholesky,
    PermutedCholesky,
    chol_delete,
    chol_delete_append,
    chol_rank1_positive,
    chol_replace,
    cholesky,
)

DTYPE = np.float64 if jax.config.x64_enabled else np.float32
TOL = 4e-10 if jax.config.x64_enabled else 2e-4


def close(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=TOL, atol=TOL * 0.1)


def direction(x):
    return jnp.cos(jnp.arange(x.size, dtype=x.dtype) + 0.3).reshape(x.shape) * 0.03


def covariance(L):
    L = jnp.tril(L)
    return L @ L.T


def replaced(L, index, k, c, append=False):
    A = covariance(L)
    keep = jnp.arange(L.shape[0] - 1)
    keep = keep + (keep >= index)
    if append:
        A = A[keep[:, None], keep[None, :]]
        A = jnp.pad(A, ((0, 1), (0, 1)))
        index, keep = L.shape[0] - 1, jnp.arange(L.shape[0] - 1)
    A = A.at[index, keep].set(k).at[keep, index].set(k).at[index, index].set(c)
    return jnp.linalg.cholesky(A)


@pytest.mark.parametrize("index", [0, 2, 3, 4])
@pytest.mark.parametrize("append", [False, True])
def test_replacement_tangent_all_inputs_and_duality(index, append):
    L = jnp.linalg.cholesky(jnp.eye(5) + 0.2 * jnp.ones((5, 5)))
    args = L, jnp.linspace(-0.1, 0.2, 4), jnp.asarray(1.5)
    tangents = jax.tree.map(direction, args)
    fn = chol_delete_append if append else chol_replace
    G = direction(L)

    def evaluate(i, *args):
        def operation(*xs):
            return fn(
                xs[0], i[None] if append else i, xs[1][:, None], xs[2].reshape(1, 1)
            )

        value, tangent = jax.jvp(operation, args, tangents)
        gradients = jax.vjp(operation, *args)[1](G)
        return value, tangent, gradients

    def reference(L, k, c):
        return replaced(L, index, k, c, append)

    expected = (*jax.jvp(reference, args, tangents), jax.vjp(reference, *args)[1](G))
    for evaluate_fn in (evaluate, jax.jit(evaluate)):
        result = evaluate_fn(jnp.asarray(index), *args)
        close(result, expected)
        close(
            jnp.vdot(G, result[1]),
            sum(jnp.vdot(t, g) for t, g in zip(tangents, result[2])),
        )


@pytest.mark.parametrize("name", ["cholesky", "positive", "delete", "replace"])
def test_jacfwd_matches_jacrev_and_independent_dense(name):
    L = jnp.array([[1.3, 0.0, 0.0], [0.2, 1.1, 0.0], [-0.1, 0.3, 1.2]])
    u = jnp.array([0.1, -0.2, 0.05])
    if name == "cholesky":
        x, fn, ref = covariance(L), cholesky, jnp.linalg.cholesky
    elif name == "positive":
        x, fn = L, lambda L: chol_rank1_positive(L, u)

        def ref(L):
            return jnp.linalg.cholesky(covariance(L) + jnp.outer(u, u))
    elif name == "delete":
        x, fn = L, lambda L: chol_delete(L, jnp.asarray([1]))

        def ref(L):
            return jnp.linalg.cholesky(
                covariance(L)[jnp.array([0, 2])[:, None], jnp.array([0, 2])[None, :]]
            )
    else:
        x, fn = (
            L,
            lambda L: chol_replace(
                L, 1, u[:2, None], jnp.array([[1.5]], dtype=L.dtype)
            ),
        )

        def ref(L):
            return replaced(L, 1, u[:2], 1.5)

    forward_jacobian = jax.jit(jax.jacfwd(fn))(x)
    close(forward_jacobian, jax.jit(jax.jacrev(fn))(x))
    close(forward_jacobian, jax.jacfwd(ref)(x))
    # A cached linearization must give the same directional derivative as JVP.
    _, push = jax.linearize(fn, x)
    close(push(direction(x)), jax.jvp(ref, (x,), (direction(x),))[1])


@pytest.mark.parametrize("n", [1, 5])
def test_deletion_shared_factor_batched_indices(n):
    # All deletion branches share one factor, so their adjoints must add.
    # The singleton also exercises the empty output and its zero pullback.
    L = jnp.linalg.cholesky(jnp.eye(n) + 0.2 * jnp.ones((n, n)))
    L = L.at[jnp.triu_indices(n, 1)].set(jnp.nan)
    indices = jnp.arange(n)

    def candidate(L):
        return jax.vmap(chol_delete, (None, 0))(L, indices[:, None])

    def reference(L):
        A = covariance(L)

        def one(index):
            keep = jnp.arange(n - 1)
            keep = keep + (keep >= index)
            return jnp.linalg.cholesky(A[keep[:, None], keep[None, :]])

        return jax.vmap(one)(indices)

    tangent = direction(L)
    G = direction(candidate(L))

    def evaluate(fn, L):
        return *jax.jvp(fn, (L,), (tangent,)), jax.vjp(fn, L)[1](G)

    close(jax.jit(lambda L: evaluate(candidate, L))(L), evaluate(reference, L))


def test_batched_dynamic_indices_shared_tangents_and_unused_triangle():
    L = jnp.linalg.cholesky(jnp.eye(5) + 0.2 * jnp.ones((5, 5)))
    L = L.at[jnp.triu_indices(5, 1)].set(jnp.nan)
    factors = jnp.stack([L, L * 1.2, L * 0.9, L * 1.1])
    indices = jnp.asarray([0, 2, 3, 4])
    args = factors, jnp.linspace(-0.1, 0.2, 4), jnp.asarray(1.5)
    tangents = jax.tree.map(direction, args)
    for fn in (chol_replace, chol_delete_append):

        def mapped(L, k, c):
            return jax.vmap(fn, (0, 0, None, None))(
                L,
                indices[:, None] if fn is chol_delete_append else indices,
                k[:, None],
                c.reshape(1, 1),
            )

        def ref(L, k, c):
            return jax.vmap(
                lambda a, i: replaced(a, i, k, c, fn is chol_delete_append)
            )(L, indices)

        close(
            jax.jit(lambda *xs: jax.jvp(mapped, xs, tangents))(*args),
            jax.jvp(ref, args, tangents),
        )
        # Check both orders of vmap and JVP, including batches of tangent directions.
        close(
            jax.jit(jax.vmap(lambda d: jax.jvp(mapped, args, d)[1]))(
                jax.tree.map(lambda t: jnp.stack([t, -t]), tangents)
            ),
            jax.vmap(lambda d: jax.jvp(ref, args, d)[1])(
                jax.tree.map(lambda t: jnp.stack([t, -t]), tangents)
            ),
        )


@pytest.mark.parametrize("cls", [Cholesky, PermutedCholesky])
def test_matrix_to_managed_sequence_to_consumers_both_modes(cls):
    # Eight updates include all index positions and repeated replacement. This
    # checks gradients through factor construction and checkpointed replacements.
    X = jnp.arange(15, dtype=jnp.asarray(0.0).dtype).reshape(5, 3) / 20
    incoming = jnp.sin(jnp.arange(24, dtype=X.dtype)).reshape(8, 3) / 5
    indices = jnp.array([0, 2, 3, 4, 0, 3, 2, 4])
    args = X, incoming, jnp.asarray(0.7)

    def run(X, incoming, scale, recompute=False):
        def matrix(X):
            return jnp.eye(5) + scale * X @ X.T

        initial = cls.from_matrix(matrix(X))

        def step(carry, request):
            factor, X = carry
            i, new = request
            keep = jnp.arange(4) + (jnp.arange(4) >= i)
            k, c = scale * X[keep] @ new, 1 + scale * jnp.vdot(new, new)
            X = X.at[i].set(new)
            if recompute:
                factor = cls.from_matrix(matrix(X))
            else:
                factor = factor.replace(i, k[:, None], c.reshape(1, 1))
            rhs = X[:, 0]
            score = factor.logdet() + factor.quadratic(rhs) + jnp.sum(factor.solve(rhs))
            return (factor, X), score

        return jax.lax.scan(jax.checkpoint(step), (initial, X), (indices, incoming))[1]

    tangent = jax.tree.map(direction, args)

    def candidate(*xs):
        return run(*xs)

    def reference(*xs):
        return run(*xs, recompute=True)

    close(
        jax.jit(lambda *xs: jax.jvp(candidate, xs, tangent))(*args),
        jax.jvp(reference, args, tangent),
    )
    close(
        jax.jit(jax.grad(lambda *xs: jnp.sum(candidate(*xs)), (0, 1, 2)))(*args),
        jax.grad(lambda *xs: jnp.sum(reference(*xs)), (0, 1, 2))(*args),
    )


@pytest.mark.parametrize("operation", ["cholesky", "positive", "replace"])
def test_conditioned_jvp_against_decimal(operation):
    from decimal import Decimal, localcontext

    from tests.reference.decimal_ad import (
        decimal_matrix,
        factor_and_tangent,
        matmul,
        transpose,
    )

    # A small clustered SE covariance exposes cancellation without a large
    # fixture. Float32 needs a larger diagonal to retain an SPD rounded target.
    n, index = 8, 3
    x = np.linspace(-0.3, 0.4, n)
    noise = 1e-8 if jax.config.x64_enabled else 1e-3
    A = jnp.asarray(
        np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2) + noise * np.eye(n), DTYPE
    )
    L = jnp.asarray(np.linalg.cholesky(np.asarray(A, dtype=np.float64)), DTYPE)
    keep = np.delete(np.arange(n), index)
    k = jnp.asarray(np.exp(-0.5 * (x[keep] - 0.17) ** 2), DTYPE)
    c = jnp.asarray(1 + noise, DTYPE)
    u = jnp.asarray(np.sin(np.arange(n)) * 0.2, DTYPE)
    if operation == "cholesky":
        args = (A,)
        tangents = ((direction(A) + direction(A).T) / 2,)
        fn = cholesky
    elif operation == "positive":
        args, tangents, fn = (
            (L, u),
            (jnp.tril(direction(L)), direction(u)),
            chol_rank1_positive,
        )
    else:
        args = L, k, c
        tangents = jnp.tril(direction(L)), direction(k), direction(c)

        def fn(L, k, c):
            return chol_replace(L, jnp.asarray(index), k[:, None], c.reshape(1, 1))

    with localcontext() as ctx:
        ctx.prec = 80
        if operation == "cholesky":
            target, perturbation = decimal_matrix(A), decimal_matrix(tangents[0])
        else:
            dl, dt = decimal_matrix(L), decimal_matrix(tangents[0])
            target = matmul(dl, transpose(dl))
            left, right = matmul(dt, transpose(dl)), matmul(dl, transpose(dt))
            perturbation = [
                [left[i][j] + right[i][j] for j in range(n)] for i in range(n)
            ]
            if operation == "positive":
                v = [Decimal.from_float(float(a)) for a in u]
                dv = [Decimal.from_float(float(a)) for a in tangents[1]]
                for i in range(n):
                    for j in range(n):
                        target[i][j] += v[i] * v[j]
                        perturbation[i][j] += dv[i] * v[j] + v[i] * dv[j]
            else:
                for j, value, tangent in zip(keep, k, tangents[1], strict=True):
                    target[index][j] = target[j][index] = Decimal.from_float(
                        float(value)
                    )
                    perturbation[index][j] = perturbation[j][index] = (
                        Decimal.from_float(float(tangent))
                    )
                target[index][index] = Decimal.from_float(float(c))
                perturbation[index][index] = Decimal.from_float(float(tangents[2]))
        expected = factor_and_tangent(target, perturbation)
    actual = jax.jit(lambda *xs: jax.jvp(fn, xs, tangents))(*args)
    for a, b in zip(actual, expected, strict=True):
        relative = np.linalg.norm(np.asarray(a, dtype=np.float64) - b) / np.linalg.norm(
            b
        )
        assert relative < (3e-6 if jax.config.x64_enabled else 3e-3)
