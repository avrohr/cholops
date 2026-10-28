"""Block deletion preserves survivor order and differentiates the requested matrix."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import (
    Cholesky,
    PermutedCholesky,
    chol_delete,
    chol_delete_append,
    chol_delete_last,
    cholesky,
)

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 3e-10 if jax.config.x64_enabled else 8e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def fixture(n, m, s):
    rng = np.random.default_rng(53)
    X = jnp.asarray(rng.normal(size=(n, 3)) / 3, DTYPE)
    A = jnp.eye(n, dtype=DTYPE) + X @ X.T
    K = jnp.asarray(rng.normal(size=(m, s)) / 10, DTYPE)
    Z = jnp.asarray(rng.normal(size=(s, 3)) / 3, DTYPE)
    C = jnp.eye(s, dtype=DTYPE) + Z @ Z.T
    return A, K, C


def evaluate(fn, args):
    directions = jax.tree.map(lambda x: jnp.cos(x) / 20, args)
    value, tangent = jax.jit(lambda *a: jax.jvp(fn, a, directions))(*args)
    G = jnp.sin(jnp.arange(value.size, dtype=DTYPE)).reshape(value.shape)
    gradients = jax.jit(jax.grad(lambda *a: jnp.sum(fn(*a) * G), range(len(args))))(
        *args
    )
    close(
        jnp.vdot(G, tangent), sum(jnp.vdot(d, g) for d, g in zip(directions, gradients))
    )
    return value, tangent, gradients


@pytest.mark.parametrize(
    "deleted", [[], [0], [3], [5], [1, 2], [4, 1], [5, 4], list(range(6))]
)
@pytest.mark.parametrize("s", [0, 3])
def test_block_deletion_and_append_values_and_all_derivatives(deleted, s):
    keep = np.delete(np.arange(6), deleted)
    indices = jnp.asarray(deleted, dtype=jnp.int32)
    A, K, C = fixture(6, len(keep), s)
    L = cholesky(A).at[jnp.triu_indices(6, 1)].set(jnp.nan)

    def actual(L, K, C):
        if s == 0:
            return chol_delete(L, indices)
        return chol_delete_append(L, indices, K, C)

    def reference(L, K, C):
        A = jnp.tril(L) @ jnp.tril(L).T
        A = A[keep[:, None], keep]
        return jnp.linalg.cholesky(jnp.block([[A, K], [K.T, C]]))

    # Use finite tangents even though the ignored upper triangle contains NaNs.
    args = (L, K, C)
    directions = tuple(jnp.full(x.shape, 0.03, DTYPE) for x in args)
    close(jax.jit(actual)(*args), reference(*args))
    close(
        jax.jit(lambda *a: jax.jvp(actual, a, directions))(*args),
        jax.jvp(reference, args, directions),
    )
    size = len(keep) + s
    G = jnp.cos(jnp.arange(size * size, dtype=DTYPE)).reshape(size, size)
    got = jax.jit(lambda *a: jax.vjp(actual, *a)[1](G))(*args)
    close(got, jax.vjp(reference, *args)[1](G))
    np.testing.assert_array_equal(jnp.triu(got[0], 1), 0)
    np.testing.assert_array_equal(got[2], got[2].T)


def test_runtime_block_indices_batched_with_shared_factor_and_inputs():
    A, K, C = fixture(6, 4, 2)
    L = cholesky(A)
    indices = jnp.array([[0, 1], [4, 1], [5, 4]])
    kept = jnp.array([[2, 3, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3]])

    def actual(L, K, C):
        return jax.vmap(chol_delete_append, (None, 0, None, None))(L, indices, K, C)

    def reference(L, K, C):
        A = jnp.tril(L) @ jnp.tril(L).T
        return jax.vmap(
            lambda k: jnp.linalg.cholesky(jnp.block([[A[k[:, None], k], K], [K.T, C]]))
        )(kept)

    close(evaluate(actual, (L, K, C)), evaluate(reference, (L, K, C)))


@pytest.mark.parametrize("cls", [Cholesky, PermutedCholesky])
@pytest.mark.parametrize(
    "deleted,s", [([4, 1], 3), ([5, 4], 0), ([], 2), (list(range(6)), 2)]
)
def test_factor_block_modifications_and_consumer_derivatives(cls, deleted, s):
    keep = np.delete(np.arange(6), deleted)
    A, K, C = fixture(6, len(keep), s)
    indices = jnp.array(deleted, dtype=jnp.int32)
    rhs = jnp.linspace(-0.3, 0.6, len(keep) + s, dtype=DTYPE)
    order = jnp.array([4, 0, 3, 5, 1, 2])

    def state(A):
        if cls is Cholesky:
            return cls.from_matrix(A)
        return cls(cholesky(A[order[:, None], order]), order, jnp.argsort(order))

    def actual(A, K, C, rhs):
        original = state(A)
        out = (
            original.delete(indices)
            if s == 0
            else original.delete_append(indices, K, C)
        )
        return (
            out.logdet() + out.quadratic(rhs) + jnp.sum(jnp.sin(rhs) * out.solve(rhs))
        )

    def reference(A, K, C, rhs):
        A = (A + A.T) / 2
        B = jnp.block([[A[keep[:, None], keep], K], [K.T, (C + C.T) / 2]])
        x = jnp.linalg.solve(B, rhs)
        return jnp.linalg.slogdet(B)[1] + rhs @ x + jnp.sum(jnp.sin(rhs) * x)

    args = A, K, C, rhs
    close(
        jax.jit(jax.value_and_grad(actual, (0, 1, 2, 3)))(*args),
        jax.jit(jax.value_and_grad(reference, (0, 1, 2, 3)))(*args),
    )
    directions = jax.tree.map(lambda x: jnp.cos(x) / 20, args)
    close(
        jax.jit(lambda *a: jax.jvp(actual, a, directions))(*args),
        jax.jvp(reference, args, directions),
    )
    out = jax.jit(lambda a: state(a).delete_append(indices, K, C))(A)
    if cls is Cholesky:
        matrix = out.factor @ out.factor.T
    else:
        p = out.inverse_order
        matrix = (out.physical_factor @ out.physical_factor.T)[p[:, None], p]
        np.testing.assert_array_equal(out.order[p], np.arange(p.size))
    close(matrix, jnp.block([[A[keep[:, None], keep], K], [K.T, C]]))


@pytest.mark.parametrize("r", [0, 2, 5])
def test_delete_last_ignores_discarded_entries_and_differentiates_the_slice(r):
    m = 5 - r
    L = jnp.eye(5, dtype=DTYPE).at[m:].set(jnp.nan)
    expected = jnp.eye(m, dtype=DTYPE)
    for fn in (
        lambda L: chol_delete_last(L, r),
        lambda L: Cholesky.from_factor(L).delete_last(r).factor,
    ):
        close(jax.jit(fn)(L), expected)
        close(
            jax.jit(lambda L: jax.jvp(fn, (L,), (jnp.ones_like(L),)))(L)[1],
            jnp.tril(jnp.ones((m, m), DTYPE)),
        )
        gradient = jax.jit(jax.grad(lambda L: fn(L).sum()))(L)
        close(
            gradient,
            jnp.zeros_like(L).at[:m, :m].set(jnp.tril(jnp.ones((m, m), DTYPE))),
        )


def test_empty_factor_delete_and_append():
    L = jnp.empty((0, 0), DTYPE)
    indices = jnp.empty((0,), jnp.int32)
    C = jnp.array([[1.5, 0.2], [0.2, 1.1]], DTYPE)
    assert chol_delete(L, indices).shape == (0, 0)
    close(chol_delete_append(L, indices, jnp.empty((0, 2), DTYPE), C), cholesky(C))


@pytest.mark.parametrize("implementation", ["tensor", Cholesky, PermutedCholesky])
def test_block_delete_append_in_checkpointed_scan(implementation):
    X = jnp.sin(jnp.arange(18, dtype=DTYPE)).reshape(6, 3) / 3
    incoming = jnp.cos(jnp.arange(24, dtype=DTYPE)).reshape(4, 2, 3) / 3
    deleted = jnp.array([[0, 1], [4, 1], [5, 4], [2, 0]])
    kept = jnp.array([[2, 3, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3], [1, 3, 4, 5]])
    rhs = jnp.linspace(-0.4, 0.7, 6, dtype=DTYPE)

    def run(X, incoming, scale, dense=False):
        A = jnp.eye(6, dtype=DTYPE) + scale * X @ X.T
        state = (
            A
            if dense
            else cholesky(A)
            if implementation == "tensor"
            else implementation.from_matrix(A)
        )

        def step(carry, inputs):
            state, X = carry
            indices, keep, new = inputs
            K = scale * X[keep] @ new.T
            C = jnp.eye(2, dtype=DTYPE) + scale * new @ new.T
            X = jnp.concatenate((X[keep], new))
            if dense:
                state = jnp.eye(6, dtype=DTYPE) + scale * X @ X.T
                score = jnp.linalg.slogdet(state)[1] + rhs @ jnp.linalg.solve(
                    state, rhs
                )
            else:
                if implementation == "tensor":
                    state = chol_delete_append(state, indices, K, C)
                    factor = Cholesky.from_factor(state)
                else:
                    state = state.delete_append(indices, K, C)
                    factor = state
                score = factor.logdet() + factor.quadratic(rhs)
            return (state, X), score

        return jax.lax.scan(
            jax.checkpoint(step), (state, X), (deleted, kept, incoming)
        )[1]

    args = X, incoming, jnp.asarray(0.7, DTYPE)
    weights = jnp.arange(1, 5, dtype=DTYPE)

    def candidate(*a):
        return weights @ run(*a)

    def reference(*a):
        return weights @ run(*a, dense=True)

    close(
        jax.jit(jax.value_and_grad(candidate, (0, 1, 2)))(*args),
        jax.jit(jax.value_and_grad(reference, (0, 1, 2)))(*args),
    )
    directions = jax.tree.map(lambda x: jnp.cos(x) / 20, args)
    close(
        jax.jit(lambda *a: jax.jvp(candidate, a, directions))(*args),
        jax.jvp(reference, args, directions),
    )


@pytest.mark.parametrize("r", [5, 27])
def test_block_deletion_of_dense_factor_with_few_or_many_survivors(r):
    n = 32
    rng = np.random.default_rng(85)
    X = jnp.asarray(rng.normal(size=(n, n)) / np.sqrt(n), DTYPE)
    L = cholesky(jnp.eye(n, dtype=DTYPE) + X @ X.T)
    deleted = rng.choice(n, r, replace=False)
    keep = np.delete(np.arange(n), deleted)
    indices = jnp.asarray(deleted, jnp.int32)

    def reference(L):
        rows = jnp.tril(L)[keep]
        return jnp.linalg.cholesky(rows @ rows.T)

    close(
        evaluate(lambda L: chol_delete(L, indices), (L,)),
        evaluate(reference, (L,)),
    )
