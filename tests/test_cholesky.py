"""Canonical factor operations and complete consumer gradients against matrices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import Cholesky, PermutedCholesky, chol_delete

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 3e-10 if jax.config.x64_enabled else 8e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def fixture():
    rng = np.random.default_rng(31)
    X = jnp.asarray(rng.normal(size=(5, 3)) / 3, DTYPE)
    return jnp.eye(5, dtype=DTYPE) + X @ X.T


def modified_matrix(A, method, args):
    keep = jnp.array([0, 1, 3, 4])
    if method == "append":
        K, C = args
        return jnp.block([[A, K], [K.T, (C + C.T) / 2]])
    if method == "delete":
        return A[keep[:, None], keep[None, :]]
    if method in ("replace", "delete_append"):
        k, c = args
        target = A.at[keep, 2].set(k).at[2, keep].set(k).at[2, 2].set(c)
        if method == "delete_append":
            order = jnp.concatenate((keep, jnp.array([2])))
            target = target[order[:, None], order[None, :]]
        return target
    if method == "replace_last":
        K, C = args
        return jnp.block([[A[:-2, :-2], K], [K.T, (C + C.T) / 2]])
    if method == "diag_update":
        return A.at[2, 2].add(args[0])
    u = args[0]
    alpha = args[1] if method == "rank1" else 1
    return A + alpha * jnp.outer(u, u)


def modify(state, method, args):
    if method in ("delete", "replace", "delete_append", "diag_update"):
        return getattr(state, method)(jnp.array(2), *args)
    return getattr(state, method)(*args)


def parameters(method, weight):
    k = jnp.linspace(0.1, 0.2, 5, dtype=DTYPE)
    c = jnp.asarray(1.5, DTYPE)
    if method == "append":
        return jnp.stack((k, -k / 2), axis=1), jnp.array(
            [[1.5, 0.1], [0.1, 1.2]], DTYPE
        )
    if method in ("replace", "delete_append"):
        return k[:-1], c
    if method == "delete":
        return ()
    if method == "replace_last":
        return jnp.outer(k[:3], k[:2]), jnp.array([[1.5, 0.1], [0.1, 1.2]], DTYPE)
    if method == "diag_update":
        return (jnp.asarray(weight, DTYPE),)
    if method == "rank1":
        return k, jnp.asarray(weight, DTYPE)
    return (k,)


CASES = [
    (name, 1.0)
    for name in (
        "append",
        "delete",
        "replace",
        "delete_append",
        "replace_last",
        "rank1_positive",
    )
] + [(name, weight) for name in ("rank1", "diag_update") for weight in (-0.3, 0.0, 0.4)]


@pytest.mark.parametrize("method,weight", CASES)
def test_modifications_and_consumer_gradients(method, weight):
    A = fixture()
    args = parameters(method, weight)
    target = modified_matrix(A, method, args)
    n = target.shape[0]
    rhs = jnp.arange(n * 2, dtype=DTYPE).reshape(n, 2) / (2 * n) - 0.2
    G = jnp.sin(jnp.arange(n * n, dtype=DTYPE).reshape(n, n)) / n
    original = Cholesky.from_matrix(A)
    updated = jax.jit(lambda s, a: modify(s, method, a))(original, args)
    close(updated.factor @ updated.factor.T, target)
    close(original.factor @ original.factor.T, A)

    def actual(A, args, rhs):
        state = modify(Cholesky.from_matrix(A), method, args)
        return (
            jnp.sum(G * state.factor)
            + state.logdet()
            + jnp.sum(state.quadratic(rhs))
            + jnp.sum(jnp.cos(rhs) * state.solve(rhs))
        )

    def reference(A, args, rhs):
        B = modified_matrix((A + A.T) / 2, method, args)
        x = jnp.linalg.solve(B, rhs)
        return (
            jnp.sum(G * jnp.linalg.cholesky(B))
            + jnp.linalg.slogdet(B)[1]
            + jnp.sum(rhs * x)
            + jnp.sum(jnp.cos(rhs) * x)
        )

    got = jax.jit(jax.value_and_grad(actual, argnums=(0, 1, 2)))(A, args, rhs)
    expected = jax.jit(jax.value_and_grad(reference, argnums=(0, 1, 2)))(A, args, rhs)
    close(got, expected)
    if method in ("rank1", "diag_update") and weight == 0:
        assert abs(float(got[1][1][-1])) > 1e-4


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
@pytest.mark.parametrize("columns", [None, 2])
def test_consumers_from_factor_and_upstream_gradients(container, columns):
    L = jnp.linalg.cholesky(fixture())
    rhs = jnp.linspace(-0.3, 0.5, 5, dtype=DTYPE)
    if columns:
        rhs = jnp.stack((rhs, 2 * rhs), axis=1)

    def consume(L, rhs):
        state = container.from_factor(L)
        return state.solve(rhs), state.logdet(), state.quadratic(rhs)

    def dense(L, rhs):
        L = jnp.tril(L)
        A = L @ L.T
        x = jnp.linalg.solve(A, rhs)
        return x, jnp.linalg.slogdet(A)[1], jnp.sum(rhs * x, axis=0)

    close(jax.jit(consume)(L, rhs), dense(L, rhs))

    def loss(fn, L, rhs):
        x, logdet, q = fn(L, rhs)
        return jnp.sum(jnp.sin(x)) + logdet + jnp.sum(q)

    close(
        jax.jit(jax.grad(lambda L, b: loss(consume, L, b), argnums=(0, 1)))(L, rhs),
        jax.jit(jax.grad(lambda L, b: loss(dense, L, b), argnums=(0, 1)))(L, rhs),
    )
    # Wrapping native Cholesky must not replace its upstream derivative.
    close(
        jax.grad(lambda A: loss(consume, jnp.linalg.cholesky(A), rhs))(fixture()),
        jax.grad(lambda A: loss(dense, jnp.linalg.cholesky(A), rhs))(fixture()),
    )


@pytest.mark.parametrize("method", list(dict.fromkeys(name for name, _ in CASES)))
def test_jvp_from_factor_for_all_methods(method):
    L = jnp.linalg.cholesky(fixture())
    args = parameters(method, 0.0)
    tangent = jnp.tril(jnp.ones_like(L) / 20)

    def actual(L):
        return modify(Cholesky.from_factor(L), method, args).factor

    def expected(L):
        return jnp.linalg.cholesky(modified_matrix(L @ L.T, method, args))

    close(jax.jvp(actual, (L,), (tangent,)), jax.jvp(expected, (L,), (tangent,)))


def test_from_matrix_matches_dense_factor_and_gradient():
    A = fixture()
    state = jax.jit(Cholesky.from_matrix)(A)
    close(state.factor, jnp.linalg.cholesky(A))
    G = jnp.arange(25, dtype=DTYPE).reshape(5, 5) / 25
    close(
        jax.grad(lambda A: jnp.sum(Cholesky.from_matrix(A).factor * G))(A),
        jax.grad(lambda A: jnp.sum(jnp.linalg.cholesky(A) * G))(A),
    )
    _, tangent = jax.jvp(lambda A: Cholesky.from_matrix(A).logdet(), (A,), (A,))
    close(tangent, jnp.asarray(A.shape[0], DTYPE))


def test_factor_is_immutable():
    state = Cholesky.from_matrix(fixture())
    with pytest.raises((AttributeError, TypeError)):
        state.factor = jnp.eye(5, dtype=DTYPE)


def test_factories_and_solve_reject_incompatible_inputs():
    state = Cholesky.from_matrix(fixture())
    with pytest.raises(ValueError):
        Cholesky.from_matrix(jnp.ones((2, 2, 2), DTYPE))
    with pytest.raises(TypeError):
        Cholesky.from_factor(jnp.ones((2, 2), jnp.int32))
    with pytest.raises(ValueError):
        state.solve(jnp.ones((4,), DTYPE))


def test_positive_update_logdet_jvp_from_factor():
    state = Cholesky.from_matrix(fixture())
    actual = jax.jvp(
        lambda L: Cholesky.from_factor(L).rank1_positive(jnp.ones(5, DTYPE)).logdet(),
        (state.factor,),
        (state.factor,),
    )
    expected = jax.jvp(
        lambda L: jnp.linalg.slogdet(L @ L.T + jnp.ones((5, 5), DTYPE))[1],
        (state.factor,),
        (state.factor,),
    )
    close(actual, expected)


def test_empty_factor_append_delete_and_consumers():
    empty = Cholesky.from_factor(jnp.empty((0, 0), DTYPE))
    one = jax.jit(
        lambda s: s.append(jnp.empty((0, 1), DTYPE), jnp.array([[2.0]], DTYPE))
    )(empty)
    close(one.factor @ one.factor.T, jnp.array([[2.0]], DTYPE))
    back = jax.jit(lambda s: s.delete(0))(one)
    assert back.factor.shape == (0, 0)
    close(back.logdet(), jnp.asarray(0, DTYPE))
    close(back.quadratic(jnp.empty((0,), DTYPE)), jnp.asarray(0, DTYPE))


def test_delete_from_factor_matches_tensor_and_dense_gradient():
    L = jnp.array([[1.3, 0.0, 0.0], [0.2, 1.1, 0.0], [-0.1, 0.3, 1.2]])
    keep = jnp.array([0, 2])

    def reference(L):
        L = jnp.tril(L)
        A = L @ L.T
        return jnp.linalg.cholesky(A[keep[:, None], keep[None, :]])

    functions = (
        lambda L: chol_delete(L, jnp.array(1)),
        lambda L: Cholesky.from_factor(L).delete(jnp.array(1)).factor,
        reference,
    )
    results = [
        jax.jit(jax.value_and_grad(lambda L: jnp.sum(fn(L) ** 3)))(L)
        for fn in functions
    ]
    tol = 1e-11 if jax.config.x64_enabled else 2e-5
    for result in results[:-1]:
        for actual, expected in zip(result, results[-1], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
