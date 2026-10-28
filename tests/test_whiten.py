"""Whitened coordinates and reusable inverse covariance reductions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import Cholesky, PermutedCholesky

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 2e-11 if jax.config.x64_enabled else 2e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def wrap(container, L, order):
    if container is Cholesky:
        return container.from_factor(L)
    return container(L, order, jnp.argsort(order))


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
@pytest.mark.parametrize("compiled", [False, True])
def test_whiten_coordinates_and_covariance_reductions(container, compiled):
    X = np.sin(np.arange(12).reshape(4, 3)) / 2
    A = np.eye(4) + X @ X.T
    order = np.arange(4) if container is Cholesky else np.array([2, 0, 3, 1])
    L = jnp.asarray(np.linalg.cholesky(A[np.ix_(order, order)]), DTYPE)
    state = wrap(container, L, jnp.asarray(order))
    B = jnp.asarray(np.cos(np.arange(12).reshape(4, 3)), DTYPE)
    C = jnp.asarray([[0.3], [-0.2], [0.7], [0.1]], DTYPE)

    def whiten(state, B, C):
        return state.whiten(B), state.whiten(C)

    W, V = (jax.jit(whiten) if compiled else whiten)(state, B, C)
    # Gram identities alone would miss an unwanted output permutation.
    close(L @ W, B[order])
    close(L @ V, C[order])
    reference = np.asarray(B).T @ np.linalg.solve(A, np.asarray(B))
    close(W.T @ W, reference)
    close(jnp.sum(W * W, axis=0), np.diag(reference))
    close(state.quadratic(B), np.diag(reference))
    close(W.T @ V, np.asarray(B).T @ np.linalg.solve(A, np.asarray(C)))


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
def test_whiten_batched_forward_and_reverse_derivatives(container):
    L = jnp.array(
        [
            [1.4, 0.2, 0.3, 0.1],
            [0.2, 1.2, 0.1, 0.3],
            [-0.1, 0.3, 1.5, 0.2],
            [0.4, -0.2, 0.1, 1.3],
        ],
        dtype=DTYPE,
    )
    B = jnp.sin(jnp.arange(12, dtype=DTYPE).reshape(4, 3))
    weights = jnp.cos(B)
    orders = jnp.array([[2, 0, 3, 1], [1, 3, 0, 2]])
    if container is Cholesky:
        orders = jnp.tile(jnp.arange(4), (2, 1))

    def actual(L, B, order):
        return wrap(container, L, order).whiten(B)

    def reference(L, B, order):
        # General dense solve also checks that the unused upper triangle and
        # its tangent do not affect whitening.
        return jnp.linalg.solve(jnp.tril(L), B[order])

    def evaluate(fn):
        def derivatives(L, B, order):
            forward = jax.jvp(
                lambda L, B: fn(L, B, order), (L, B), (jnp.cos(L) / 20, B / 10)
            )
            reverse = jax.grad(
                lambda L, B: jnp.sum(weights * fn(L, B, order)), argnums=(0, 1)
            )(L, B)
            return forward, reverse

        return jax.jit(jax.vmap(derivatives))(
            jnp.stack((L, 1.1 * L)), jnp.stack((B, B + 0.2)), orders
        )

    close(evaluate(actual), evaluate(reference))


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
@pytest.mark.parametrize("n,r", [(0, 1), (3, 0)])
def test_whiten_empty_factor_or_columns(container, n, r):
    state = container.from_factor(jnp.eye(n, dtype=DTYPE))
    B = jnp.empty((n, r), DTYPE)
    W = jax.jit(lambda state, B: state.whiten(B))(state, B)
    assert W.shape == B.shape
    close(W.T @ W, jnp.zeros((r, r), DTYPE))


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
def test_whiten_requires_a_matrix_in_logical_order(container):
    state = wrap(container, jnp.eye(4, dtype=DTYPE), jnp.array([2, 0, 3, 1]))
    for shape in ((4,), (3, 1), (4, 1, 1)):
        with pytest.raises(ValueError):
            state.whiten(jnp.ones(shape, DTYPE))
