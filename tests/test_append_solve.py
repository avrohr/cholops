"""Coefficient reuse agrees with a fresh solve, including its derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import Cholesky, PermutedCholesky, cholesky

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 2e-10 if jax.config.x64_enabled else 2e-4


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("cls", [Cholesky, PermutedCholesky])
@pytest.mark.parametrize("n,r", [(0, 2), (4, 0), (4, 1), (4, 3)])
def test_append_solve_factor_coefficients_and_derivatives(cls, n, r):
    X = jnp.sin(jnp.arange((n + r) * 3, dtype=DTYPE)).reshape(n + r, 3) / 3
    rhs = jnp.linspace(-0.7, 0.8, n + r, dtype=DTYPE)
    order = jnp.arange(n)[::-1]

    def candidate(X, rhs):
        A = jnp.eye(n + r, dtype=DTYPE) + X @ X.T
        if cls is Cholesky:
            factor = cls.from_matrix(A[:n, :n])
        else:
            L = cholesky(A[:n, :n][order[:, None], order])
            factor = cls(L, order, jnp.argsort(order))
        updated, coefficients = factor.append_solve(
            A[:n, n:], A[n:, n:], solution=factor.solve(rhs[:n]), rhs=rhs
        )
        return coefficients, updated.logdet(), updated.solve(rhs)

    def reference(X, rhs):
        A = jnp.eye(n + r, dtype=DTYPE) + X @ X.T
        coefficients = jnp.linalg.solve(A, rhs)
        return coefficients, jnp.linalg.slogdet(A)[1], coefficients

    args = X, rhs
    close(jax.jit(candidate)(*args), reference(*args))
    directions = jax.tree.map(lambda x: jnp.cos(x) / 7, args)
    close(
        jax.jit(lambda *a: jax.jvp(candidate, a, directions))(*args),
        jax.jvp(reference, args, directions),
    )

    def objective(fn, *a):
        coefficients, logdet, _ = fn(*a)
        return logdet + jnp.sin(jnp.arange(n + r)) @ coefficients

    close(
        jax.jit(jax.grad(lambda *a: objective(candidate, *a), (0, 1)))(*args),
        jax.jit(jax.grad(lambda *a: objective(reference, *a), (0, 1)))(*args),
    )


def test_append_solve_batched_inputs():
    K = jnp.array([[[0.1], [0.2]], [[-0.2], [0.3]]], DTYPE)
    C = jnp.array([[[1.3]], [[1.5]]], DTYPE)
    rhs = jnp.array([[0.4, -0.5, 0.8], [-0.3, 0.6, 0.2]], DTYPE)
    factor = PermutedCholesky.from_matrix(jnp.eye(2, dtype=DTYPE))

    def update(K, C, rhs):
        return factor.append_solve(K, C, solution=rhs[:2], rhs=rhs)[1]

    def reference(K, C, rhs):
        return jnp.linalg.solve(jnp.block([[jnp.eye(2), K], [K.T, C]]), rhs)

    close(jax.jit(jax.vmap(update))(K, C, rhs), jax.vmap(reference)(K, C, rhs))
