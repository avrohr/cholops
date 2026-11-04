"""Positive block updates and their consumers against dense factorization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import Cholesky, PermutedCholesky, chol_update

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 3e-10 if jax.config.x64_enabled else 5e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def fixture(n, rank):
    rng = np.random.default_rng(n + rank)
    L = np.tril(rng.normal(size=(n, n)) / (4 * n**0.5))
    np.fill_diagonal(L, np.linspace(0.9, 1.3, n))
    U = rng.normal(size=(n, rank)) / n**0.5
    return jnp.asarray(L, DTYPE), jnp.asarray(U, DTYPE)


def dense(L, U):
    L = jnp.tril(L)
    return jnp.linalg.cholesky(L @ L.T + U @ U.T)


@pytest.mark.parametrize("n,rank", [(33, 1), (35, 8), (7, 11)])
def test_tensor_values_and_derivatives(n, rank):
    L, U = fixture(n, rank)
    dL = jnp.tril(jnp.sin(jnp.arange(n * n, dtype=DTYPE).reshape(n, n))) / n
    dU = jnp.cos(U) / n
    close(
        jax.jit(lambda L, U: jax.jvp(chol_update, (L, U), (dL, dU)))(L, U),
        jax.jvp(dense, (L, U), (dL, dU)),
    )
    weights = jnp.cos(jnp.arange(n * n, dtype=DTYPE).reshape(n, n)) / n
    close(
        jax.jit(jax.grad(lambda L, U: jnp.sum(chol_update(L, U) * weights), (0, 1)))(
            L, U
        ),
        jax.grad(lambda L, U: jnp.sum(dense(L, U) * weights), (0, 1))(L, U),
    )


@pytest.mark.parametrize("rank", [1, 8])
def test_conditioning_improving_updates_with_mixed_columns(rank):
    # The update repairs small pivots. Rotating its columns leaves U @ U.T
    # unchanged, but exposes accuracy loss from forming prefix Gram matrices.
    delta, c = 0.01, 1 / np.sqrt(2)
    q = np.sqrt(1 + delta**2)
    L = jnp.asarray(np.kron(np.eye(rank), [[delta / q, 0], [c, c]]), DTYPE)
    U = jnp.asarray(np.kron(np.eye(rank), [[1 / q], [0]]), DTYPE)
    rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(rank, rank)))
    mixed = U @ jnp.asarray(rotation, DTYPE)
    close(jax.jit(chol_update)(L, mixed), chol_update(L, U))
    tangents = jnp.tril(jnp.cos(L)) / 20, jnp.sin(mixed) / 20
    close(
        jax.jit(lambda L, U: jax.jvp(chol_update, (L, U), tangents))(L, mixed),
        jax.jvp(dense, (L, mixed), tangents),
    )
    weights = jnp.sin(jnp.arange(L.size, dtype=DTYPE) + 0.3).reshape(L.shape)
    close(
        jax.jit(jax.grad(lambda L, U: jnp.sum(chol_update(L, U) * weights), (0, 1)))(
            L, mixed
        ),
        jax.grad(lambda L, U: jnp.sum(dense(L, U) * weights), (0, 1))(L, mixed),
    )


@pytest.mark.parametrize("n,rank", [(0, 0), (0, 3), (9, 0), (9, 3)])
def test_empty_and_zero_updates_preserve_the_factor_and_its_tangent(n, rank):
    L = jnp.eye(n, dtype=DTYPE) * 1.2
    U = jnp.zeros((n, rank), DTYPE)
    dL = jnp.tril(jnp.ones_like(L)) / 10
    dU = jnp.ones_like(U) / 10
    value, tangent = jax.jit(lambda L, U: jax.jvp(chol_update, (L, U), (dL, dU)))(L, U)
    close(value, L)
    close(tangent, dL)


def managed(container, L):
    if container is Cholesky:
        return container.from_factor(L)
    order = jnp.asarray(np.random.default_rng(41).permutation(L.shape[0]))
    A = L @ L.T
    physical = jnp.linalg.cholesky(A[order][:, order])
    return container(physical, order, jnp.argsort(order))


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
@pytest.mark.parametrize("rank", [1, 8])
def test_factor_update_uses_logical_rows_and_preserves_order(container, rank):
    L, U = fixture(35, rank)
    rhs = jnp.sin(jnp.arange(35, dtype=DTYPE))
    initial = managed(container, L)
    updated = jax.jit(lambda factor, U: factor.update(U))(initial, U)
    expected = L @ L.T + U @ U.T
    close(updated.solve(rhs), jnp.linalg.solve(expected, rhs))
    close(initial.solve(rhs), jnp.linalg.solve(L @ L.T, rhs))
    if container is PermutedCholesky:
        np.testing.assert_array_equal(updated.order, initial.order)
        np.testing.assert_array_equal(updated.inverse_order, initial.inverse_order)
        close(
            updated.physical_factor,
            jnp.linalg.cholesky(expected[initial.order][:, initial.order]),
        )
    else:
        close(updated.factor, jnp.linalg.cholesky(expected))

    def candidate(L, U):
        factor = managed(container, jnp.tril(L)).update(U)
        return factor.logdet() + rhs @ factor.solve(rhs)

    def reference(L, U):
        L = jnp.tril(L)
        A = L @ L.T + U @ U.T
        return jnp.linalg.slogdet(A)[1] + rhs @ jnp.linalg.solve(A, rhs)

    close(
        jax.jit(jax.value_and_grad(candidate, (0, 1)))(L, U),
        jax.value_and_grad(reference, (0, 1))(L, U),
    )
    close(
        jax.jit(lambda L, U: jax.jvp(candidate, (L, U), (L / 10, U / 7)))(L, U),
        jax.jvp(reference, (L, U), (L / 10, U / 7)),
    )


def test_batched_updates_and_derivatives_with_a_shared_factor():
    L, U = fixture(9, 3)
    updates = jnp.stack((U, -0.6 * U))
    candidate = jax.vmap(chol_update, in_axes=(None, 0))
    reference = jax.vmap(dense, in_axes=(None, 0))
    close(
        jax.jit(lambda L, U: jax.jvp(candidate, (L, U), (L / 10, U / 7)))(L, updates),
        jax.jvp(reference, (L, updates), (L / 10, updates / 7)),
    )
    close(
        jax.jit(jax.grad(lambda L, U: jnp.sin(candidate(L, U)).sum(), (0, 1)))(
            L, updates
        ),
        jax.grad(lambda L, U: jnp.sin(reference(L, U)).sum(), (0, 1))(L, updates),
    )


@pytest.mark.parametrize("n,rank", [(17, 1), (35, 3)])
def test_cached_update_linearization_with_batched_directions_and_transpose(n, rank):
    L, U = fixture(n, rank)
    directions = (
        jnp.tril(jnp.stack((jnp.sin(L), jnp.cos(L + 0.4)))) / n,
        jnp.stack((jnp.cos(U), jnp.sin(U + 0.7))) / n,
    )
    _, push = jax.linearize(chol_update, L, U)
    close(
        jax.jit(jax.vmap(push))(*directions),
        jax.vmap(lambda dL, dU: jax.jvp(dense, (L, U), (dL, dU))[1])(*directions),
    )
    weights = jnp.cos(jnp.arange(n * n, dtype=DTYPE).reshape(n, n)) / n
    transpose = jax.linear_transpose(push, jnp.zeros_like(L), jnp.zeros_like(U))
    close(jax.jit(transpose)(weights), jax.vjp(dense, L, U)[1](weights))


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
def test_repeated_factor_updates_in_a_differentiated_scan(container):
    L, U = fixture(9, 3)
    updates = jnp.stack((U, -0.7 * U, 0.3 * U))
    rhs = jnp.cos(jnp.arange(9, dtype=DTYPE))

    def run(L, updates, dense=False):
        L = jnp.tril(L)
        initial = L @ L.T if dense else managed(container, L)

        def step(state, U):
            if dense:
                state = state + U @ U.T
                score = jnp.linalg.slogdet(state)[1] + rhs @ jnp.linalg.solve(
                    state, rhs
                )
            else:
                state = state.update(U)
                score = state.logdet() + rhs @ state.solve(rhs)
            return state, score

        _, scores = jax.lax.scan(jax.checkpoint(step), initial, updates)
        return scores.sum()

    def reference(L, U):
        return run(L, U, dense=True)

    close(
        jax.jit(jax.value_and_grad(run, (0, 1)))(L, updates),
        jax.value_and_grad(reference, (0, 1))(L, updates),
    )
    close(
        jax.jit(lambda L, U: jax.jvp(run, (L, U), (L / 10, U / 7)))(L, updates),
        jax.jvp(reference, (L, updates), (L / 10, updates / 7)),
    )
