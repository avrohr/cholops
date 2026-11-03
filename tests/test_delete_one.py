"""Explicit single deletion, including logical order and differentiated reuse."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import Cholesky, PermutedCholesky, chol_append, chol_delete_one, cholesky

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 3e-10 if jax.config.x64_enabled else 8e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def fixture(n):
    X = jnp.sin(jnp.arange(n * 3, dtype=DTYPE)).reshape(n, 3) / 3
    return jnp.eye(n, dtype=DTYPE) + X @ X.T


@pytest.mark.parametrize("n", [1, 7])
def test_tensor_all_positions_batched_values_and_derivatives(n):
    indices = jnp.arange(n)
    kept = jnp.asarray(np.stack([np.delete(np.arange(n), i) for i in range(n)]))
    L = cholesky(fixture(n)).at[jnp.triu_indices(n, 1)].set(jnp.nan)
    candidate = jax.vmap(chol_delete_one, in_axes=(None, 0))

    def reference(L):
        rows = jnp.tril(L)[kept]
        return jnp.linalg.cholesky(rows @ rows.swapaxes(-1, -2))

    dL = jnp.full(L.shape, 0.03, DTYPE)
    value, tangent = jax.jit(
        lambda L, indices: jax.jvp(lambda L: candidate(L, indices), (L,), (dL,))
    )(L, indices)
    close((value, tangent), jax.jvp(reference, (L,), (dL,)))
    G = jnp.cos(jnp.arange(value.size, dtype=DTYPE)).reshape(value.shape)
    gradient = jax.jit(jax.grad(lambda L, i: jnp.sum(candidate(L, i) * G)))(L, indices)
    close(gradient, jax.grad(lambda L: jnp.sum(reference(L) * G))(L))
    np.testing.assert_array_equal(jnp.triu(gradient, 1), 0)


def managed(cls, A):
    if cls is Cholesky:
        return cls.from_matrix(A)
    order = jnp.array([3, 0, 4, 1, 2])
    return cls(cholesky(A[order][:, order]), order, jnp.argsort(order))


@pytest.mark.parametrize("cls", [Cholesky, PermutedCholesky])
def test_factor_single_deletion_uses_logical_indices_and_consumer_derivatives(cls):
    A = fixture(5)
    indices = jnp.array([0, 2, 4])
    kept = jnp.asarray(np.stack([np.delete(np.arange(5), i) for i in [0, 2, 4]]))
    rhs = jnp.linspace(-0.3, 0.6, 4, dtype=DTYPE)

    def candidate(A, indices):
        factor = managed(cls, A)

        def score(index):
            out = factor.delete_one(index)
            return out.logdet() + rhs @ out.solve(rhs)

        return jnp.array([0.3, 0.7, -0.2], DTYPE) @ jax.vmap(score)(indices)

    def reference(A):
        A = (A + A.T) / 2
        matrices = jax.vmap(lambda k: A[k][:, k])(kept)
        scores = jax.vmap(
            lambda B: jnp.linalg.slogdet(B)[1] + rhs @ jnp.linalg.solve(B, rhs)
        )(matrices)
        return jnp.array([0.3, 0.7, -0.2], DTYPE) @ scores

    close(
        jax.jit(jax.value_and_grad(candidate))(A, indices),
        jax.value_and_grad(reference)(A),
    )
    direction = jnp.cos(A) / 20
    close(
        jax.jit(lambda A, i: jax.jvp(lambda A: candidate(A, i), (A,), (direction,)))(
            A, indices
        ),
        jax.jvp(reference, (A,), (direction,)),
    )
    out = jax.jit(lambda factor, index: factor.delete_one(index))(
        managed(cls, A), indices[1]
    )
    if cls is Cholesky:
        matrix = out.factor @ out.factor.T
    else:
        matrix = (out.physical_factor @ out.physical_factor.T)[out.inverse_order][
            :, out.inverse_order
        ]
    close(matrix, A[kept[1]][:, kept[1]])


@pytest.mark.parametrize("implementation", ["tensor", Cholesky, PermutedCholesky])
def test_checkpointed_single_delete_append_sequence(implementation):
    A = fixture(5)
    indices = jnp.array([1, 0, 4])
    kept = jnp.asarray(np.stack([np.delete(np.arange(5), i) for i in [1, 0, 4]]))
    K = jnp.full((4, 1), 0.03, DTYPE)
    C = jnp.full((1, 1), 1.3, DTYPE)

    def run(A, dense=False):
        if dense:
            initial = (A + A.T) / 2
        elif implementation == "tensor":
            initial = cholesky(A)
        else:
            initial = managed(implementation, A)

        def step(state, inputs):
            index, keep = inputs
            if dense:
                state = jnp.block([[state[keep][:, keep], K], [K.T, C]])
                score = jnp.linalg.slogdet(state)[1]
            elif implementation == "tensor":
                state = chol_append(chol_delete_one(state, index), K, C)
                score = 2 * jnp.log(jnp.diag(state)).sum()
            else:
                state = state.delete_one(index).append(K, C)
                score = state.logdet()
            return state, score

        return jax.lax.scan(jax.checkpoint(step), initial, (indices, kept))[1].sum()

    close(
        jax.jit(jax.value_and_grad(run))(A),
        jax.value_and_grad(lambda A: run(A, True))(A),
    )
    direction = jnp.cos(A) / 20
    close(
        jax.jit(lambda A: jax.jvp(run, (A,), (direction,)))(A),
        jax.jvp(lambda A: run(A, True), (A,), (direction,)),
    )
