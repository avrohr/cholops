"""Fixed-capacity modifications and masked consumers against dense systems."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import PaddedCholesky, PermutedCholesky, cholesky

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 3e-10 if jax.config.x64_enabled else 3e-4


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def covariance(X, active):
    X = jnp.where(active[:, None], X, 0)
    return jnp.eye(active.size, dtype=X.dtype) + X @ X.T


def managed(A, active, permuted):
    if not permuted:
        return PaddedCholesky.from_matrix(A, active=active)
    order = jnp.array([4, 0, 5, 2, 1, 3])
    factor = PermutedCholesky(
        cholesky(A[order[:, None], order]), order, jnp.argsort(order)
    )
    return PaddedCholesky(factor, active)


def represented(factor):
    L = factor.factor.physical_factor
    p = factor.factor.inverse_order
    return (L @ L.T)[p[:, None], p]


def replaced(A, active, indices, K, C):
    retained = active.at[indices].set(False)
    K = jnp.where(retained[:, None], K, 0)
    A = A.at[:, indices].set(K).at[indices, :].set(K.T)
    return A.at[indices[:, None], indices].set((C + C.T) / 2)


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("matrix_rhs", [False, True])
def test_consumers_ignore_inactive_nonfinite_entries_and_their_gradients(
    permuted, matrix_rhs
):
    active = jnp.array([True, False, True, False, False, True])
    X = jnp.sin(jnp.arange(18, dtype=DTYPE)).reshape(6, 3) / 3
    A = covariance(X, active)
    factor = managed(A, active, permuted)
    shape = (6, 2) if matrix_rhs else (6,)
    rhs = jnp.cos(jnp.arange(np.prod(shape), dtype=DTYPE)).reshape(shape)
    mask = active[:, None] if matrix_rhs else active
    rhs = jnp.where(mask, rhs, jnp.nan)
    clean = jnp.where(mask, rhs, 0)
    expected = jnp.linalg.solve(A, clean)
    close(jax.jit(factor.solve)(rhs), expected)
    close(jax.jit(factor.quadratic)(rhs), jnp.sum(clean * expected, axis=0))
    close(factor.logdet(), jnp.linalg.slogdet(A)[1])
    matrix = rhs if matrix_rhs else rhs[:, None]
    W = jax.jit(factor.whiten)(matrix)
    B = clean if matrix_rhs else clean[:, None]
    close(W.T @ W, B.T @ jnp.linalg.solve(A, B))
    np.testing.assert_array_equal(W[~active[factor.factor.order]], 0)
    np.testing.assert_array_equal(factor.solve(rhs)[~active], 0)

    def objective(rhs):
        W = factor.whiten(rhs if matrix_rhs else rhs[:, None])
        return factor.solve(rhs).sum() + factor.quadratic(rhs).sum() + (W * W).sum()

    gradient = jax.jit(jax.grad(objective))(rhs)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_array_equal(gradient[~active], 0)
    close(jax.jvp(objective, (rhs,), (jnp.ones_like(rhs),))[1], gradient.sum())


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize(
    "operation,selected",
    [("activate", [4, 1]), ("replace", [2, 1]), ("replace", [5, 2])],
    ids=["activation", "mixed-replacement", "occupied-replacement"],
)
def test_modifications_and_solution_derivatives_with_full_capacity_blocks(
    permuted, operation, selected
):
    active = jnp.array([True, False, True, False, False, True])
    indices = jnp.array(selected)
    X = jnp.sin(jnp.arange(18, dtype=DTYPE)).reshape(6, 3) / 3
    K = jnp.cos(jnp.arange(12, dtype=DTYPE)).reshape(6, 2) / 10
    used = active.at[indices].set(False)
    K = jnp.where(used[:, None], K, jnp.nan)
    C = jnp.array([[1.4, 0.2], [0.2, 1.2]], DTYPE)
    new_active = active.at[indices].set(True)
    rhs = jnp.where(new_active, jnp.linspace(-0.6, 0.8, 6, dtype=DTYPE), jnp.nan)

    def candidate(X, K, C, rhs):
        factor = managed(covariance(X, active), active, permuted)
        if operation == "activate":
            old_solution = jnp.where(active, factor.solve(rhs), jnp.nan)
            result, solution = factor.activate_solve(
                indices, K, C, solution=old_solution, rhs=rhs
            )
        else:
            result = factor.replace(indices, K, C)
            solution = result.solve(rhs)
        return represented(result), solution, result.logdet()

    def reference(X, K, C, rhs):
        A = replaced(covariance(X, active), active, indices, K, C)
        solution = jnp.linalg.solve(A, jnp.where(new_active, rhs, 0))
        return A, solution, jnp.linalg.slogdet(A)[1]

    args = X, K, C, rhs
    close(jax.jit(candidate)(*args), reference(*args))
    directions = tuple(jnp.full_like(x, 0.07) for x in args)
    close(
        jax.jit(lambda *a: jax.jvp(candidate, a, directions))(*args),
        jax.jvp(reference, args, directions),
    )

    def objective(fn, *args):
        A, solution, logdet = fn(*args)
        weights = jnp.sin(jnp.arange(36, dtype=DTYPE)).reshape(6, 6)
        return jnp.sum(A * weights) + jnp.cos(jnp.arange(6)) @ solution + logdet

    grad = jax.jit(jax.grad(lambda *a: objective(candidate, *a), (0, 1, 2, 3)))(*args)
    close(
        grad,
        jax.jit(jax.grad(lambda *a: objective(reference, *a), (0, 1, 2, 3)))(*args),
    )
    np.testing.assert_array_equal(grad[1][~used], 0)
    np.testing.assert_array_equal(grad[3][~new_active], 0)
    close(grad[2], grad[2].T)


def test_activation_batched_runtime_indices_and_activity():
    X = jnp.sin(jnp.arange(18, dtype=DTYPE)).reshape(6, 3) / 3
    masks = jnp.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0]], bool)
    indices = jnp.array([[5, 1], [2, 0]])
    K = jnp.full((6, 2), 0.03, DTYPE)
    C = jnp.eye(2, dtype=DTYPE)
    rhs = jnp.linspace(-0.3, 0.8, 6, dtype=DTYPE)

    @jax.jit
    def prepare(X, masks):
        return jax.vmap(lambda active: managed(covariance(X, active), active, True))(
            masks
        )

    @jax.jit
    def update(factors, indices):
        def one(factor, indices):
            result, solution = factor.activate_solve(
                indices, K, C, solution=factor.solve(rhs), rhs=rhs
            )
            return result.logdet() + solution @ rhs

        return jax.vmap(one)(factors, indices).sum()

    def candidate(X):
        return update(prepare(X, masks), indices)

    def reference(X):
        def update(active, indices):
            A = replaced(covariance(X, active), active, indices, K, C)
            b = jnp.where(active.at[indices].set(True), rhs, 0)
            return jnp.linalg.slogdet(A)[1] + b @ jnp.linalg.solve(A, b)

        return jax.vmap(update)(masks, indices).sum()

    close(jax.jit(jax.value_and_grad(candidate))(X), jax.value_and_grad(reference)(X))
    close(jax.jvp(candidate, (X,), (X / 5,)), jax.jvp(reference, (X,), (X / 5,)))


@pytest.mark.parametrize("n,r", [(0, 0), (5, 0), (5, 1), (5, 5)])
def test_empty_single_and_full_capacity_activation_and_deactivation(n, r):
    active = jnp.zeros(n, bool)
    factor = PaddedCholesky.from_factor(jnp.eye(n, dtype=DTYPE), active=active)
    indices = jnp.arange(r)[::-1]
    C = jnp.eye(r, dtype=DTYPE) * 2
    result, solution = jax.jit(
        lambda f: f.activate_solve(
            indices,
            jnp.full((n, r), jnp.nan, DTYPE),
            C,
            solution=jnp.full(n, jnp.nan, DTYPE),
            rhs=jnp.ones(n, DTYPE),
        )
    )(factor)
    close(solution, result.solve(jnp.ones(n, DTYPE)))
    close(represented(result), jnp.eye(n, dtype=DTYPE).at[indices, indices].set(2))
    emptied = jax.jit(lambda f: f.deactivate(indices))(result)
    close(represented(emptied), jnp.eye(n, dtype=DTYPE))
    np.testing.assert_array_equal(emptied.active, active)


def test_invalid_active_covariance_remains_a_numerical_failure():
    factor = PaddedCholesky.from_factor(
        jnp.eye(3, dtype=DTYPE), active=jnp.zeros(3, bool)
    )
    result, solution = jax.jit(
        lambda C: factor.activate_solve(
            jnp.array([1]),
            jnp.zeros((3, 1), DTYPE),
            C,
            solution=jnp.zeros(3, DTYPE),
            rhs=jnp.ones(3, DTYPE),
        )
    )(jnp.array([[-1.0]], DTYPE))
    assert not jnp.isfinite(result.logdet())
    assert not jnp.isfinite(solution[1])


def test_fixed_shape_updates_reuse_compilation_as_activity_and_order_change():
    traces = []

    @jax.jit
    def update(factor, indices, K, C, solution, rhs, queries):
        traces.append(None)
        factor, solution = factor.activate_solve(
            indices, K, C, solution=solution, rhs=rhs
        )
        W = factor.whiten(queries)
        return factor, solution, jnp.sum(W * W, axis=0)

    factor = PaddedCholesky.from_factor(
        jnp.eye(6, dtype=DTYPE), active=jnp.zeros(6, bool)
    )
    solution = jnp.zeros(6, DTYPE)
    rhs = jnp.ones(6, DTYPE)
    queries = jnp.ones((6, 2), DTYPE) / 5
    A = jnp.eye(6, dtype=DTYPE)
    active = factor.active
    for indices in (jnp.array([4, 0]), jnp.array([1, 5]), jnp.array([3, 2])):
        K, C = jnp.full((6, 2), 0.03, DTYPE), jnp.eye(2, dtype=DTYPE)
        A = replaced(A, active, indices, K, C)
        active = active.at[indices].set(True)
        factor, solution, variance_reduction = update(
            factor, indices, K, C, solution, rhs, queries
        )
        close(solution, jnp.linalg.solve(A, jnp.where(active, rhs, 0)))
        B = jnp.where(active[:, None], queries, 0)
        close(variance_reduction, jnp.sum(B * jnp.linalg.solve(A, B), axis=0))
    assert len(traces) == 1


def test_checkpointed_mixed_update_sequence_values_and_derivatives():
    X = jnp.sin(jnp.arange(24, dtype=DTYPE)).reshape(8, 3) / 3
    incoming = jnp.cos(jnp.arange(30, dtype=DTYPE)).reshape(5, 2, 3) / 3
    initial_active = jnp.array([False, True, True, False, False, False, False, False])
    slots = jnp.array([[3, 0], [7, 5], [2, 4], [0, 1], [1, 0]])
    operations = jnp.array([0, 0, 1, 2, 0])
    rhs = jnp.linspace(-0.4, 0.8, 8, dtype=DTYPE)

    def run(X, incoming, dense=False):
        factor = PaddedCholesky.from_matrix(
            covariance(X, initial_active), active=initial_active
        )
        initial = covariance(X, initial_active) if dense else factor
        solution = factor.solve(rhs)

        def step(carry, inputs):
            state, X, active, solution = carry
            operation, indices, new = inputs
            K = X @ new.T
            C = jnp.eye(2, dtype=DTYPE) + new @ new.T
            X = X.at[indices].set(new)
            next_active = active.at[indices].set(operation != 2)
            if dense:
                state = covariance(X, next_active)
                solution = jnp.linalg.solve(state, jnp.where(next_active, rhs, 0))
                score = jnp.linalg.slogdet(state)[1] + rhs @ solution
            else:

                def activate(_):
                    return state.activate_solve(
                        indices, K, C, solution=solution, rhs=rhs
                    )

                def replace(_):
                    result = state.replace(indices, K, C)
                    return result, result.solve(rhs)

                def deactivate(_):
                    result = state.deactivate(indices)
                    return result, result.solve(rhs)

                state, solution = jax.lax.switch(
                    operation, (activate, replace, deactivate), None
                )
                score = state.logdet() + rhs @ solution
            return (state, X, next_active, solution), score

        _, scores = jax.lax.scan(
            jax.checkpoint(step),
            (initial, X, initial_active, solution),
            (operations, slots, incoming),
        )
        return jnp.arange(1, 6, dtype=DTYPE) @ scores

    close(
        jax.jit(jax.value_and_grad(run, (0, 1)))(X, incoming),
        jax.jit(jax.value_and_grad(lambda X, new: run(X, new, True), (0, 1)))(
            X, incoming
        ),
    )
    directions = X / 7, incoming / 5
    close(
        jax.jit(lambda X, new: jax.jvp(run, (X, new), directions))(X, incoming),
        jax.jvp(lambda X, new: run(X, new, True), (X, incoming), directions),
    )
