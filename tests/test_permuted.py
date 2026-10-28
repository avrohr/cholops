"""Canonical caller semantics across physical reorderings."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import Cholesky, PermutedCholesky, cholesky

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 3e-10 if jax.config.x64_enabled else 8e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("n", [1, 8])
def test_logical_replacements_and_consumers(n):
    rng = np.random.default_rng(24)
    X = jnp.asarray(rng.normal(size=(n, 3)) / 2, DTYPE)
    A = jnp.eye(n, dtype=DTYPE) + X @ X.T
    state = PermutedCholesky.from_matrix(A)
    rhs = jnp.asarray(rng.normal(size=(n, 3)), DTYPE)
    for i in [0, n // 2, max(n - 2, 0), n - 1, 0, 0, n // 2]:
        new = jnp.asarray(rng.normal(size=3) / 2, DTYPE)
        keep = jnp.arange(n - 1) + (jnp.arange(n - 1) >= i)
        state = jax.jit(lambda s, i, k, c: s.replace(i, k, c))(
            state, jnp.asarray(i), X[keep] @ new, 1 + new @ new
        )
        X = X.at[i].set(new)
        A = jnp.eye(n, dtype=DTYPE) + X @ X.T
        np.testing.assert_array_equal(state.order[state.inverse_order], np.arange(n))
        physical_A = state.physical_factor @ state.physical_factor.T
        close(physical_A[state.inverse_order[:, None], state.inverse_order[None, :]], A)
        for b in (rhs[:, 0], rhs):
            solved = jnp.linalg.solve(A, b)
            close(jax.jit(lambda s, b: s.solve(b))(state, b), solved)
            close(state.quadratic(b), jnp.sum(b * solved, axis=0))
        close(state.logdet(), jnp.linalg.slogdet(A)[1])


@pytest.mark.parametrize("container", [Cholesky, PermutedCholesky])
def test_scan_gradients_and_batching(container):
    rng = np.random.default_rng(9)
    n, steps = 7, 12
    X = jnp.asarray(rng.normal(size=(n, 3)) / 3, DTYPE).at[0].set(0)
    incoming = jnp.asarray(rng.normal(size=(steps, 3)) / 3, DTYPE)
    indices = jnp.asarray([0, 3, 5, 6, 0, 0] * 2)
    rhs = jnp.asarray(rng.normal(size=(n, 2)), DTYPE)

    def sequence(X, incoming, scale, rhs, dense=False):
        A = jnp.eye(n, dtype=DTYPE) + scale * X @ X.T
        state = A if dense else container.from_matrix(A)

        def step(carry, request):
            X, state = carry
            i, new = request
            keep = jnp.arange(n - 1) + (jnp.arange(n - 1) >= i)
            k, c = scale * (X[keep] @ new), 1 + scale * (new @ new)
            X = X.at[i].set(new)
            if dense:
                state = jnp.eye(n, dtype=DTYPE) + scale * X @ X.T
                loss = jnp.linalg.slogdet(state)[1] + jnp.sum(
                    rhs * jnp.linalg.solve(state, rhs)
                )
            else:
                state = state.replace(i, k, c)
                loss = state.logdet() + jnp.sum(state.quadratic(rhs))
            return (X, state), loss

        return jnp.mean(jax.lax.scan(step, (X, state), (indices, incoming))[1])

    args = X, incoming, jnp.asarray(0.7, DTYPE), rhs
    actual = jax.jit(jax.value_and_grad(sequence, argnums=(0, 1, 2, 3)))
    expected = jax.jit(
        jax.value_and_grad(lambda *a: sequence(*a, dense=True), argnums=(0, 1, 2, 3))
    )
    close(actual(*args), expected(*args))
    close(
        jax.jit(jax.vmap(actual, (0, None, None, None)))(
            jnp.stack((X, 1.1 * X)), *args[1:]
        ),
        jax.jit(jax.vmap(expected, (0, None, None, None)))(
            jnp.stack((X, 1.1 * X)), *args[1:]
        ),
    )


def test_from_matrix_preserves_covariance_derivatives():
    A = jnp.array([[2.0, 0.3], [0.3, 1.0]], dtype=DTYPE)
    state = jax.jit(PermutedCholesky.from_matrix)(A)
    close(state.physical_factor, jnp.linalg.cholesky(A))
    # The constructor must keep the covariance derivative, including asymmetric
    # perturbations under the documented symmetric-input convention.
    G = jnp.array([[0.2, 8.0], [-0.7, 0.5]], dtype=DTYPE)
    actual = jax.jit(
        jax.grad(lambda A: jnp.sum(PermutedCholesky.from_matrix(A).physical_factor * G))
    )(A)
    expected = jax.grad(lambda A: jnp.sum(jnp.linalg.cholesky(A) * G))(A)
    close(actual, expected)
    _, tangent = jax.jvp(lambda a: PermutedCholesky.from_matrix(a).logdet(), (A,), (A,))
    close(tangent, jnp.asarray(A.shape[0], DTYPE))


def test_constructor_and_consumers_reject_incompatible_inputs():
    state = PermutedCholesky.from_matrix(jnp.eye(2, dtype=DTYPE))
    with pytest.raises(ValueError):
        PermutedCholesky.from_matrix(jnp.ones((2, 2, 2), DTYPE))
    with pytest.raises(ValueError):
        state.solve(jnp.ones((3,), DTYPE))
    with pytest.raises(ValueError):
        state.quadratic(jnp.ones((2, 1, 1), DTYPE))
    with pytest.raises(TypeError):
        state.solve(jnp.ones((2,), dtype=jnp.int32))


def reordered_state(A, order=(2, 3, 4, 1, 0)):
    """Represent the five-variable fixture in a specified physical order."""
    order = jnp.asarray(order)
    physical = A[order[:, None], order]
    return PermutedCholesky(cholesky(physical), order, jnp.argsort(order))


def logical_matrix(state):
    physical = state.physical_factor @ state.physical_factor.T
    return physical[state.inverse_order[:, None], state.inverse_order[None, :]]


def apply_method(state, method, index, args):
    if method == "delete":
        return state.delete(index[None])
    if method == "delete_append":
        return state.delete_append(index[None], args[0][:, None], args[1].reshape(1, 1))
    if method == "diag_update":
        return getattr(state, method)(index, *args)
    return getattr(state, method)(*args)


def dense_modification(A, method, index, args):
    if method in ("delete", "delete_append"):
        keep = jnp.arange(A.shape[0] - 1)
        keep = keep + (keep >= index)
        A = A[keep[:, None], keep[None, :]]
        if method == "delete":
            return A
    if method == "append":
        K, C = args
        return jnp.block([[A, K], [K.T, (C + C.T) / 2]])
    if method == "delete_append":
        k, c = args
        return jnp.block([[A, k[:, None]], [k[None, :], c.reshape(1, 1)]])
    if method == "diag_update":
        return A.at[index, index].add(args[0])
    u = args[0]
    weight = args[1] if method == "rank1" else 1
    return A + weight * jnp.outer(u, u)


def modification_fixture(method, weight=0.2):
    X = jnp.sin(jnp.arange(15, dtype=DTYPE).reshape(5, 3)) / 3
    A = jnp.eye(5, dtype=DTYPE) + X @ X.T
    vector = jnp.linspace(-0.15, 0.2, 5, dtype=DTYPE)
    scalar = jnp.asarray(weight, DTYPE)
    if method == "append":
        args = (
            jnp.stack((vector, -vector / 2), axis=1),
            jnp.array([[1.5, 0.1], [0.1, 1.2]], DTYPE),
        )
    elif method == "delete_append":
        args = (vector[:-1], jnp.asarray(1.5, DTYPE))
    elif method == "delete":
        args = ()
    elif method == "diag_update":
        args = (scalar,)
    elif method == "rank1":
        args = (vector, scalar)
    else:
        args = (vector,)
    return A, args


def check_permutations(state):
    n = state.physical_factor.shape[0]
    np.testing.assert_array_equal(state.order[state.inverse_order], np.arange(n))
    np.testing.assert_array_equal(state.inverse_order[state.order], np.arange(n))


@pytest.mark.parametrize(
    "method,weight",
    [(name, 1.0) for name in ("append", "delete", "delete_append", "rank1_positive")]
    + [
        (name, weight)
        for name in ("rank1", "diag_update")
        for weight in (-0.3, 0.0, 0.4)
    ],
)
def test_modifications_after_reordering_and_all_input_derivatives(method, weight):
    A, args = modification_fixture(method, weight)
    index = jnp.asarray(2)
    state = reordered_state(A)
    updated = jax.jit(lambda s, i, a: apply_method(s, method, i, a))(state, index, args)
    target = dense_modification(A, method, index, args)
    check_permutations(updated)
    close(logical_matrix(updated), target)
    close(logical_matrix(state), A)

    n = target.shape[0]
    rhs = jnp.linspace(-0.4, 0.7, n * 2, dtype=DTYPE).reshape(n, 2)

    def actual(A, args, rhs):
        state = apply_method(reordered_state((A + A.T) / 2), method, index, args)
        return (
            state.logdet()
            + jnp.sum(state.quadratic(rhs))
            + jnp.sum(jnp.sin(rhs) * state.solve(rhs))
        )

    def reference(A, args, rhs):
        B = dense_modification((A + A.T) / 2, method, index, args)
        solved = jnp.linalg.solve(B, rhs)
        return (
            jnp.linalg.slogdet(B)[1]
            + jnp.sum(rhs * solved)
            + jnp.sum(jnp.sin(rhs) * solved)
        )

    primals = A, args, rhs
    got = jax.jit(jax.value_and_grad(actual, argnums=(0, 1, 2)))(*primals)
    expected = jax.jit(jax.value_and_grad(reference, argnums=(0, 1, 2)))(*primals)
    close(got, expected)
    tangents = jax.tree.map(lambda a: jnp.cos(a) / 10, primals)
    close(
        jax.jit(lambda *a: jax.jvp(actual, a, tangents))(*primals),
        jax.jit(lambda *a: jax.jvp(reference, a, tangents))(*primals),
    )
    if method in ("rank1", "diag_update") and weight == 0:
        assert abs(float(got[1][1][-1])) > 1e-4


@pytest.mark.parametrize(
    "method",
    ["append", "delete", "delete_append", "diag_update", "rank1", "rank1_positive"],
)
def test_batched_modifications_with_runtime_indices(method):
    A, args = modification_fixture(method)
    matrices = jnp.stack((A, 1.1 * A, 1.2 * A))
    # Different storage orders must still give the correct logical results.
    orders = jnp.array([[2, 3, 4, 1, 0], [4, 2, 0, 3, 1], [1, 0, 3, 4, 2]])
    states = jax.vmap(reordered_state)(matrices, orders)
    indices = jnp.array([0, 2, 4])

    def actual(states, args):
        out = jax.vmap(lambda s, i: apply_method(s, method, i, args))(states, indices)
        return jax.vmap(logical_matrix)(out)

    def reference(matrices, args):
        return jax.vmap(lambda A, i: dense_modification(A, method, i, args))(
            matrices, indices
        )

    close(jax.jit(actual)(states, args), jax.jit(reference)(matrices, args))
    # Shared update inputs must collect gradients from all batch members.
    close(
        jax.jit(jax.grad(lambda a: jnp.sum(actual(states, a))))(args),
        jax.jit(jax.grad(lambda a: jnp.sum(reference(matrices, a))))(args),
    )


def test_size_changes_and_repeated_mixed_updates():
    A, _ = modification_fixture("append")
    state = reordered_state(A)
    # Exercise the maps repeatedly, including deletion of the physical last slot.
    for index in (0, 3, 1, 0, 0):
        state = jax.jit(lambda s, i: s.delete(jnp.asarray([i])))(
            state, jnp.asarray(index)
        )
        A = jnp.asarray(
            np.delete(np.delete(np.asarray(A), index, axis=0), index, axis=1)
        )
        check_permutations(state)
        close(logical_matrix(state), A)
    assert state.physical_factor.shape == (0, 0)
    empty_vector = jnp.empty((0,), DTYPE)
    state = state.rank1(empty_vector, -0.2).rank1_positive(empty_vector)
    singleton = jax.jit(
        lambda s: s.append(
            empty_vector[:, None], jnp.array([[2.0]], DTYPE)
        ).delete_append(
            jnp.asarray([0]),
            jnp.reshape(empty_vector, (-1, 1)),
            jnp.asarray([[3.0]]),
        )
    )(state)
    check_permutations(singleton)
    close(logical_matrix(singleton), jnp.asarray([[3.0]], DTYPE))
    for diagonal in (1.5, 2.0, 2.5):
        k = jnp.full((A.shape[0],), 0.1, DTYPE)
        state = jax.jit(lambda s, k, c: s.append(k, c))(
            state, k[:, None], jnp.array([[diagonal]], DTYPE)
        )
        A = jnp.block([[A, k[:, None]], [k[None, :], jnp.asarray([[diagonal]], DTYPE)]])
        check_permutations(state)
        close(logical_matrix(state), A)
    # delete_append changes logical order; replace preserves the resulting slots.
    state = state.delete_append(
        jnp.asarray([1]),
        jnp.reshape(jnp.array([0.1, 0.2], DTYPE), (-1, 1)),
        jnp.asarray([[1.3]]),
    )
    A = dense_modification(
        A,
        "delete_append",
        jnp.asarray(1),
        (jnp.array([0.1, 0.2], DTYPE), jnp.asarray(1.3, DTYPE)),
    )
    state = state.replace(0, jnp.array([0.2, 0.1], DTYPE), 1.6)
    A = A.at[0, :].set(jnp.array([1.6, 0.2, 0.1], DTYPE))
    A = A.at[:, 0].set(jnp.array([1.6, 0.2, 0.1], DTYPE))
    check_permutations(state)
    close(state.solve(jnp.ones(3, DTYPE)), jnp.linalg.solve(A, jnp.ones(3, DTYPE)))


def test_mixed_modifications_in_scan_and_gradients():
    A, _ = modification_fixture("rank1")
    updates = jnp.sin(jnp.arange(20, dtype=DTYPE).reshape(4, 5)) / 10
    weights = jnp.asarray([-0.2, 0.0, 0.3, 0.0], DTYPE)
    rhs = jnp.linspace(-0.3, 0.6, 5, dtype=DTYPE)
    indices = jnp.array([1, 3, 0, 4])

    def objective(A, updates, weights, rhs, *, dense=False):
        A = (A + A.T) / 2
        state = A if dense else reordered_state(A)

        def step(state, inputs):
            index, u, weight = inputs
            diagonal = jnp.asarray(1.5, DTYPE)
            if dense:
                state = state + weight * jnp.outer(u, u)
                state = state.at[index, index].add(0.1)
                state = dense_modification(
                    state, "delete_append", index, (u[:-1], diagonal)
                )
                score = jnp.linalg.slogdet(state)[1] + rhs @ jnp.linalg.solve(
                    state, rhs
                )
            else:
                state = state.rank1(u, weight).diag_update(index, 0.1)
                state = state.delete_append(
                    jnp.asarray([index]),
                    jnp.reshape(u[:-1], (-1, 1)),
                    diagonal.reshape(1, 1),
                )
                score = state.logdet() + state.quadratic(rhs)
            return state, score

        _, scores = jax.lax.scan(step, state, (indices, updates, weights))
        return jnp.sum(scores)

    args = A, updates, weights, rhs
    close(
        jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2, 3)))(*args),
        jax.jit(
            jax.value_and_grad(
                lambda *a: objective(*a, dense=True), argnums=(0, 1, 2, 3)
            )
        )(*args),
    )
