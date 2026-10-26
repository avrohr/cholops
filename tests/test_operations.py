"""Append, signed updates, deletion and their values and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import (
    chol_append,
    chol_delete,
    chol_delete_append,
    chol_diag_update,
    chol_rank1,
)

X64 = jax.config.x64_enabled
RTOL = 2e-11 if X64 else 4e-5
ATOL = 2e-12 if X64 else 4e-6
AD_RTOL = 8e-10 if X64 else 3e-4
AD_ATOL = 5e-11 if X64 else 3e-5


def factor(n=5):
    rng = np.random.default_rng(42 + n)
    x = jnp.asarray(rng.normal(size=(n, n)))
    return jnp.linalg.cholesky(x @ x.T + 2 * jnp.eye(n))


def matrix(L):
    L = jnp.tril(L)
    return L @ L.T


def retained_indices(n, index):
    positions = jnp.arange(n - 1)
    return positions + (positions >= index)


def append_matrix(A, k, kappa):
    n = A.shape[0]
    result = jnp.zeros((n + 1, n + 1), dtype=A.dtype)
    return (
        result.at[:n, :n].set(A).at[:n, n].set(k).at[n, :n].set(k).at[n, n].set(kappa)
    )


def delete_matrix(A, index):
    keep = retained_indices(A.shape[0], index)
    return A[keep[:, None], keep[None, :]]


def assert_factor(result, target):
    # NumPy float64 factors the actual rounded target supplied to this check.
    target64 = np.asarray(target, dtype=np.float64)
    expected = np.linalg.cholesky(target64)
    np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result @ result.T, target64, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(jnp.triu(result, 1), 0)
    assert np.all(np.diag(result) > 0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("alpha", [-0.9, -1e-5, 0.0, 1e-5, 1.3])
def test_signed_rank1(compiled, alpha):
    L = factor()
    p = jnp.arange(1.0, 6.0)
    p = 0.7 * p / jnp.linalg.norm(p)
    u = L @ p
    operation = jax.jit(chol_rank1) if compiled else chol_rank1
    result = operation(L, u, alpha)
    # Form in float64 independently to avoid hiding update rounding error.
    l64, u64 = np.asarray(L, dtype=float), np.asarray(u, dtype=float)
    assert_factor(result, l64 @ l64.T + alpha * np.outer(u64, u64))


@pytest.mark.parametrize("index", [0, 2, 4])
@pytest.mark.parametrize("compiled", [False, True])
def test_delete_and_delete_append(index, compiled):
    L = factor()
    A = matrix(L)
    delete = jax.jit(chol_delete) if compiled else chol_delete
    replace = jax.jit(chol_delete_append) if compiled else chol_delete_append
    runtime_index = jnp.asarray(index)
    keep = np.delete(np.arange(5), index)
    retained = np.asarray(A)[np.ix_(keep, keep)]
    assert_factor(delete(L, runtime_index), retained)
    k = jnp.linspace(-0.2, 0.3, 4)
    kappa = jnp.asarray(2.0)
    expected = append_matrix(jnp.asarray(retained), k, kappa)
    assert_factor(replace(L, runtime_index, k, kappa), expected)


@pytest.mark.parametrize("index", [0, 2, 4])
@pytest.mark.parametrize("delta", [-0.8, 0.0, 1.2])
def test_diagonal_update(index, delta):
    L = factor()
    result = jax.jit(chol_diag_update)(L, jnp.asarray(index), delta)
    expected = matrix(L).at[index, index].add(delta)
    assert_factor(result, expected)
    np.testing.assert_array_equal(result[:, :index], L[:, :index])


@pytest.mark.parametrize("n", [0, 1, 5])
def test_append(n):
    L = factor(n)
    k = jnp.linspace(-0.3, 0.3, n)
    kappa = jnp.asarray(2.0)
    assert_factor(jax.jit(chol_append)(L, k, kappa), append_matrix(matrix(L), k, kappa))


def test_singleton_and_empty():
    L = jnp.asarray([[2.0]])
    assert jax.jit(chol_delete)(L, jnp.asarray(0)).shape == (0, 0)
    assert_factor(jax.jit(chol_delete_append)(L, 0, jnp.empty(0), 9.0), [[9.0]])
    assert_factor(chol_diag_update(L, 0, -1.0), [[3.0]])
    empty = jnp.empty((0, 0))
    assert jax.jit(chol_rank1)(empty, jnp.empty(0), -1.0).shape == (0, 0)


def test_singleton_derivatives():
    L = jnp.asarray([[2.0]])
    grad_L, grad_kappa = jax.jit(
        jax.grad(
            lambda factor_input, kappa: jnp.sum(
                chol_delete_append(factor_input, 0, jnp.empty(0), kappa)
            ),
            argnums=(0, 1),
        )
    )(L, jnp.asarray(3.0))
    np.testing.assert_array_equal(grad_L, 0)
    np.testing.assert_allclose(grad_kappa, 0.5 / np.sqrt(3), rtol=RTOL)
    deleted_grad = jax.grad(lambda x: jnp.sum(chol_delete(x, 0)))(L)
    np.testing.assert_array_equal(deleted_grad, 0)


def test_python_scalar_precision_in_jit():
    # Even in an x64 process, weak Python scalars must follow an explicit f32
    # factor. A typed array of a different dtype remains an input error.
    L = jnp.eye(2, dtype=jnp.float32)
    result = jax.jit(chol_diag_update)(L, jnp.asarray(0), 0.2)
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result @ result.T, [[1.2, 0.0], [0.0, 1.0]], rtol=2e-6)
    with pytest.raises(TypeError, match="same dtype"):
        chol_diag_update(L, 0, jnp.asarray(1, dtype=jnp.int32))


def test_zero_vector_derivative():
    L = factor()
    u = jnp.zeros(5)
    value, tangent = jax.jvp(
        lambda x: chol_rank1(L, x, -0.7), (u,), (jnp.ones_like(u),)
    )
    np.testing.assert_array_equal(value, L)
    np.testing.assert_array_equal(tangent, 0)


def derivative_case(name):
    L = factor()
    k = jnp.linspace(-0.2, 0.3, 5)
    if name.startswith("rank1"):
        alpha = {
            "rank1_positive_weight": 0.8,
            "rank1_negative_weight": -0.8,
            "rank1_zero_weight": 0.0,
        }[name]
        args = L, k, jnp.asarray(alpha)
        operation = chol_rank1

        def reference(factor_input, u, a):
            return jnp.linalg.cholesky(matrix(factor_input) + a * jnp.outer(u, u))
    elif name.startswith("diagonal"):
        delta = {
            "diagonal_positive": 0.8,
            "diagonal_negative": -0.8,
            "diagonal_zero": 0.0,
        }[name]
        args = L, jnp.asarray(delta)

        def operation(factor_input, d):
            return chol_diag_update(factor_input, jnp.asarray(2), d)

        def reference(factor_input, d):
            return jnp.linalg.cholesky(matrix(factor_input).at[2, 2].add(d))
    elif name == "append":
        args = L, k, jnp.asarray(2.0)
        operation = chol_append

        def reference(factor_input, k, c):
            return jnp.linalg.cholesky(append_matrix(matrix(factor_input), k, c))
    elif name == "delete":
        args = (L,)

        def operation(factor_input):
            return chol_delete(factor_input, jnp.asarray(2))

        def reference(factor_input):
            return jnp.linalg.cholesky(delete_matrix(matrix(factor_input), 2))
    elif name == "delete_append":
        args = L, k[:-1], jnp.asarray(2.0)

        def operation(factor_input, k, c):
            return chol_delete_append(factor_input, jnp.asarray(2), k, c)

        def reference(factor_input, k, c):
            return jnp.linalg.cholesky(
                append_matrix(delete_matrix(matrix(factor_input), 2), k, c)
            )

    else:
        raise ValueError(f"Unknown derivative case: {name}")

    return operation, reference, args


CASES = [
    "append",
    "delete",
    "delete_append",
    "rank1_positive_weight",
    "rank1_negative_weight",
    "rank1_zero_weight",
    "diagonal_positive",
    "diagonal_negative",
    "diagonal_zero",
]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("compiled", [False, True])
def test_jvp_and_vjp_against_dense_cholesky(name, compiled):
    operation, reference, args = derivative_case(name)
    rng = np.random.default_rng(3)
    tangents = tuple(jnp.asarray(rng.normal(size=x.shape), dtype=x.dtype) for x in args)

    def evaluate(*xs):
        value, tangent = jax.jvp(operation, xs, tangents)
        cotangent = jnp.cos(jnp.arange(value.size, dtype=value.dtype)).reshape(
            value.shape
        )
        _, pullback = jax.vjp(operation, *xs)
        return value, tangent, pullback(cotangent), cotangent

    value, tangent, cotangents, output_cotangent = (
        jax.jit(evaluate) if compiled else evaluate
    )(*args)
    expected, expected_tangent = jax.jvp(reference, args, tangents)
    _, dense_pullback = jax.vjp(reference, *args)
    expected_cotangents = dense_pullback(output_cotangent)
    np.testing.assert_allclose(value, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=AD_RTOL, atol=AD_ATOL)
    for actual, expected in zip(cotangents, expected_cotangents, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=AD_RTOL, atol=AD_ATOL)
    left = jnp.vdot(tangent, output_cotangent)
    right = sum(jnp.vdot(t, c) for t, c in zip(tangents, cotangents, strict=True))
    np.testing.assert_allclose(left, right, rtol=AD_RTOL, atol=AD_ATOL)
    # The unused triangle is not an independent input to any operation.
    np.testing.assert_array_equal(jnp.triu(cotangents[0], 1), 0)


def test_zero_weight_directional_difference():
    L = factor()
    u = jnp.linspace(-0.5, 0.4, 5)
    zero = jnp.asarray(0.0)
    _, derivative = jax.jvp(
        lambda a: chol_rank1(L, u, a), (zero,), (jnp.ones_like(zero),)
    )
    eps = 1e-4 if X64 else 1e-2
    difference = (chol_rank1(L, u, eps) - chol_rank1(L, u, -eps)) / (2 * eps)
    np.testing.assert_allclose(
        derivative, difference, rtol=1e-6 if X64 else 3e-3, atol=1e-9 if X64 else 3e-5
    )
    assert float(jnp.linalg.norm(derivative)) > 0
    _, tangent_u = jax.jvp(lambda x: chol_rank1(L, x, 0.0), (u,), (jnp.ones_like(u),))
    np.testing.assert_array_equal(tangent_u, 0)


@pytest.mark.parametrize("name", CASES)
def test_vmap_and_gradient_composition(name):
    operation, reference, args = derivative_case(name)
    # Shared continuous arguments must accumulate cotangents across the batch.
    factors = jnp.stack([args[0], args[0] * 1.1])
    mapped = jax.vmap(operation, in_axes=(0,) + (None,) * (len(args) - 1))
    dense = jax.vmap(reference, in_axes=(0,) + (None,) * (len(args) - 1))
    batch_args = (factors, *args[1:])
    np.testing.assert_allclose(
        jax.jit(mapped)(*batch_args), dense(*batch_args), rtol=RTOL, atol=ATOL
    )

    # A nonlinear sum depends on off-diagonal entries too. The squared factor
    # norm is just trace(A), whose derivative misses replacement covariances.
    def loss(*xs):
        return jnp.sum(jnp.sin(mapped(*xs)))

    def dense_loss(*xs):
        return jnp.sum(jnp.sin(dense(*xs)))

    argnums = tuple(range(len(args)))
    actual = jax.jit(jax.grad(loss, argnums=argnums))(*batch_args)
    expected = jax.grad(dense_loss, argnums=argnums)(*batch_args)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=AD_RTOL, atol=AD_ATOL)


def test_dynamic_indices_in_batch():
    L = factor()
    indices = jnp.asarray([0, 2, 4])
    k = jnp.linspace(-0.2, 0.3, 4)
    replacements = jax.jit(jax.vmap(chol_delete_append, in_axes=(None, 0, None, None)))(
        L, indices, k, 2.0
    )
    for i, result in zip(np.asarray(indices), replacements, strict=True):
        assert_factor(result, append_matrix(delete_matrix(matrix(L), int(i)), k, 2.0))


@pytest.mark.parametrize("margin", [0.1, 0.001])
def test_downdate_near_spd_boundary(margin):
    # Control the exact SPD margin using ||solve(L, u)||^2 = 1-margin.
    L = factor()
    direction = jnp.asarray([1.0, -2.0, 3.0, -1.0, 2.0])
    u = L @ (direction / jnp.linalg.norm(direction) * jnp.sqrt(1 - margin))
    result = jax.jit(chol_rank1)(L, u, -1.0)
    l64, u64 = np.asarray(L, dtype=float), np.asarray(u, dtype=float)
    target = l64 @ l64.T - np.outer(u64, u64)
    residual = np.linalg.norm(np.asarray(result) @ np.asarray(result).T - target)
    assert residual / np.linalg.norm(target) < (1e-12 if X64 else 3e-6)
    # Forward factor sensitivity increases as the SPD boundary approaches.
    np.testing.assert_allclose(
        result,
        np.linalg.cholesky(target),
        rtol=2e-10 if X64 else 2e-3,
        atol=2e-11 if X64 else 2e-4,
    )


def test_non_spd_and_invalid_data_are_visible_under_jit_and_vmap():
    L = jnp.eye(3)
    u = jnp.asarray([1.0, 0.0, 0.0])
    result = jax.jit(jax.vmap(chol_rank1, in_axes=(None, None, 0)))(
        L, u, jnp.asarray([0.2, -1.0, -1.1])
    )
    assert np.isfinite(result[0]).all()
    assert np.isnan(result[1:]).all()
    assert np.isnan(jax.jit(chol_append)(L, jnp.ones(3), 1.0)).all()
    assert np.isnan(jax.jit(chol_diag_update)(L, jnp.asarray(1), -2.0)).all()
    assert not np.isfinite(jax.jit(chol_delete_append)(L, 1, jnp.ones(2), 0.0)).all()
    assert np.isnan(jax.jit(chol_rank1)(L, u.at[0].set(jnp.inf), 0.0)).all()
    # Consuming an otherwise unchanged factor entry must not hide failure.
    selected = jax.jit(
        lambda factor_input: chol_diag_update(factor_input, 2, -2.0)[0, 0]
    )(L)
    assert np.isnan(selected)


@pytest.mark.parametrize("index", [-1, 3])
def test_invalid_runtime_indices(index):
    L = jnp.eye(3)
    i = jnp.asarray(index)
    assert np.isnan(jax.jit(chol_diag_update)(L, i, 0.1)).all()
    with pytest.raises(ValueError, match="index"):
        chol_delete(L, index)


def test_unused_upper_triangle_is_ignored():
    L = factor()
    dirty = L.at[jnp.triu_indices(5, 1)].set(jnp.nan)
    u = jnp.ones(5)
    np.testing.assert_allclose(chol_rank1(dirty, u), chol_rank1(L, u))


def test_invalid_tensor_shapes_and_dtypes_raise():
    L = factor()
    with pytest.raises(ValueError, match="square"):
        chol_rank1(jnp.ones((2, 3)), jnp.ones(2))
    with pytest.raises(ValueError, match="shape"):
        chol_delete_append(L, 1, jnp.ones(5), 1.0)
    with pytest.raises(TypeError, match="integer scalar"):
        chol_delete(L, jnp.asarray(1.0))
    with pytest.raises(TypeError, match="float32 or float64"):
        chol_rank1(jnp.eye(2, dtype=jnp.complex64), jnp.ones(2))
    with pytest.raises(ValueError, match="empty"):
        chol_delete(jnp.empty((0, 0)), 0)
