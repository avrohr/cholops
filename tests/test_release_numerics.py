"""Release checks exercise length-dependent scans and matrix-scale derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import chol_delete, chol_rank1_positive, chol_replace
from tests.reference.gp_sequences import Case, fixture, function

pytestmark = pytest.mark.full
DTYPE = np.float64 if jax.config.x64_enabled else np.float32


def assert_relative(actual, expected, tolerance):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12) < tolerance


@pytest.mark.parametrize("n", [256, 1024])
@pytest.mark.parametrize(
    "operation,position",
    [("positive", "first")]
    + [
        (op, pos) for op in ("delete", "replace") for pos in ("first", "middle", "last")
    ],
)
def test_large_factor_and_both_derivative_modes(n, operation, position):
    # n=1024 crosses scan reduction/tile boundaries absent in tiny CI fixtures.
    rng = np.random.default_rng(73)
    X = jnp.asarray(rng.normal(size=(n, 12)) / np.sqrt(12), DTYPE)
    L = jnp.linalg.cholesky(jnp.eye(n, dtype=DTYPE) + X @ X.T)
    new = jnp.asarray(rng.normal(size=12) / np.sqrt(12), DTYPE)
    i = {"first": 0, "middle": n // 2, "last": n - 1}[position]
    keep = jnp.asarray(np.delete(np.arange(n), i))
    u = jnp.asarray(rng.normal(size=n) / np.sqrt(n), DTYPE)
    args = (
        (L, u)
        if operation == "positive"
        else (L,)
        if operation == "delete"
        else (L, X[keep] @ new, 1 + new @ new)
    )

    def candidate(*a):
        if operation == "positive":
            return chol_rank1_positive(*a)
        if operation == "delete":
            return chol_delete(a[0], jnp.asarray(i))
        return chol_replace(a[0], jnp.asarray(i), *a[1:])

    def reference(L, *parameters):
        L = jnp.tril(L)
        A = L @ L.T
        if operation == "positive":
            A = A + jnp.outer(parameters[0], parameters[0])
        elif operation == "delete":
            A = A[keep[:, None], keep[None, :]]
        else:
            k, c = parameters
            A = A.at[i, keep].set(k).at[keep, i].set(k).at[i, i].set(c)
        return jnp.linalg.cholesky(A)

    tangents = jax.tree.map(
        lambda a: jnp.asarray(rng.normal(size=a.shape), DTYPE) * 0.01, args
    )
    m = n - (operation == "delete")
    G = jnp.asarray(rng.normal(size=(m, m)), DTYPE) / n

    def evaluate(fn, *a):
        value, tangent = jax.jvp(fn, a, tangents)
        return value, tangent, jax.vjp(fn, *a)[1](G)

    actual = jax.jit(lambda *a: evaluate(candidate, *a))(*args)
    expected = jax.jit(lambda *a: evaluate(reference, *a))(*args)
    assert_relative(actual, expected, 2e-8 if jax.config.x64_enabled else 2e-3)
    lhs = jnp.vdot(G, actual[1])
    rhs = sum(jnp.vdot(g, d) for g, d in zip(actual[2], tangents, strict=True))
    np.testing.assert_allclose(
        lhs,
        rhs,
        rtol=2e-8 if jax.config.x64_enabled else 2e-3,
        atol=2e-10 if jax.config.x64_enabled else 2e-5,
    )


@pytest.mark.parametrize("scenario", ["se_clustered", "matern32_uniform"])
@pytest.mark.parametrize(
    "pattern,block,method", [("fifo", 1, "solve_scan"), ("tail", 4, "schur")]
)
def test_long_gp_sequences_and_location_gradients(scenario, pattern, block, method):
    # 128 points/64 replacements check accumulated drift and checkpoint replay.
    # The tiny Decimal tests retain the severe-conditioning anchors separately.
    case = Case(scenario, 128, 64, pattern, block)
    args = fixture(case, DTYPE, 1e-2)
    result = []
    for algorithm in (method, "recompute"):
        fn = function(case, algorithm)
        result.append(jax.jit(jax.value_and_grad(fn, (0, 1, 2, 3, 4)))(*args))
    assert_relative(*result, 2e-7 if jax.config.x64_enabled else 8e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
def test_deletion_backward_storage_stays_within_quadratic_budget(n):
    # Fusion and solver workspace can change at size thresholds. Bound each
    # compiled size directly; adjacent-size ratios need not be monotone.
    L = jax.ShapeDtypeStruct((n, n), DTYPE)
    index = jax.ShapeDtypeStruct((), jnp.asarray(0).dtype)
    fn = jax.jit(jax.grad(lambda L, i: jnp.sum(chol_delete(L, i))))
    compiled = fn.lower(L, index).compile()
    memory = compiled.memory_analysis()
    assert memory is not None
    assert memory.temp_size_in_bytes < 32 * n * n * np.dtype(DTYPE).itemsize
