import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import cholesky
from cholops.tensors.factorization import pullback
from tests.reference.decimal_ad import pullback as decimal_pullback
from tests.reference.decimal_ad import sequence as decimal_sequence
from tests.reference.gp_sequences import Case, covariance, fixture
from tests.reference.gradient_sequences import function

DTYPE = np.float64 if jax.config.x64_enabled else np.float32
TOL = 2e-11 if jax.config.x64_enabled else 4e-5


@pytest.mark.parametrize("n", [0, 1, 6])
def test_primal_and_arbitrary_cotangent(n):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, n))
    A = jnp.asarray(X @ X.T + np.eye(n), DTYPE)
    G = jnp.asarray(rng.normal(size=(n, n)), DTYPE)
    actual, reverse = jax.vjp(jax.jit(cholesky), A)
    expected, reference = jax.vjp(jax.jit(jnp.linalg.cholesky), A)
    np.testing.assert_allclose(actual, expected, rtol=TOL, atol=TOL)
    close_actual = jax.jvp(jax.jit(cholesky), (A,), (G,))[1]
    close_expected = jax.jvp(jax.jit(jnp.linalg.cholesky), (A,), (G,))[1]
    np.testing.assert_allclose(close_actual, close_expected, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(reverse(G)[0], reference(G)[0], rtol=TOL, atol=TOL)
    np.testing.assert_array_equal(reverse(G)[0], reverse(G)[0].T)


def test_batch_and_symmetric_input_convention():
    rng = np.random.default_rng(11)
    X = jnp.asarray(rng.normal(size=(2, 5, 5)), DTYPE)
    A = X @ jnp.swapaxes(X, -1, -2) + jnp.eye(5, dtype=DTYPE)
    # Native Cholesky symmetrizes its input, even for asymmetric tangents.
    A = A + 0.1 * (X - jnp.swapaxes(X, -1, -2))
    G = X

    def loss(A):
        return jnp.sum(cholesky(A) * G)

    direct = jax.jit(jax.grad(loss))(A)
    mapped = jax.jit(jax.vmap(jax.grad(lambda a, g: jnp.sum(cholesky(a) * g)), (0, 0)))(
        A, G
    )
    native = jax.jit(jax.grad(lambda a: jnp.sum(jnp.linalg.cholesky(a) * G)))(A)
    np.testing.assert_allclose(direct, mapped, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(direct, native, rtol=TOL, atol=TOL)
    tangent = jax.jit(lambda a, da: jax.jvp(cholesky, (a,), (da,))[1])(A, X)
    mapped_tangent = jax.jit(jax.vmap(lambda a, da: jax.jvp(cholesky, (a,), (da,))[1]))(
        A, X
    )
    native_tangent = jax.jvp(jnp.linalg.cholesky, (A,), (X,))[1]
    np.testing.assert_allclose(tangent, mapped_tangent, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(tangent, native_tangent, rtol=TOL, atol=TOL)
    upper = jnp.triu(jnp.ones_like(G), 1)
    np.testing.assert_array_equal(jax.vjp(cholesky, A)[1](upper)[0], 0)


@pytest.mark.full
def test_local_ill_conditioned_adjoint_against_decimal():
    # A leading-factor cotangent excites the skew component in the dense rule.
    # n=16 matches the established independent GP anchor that exposed the failure.
    case = Case("se_clustered", 16, 4, "tail", 8)
    noise = 1e-12 if jax.config.x64_enabled else 1e-4
    args = fixture(case, DTYPE, noise)
    L = jnp.linalg.cholesky(covariance(args[0], args[-1], case.scenario))
    fn = function(case, "schur", "native", from_factor=True)
    G = jax.jit(jax.grad(lambda L: fn(*args, L)))(L)
    expected = decimal_pullback(np.asarray(L), np.asarray(G))
    actual = np.asarray(jax.jit(pullback)(L, G))
    relative = np.linalg.norm(actual.astype(np.float64) - expected) / np.linalg.norm(
        expected
    )
    assert relative < (1e-6 if jax.config.x64_enabled else 3e-3)


@pytest.mark.full
@pytest.mark.parametrize("method", ["schur", "solve_scan"])
def test_severe_sequence_gradient_against_decimal(method):
    case = Case("se_clustered", 16, 4, "tail", 8)
    noise = 1e-12 if jax.config.x64_enabled else 1e-4
    args = fixture(case, DTYPE, noise)
    rule = "both" if method == "schur" else "initial"
    value, grad = jax.jit(
        jax.value_and_grad(function(case, method, rule), argnums=(0, 1, 2, 3, 4))
    )(*args)
    expected_value, expected_grad = decimal_sequence(case, args)
    assert abs(float(value) - expected_value) / abs(expected_value) < (
        3e-4 if jax.config.x64_enabled else 0.03
    )
    for i, (actual, expected) in enumerate(zip(grad, expected_grad, strict=True)):
        relative = np.linalg.norm(
            np.asarray(actual, dtype=np.float64) - expected
        ) / np.linalg.norm(expected)
        # At noise 1e-12 the full-recompute initial-location gradient already
        # has ~1.8% relative error on this fixed anchor. The 3% budget records
        # that precision limit; other leaves retain a 0.3% float64 budget.
        tolerance = 0.03 if i == 0 or not jax.config.x64_enabled else 3e-3
        assert relative < tolerance


def test_public_cholesky_domain():
    with pytest.raises(TypeError):
        cholesky(jnp.eye(2, dtype=jnp.complex64))
    with pytest.raises(ValueError):
        cholesky(jnp.ones((2, 3), dtype=DTYPE))
    invalid = jax.jit(cholesky)(jnp.diag(jnp.array([1.0, -1.0], dtype=DTYPE)))
    assert not np.isfinite(invalid).all()
