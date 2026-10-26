import jax
import jax.numpy as jnp
import numpy as np

from cholops import chol_delete_append

X64 = jax.config.x64_enabled


def covariance(x, scale):
    return scale * (x @ x.T) + jnp.eye(x.shape[0])


def run_sequence(initial, indices, replacements, scale):
    """The example caller owns feature order; cholops sees only arrays."""
    L = jnp.linalg.cholesky(covariance(initial, scale))
    positions = jnp.arange(initial.shape[0] - 1)

    def step(carry, inputs):
        L, features = carry
        index, replacement = inputs
        keep = positions + (positions >= index)
        survivors = features[keep]
        k = scale * (survivors @ replacement)
        kappa = 1 + scale * jnp.vdot(replacement, replacement)
        updated = chol_delete_append(L, index, k, kappa)
        features = jnp.concatenate([survivors, replacement[None, :]], axis=0)
        return (updated, features), updated

    return jax.lax.scan(step, (L, initial), (indices, replacements))


def dense_sequence(initial, indices, replacements, scale):
    positions = jnp.arange(initial.shape[0] - 1)

    def step(features, inputs):
        index, replacement = inputs
        keep = positions + (positions >= index)
        features = jnp.concatenate([features[keep], replacement[None, :]], axis=0)
        return features, jnp.linalg.cholesky(covariance(features, scale))

    return jax.lax.scan(step, initial, (indices, replacements))


def test_replacement_history_matches_dense_with_empty_slots():
    rng = np.random.default_rng(18)
    initial = jnp.asarray(rng.normal(size=(6, 3)))
    replacements = rng.normal(size=(100, 3))
    # A zero feature vector has unit diagonal and no correlations: an empty
    # slot for this caller. Reuse and removal stay outside the numerical API.
    replacements[::4] = 0
    replacements = jnp.asarray(replacements)
    indices = jnp.asarray(rng.integers(0, 6, size=100))
    scale = jnp.asarray(0.7)
    _, history = jax.jit(run_sequence)(initial, indices, replacements, scale)
    features = np.asarray(initial, dtype=float)
    # Independent host loop checks every step, not only the final factor.
    for i, new, L in zip(
        np.asarray(indices), np.asarray(replacements), history, strict=True
    ):
        features = np.concatenate([np.delete(features, i, axis=0), new[None, :]])
        A = float(scale) * features @ features.T + np.eye(6)
        np.testing.assert_allclose(
            L,
            np.linalg.cholesky(A),
            rtol=2e-11 if X64 else 8e-5,
            atol=2e-12 if X64 else 8e-6,
        )
        np.testing.assert_allclose(
            L @ L.T, A, rtol=2e-11 if X64 else 8e-5, atol=2e-12 if X64 else 8e-6
        )


def test_differentiating_a_compiled_replacement_sequence():
    rng = np.random.default_rng(21)
    initial = jnp.asarray(rng.normal(size=(4, 2)))
    replacements = jnp.asarray(rng.normal(size=(8, 2)))
    indices = jnp.asarray([0, 3, 1, 0, 2, 3, 2, 1])
    scale = jnp.asarray(0.8)

    def objective(initial, replacements, scale):
        _, factors = run_sequence(initial, indices, replacements, scale)
        # Every step contributes: early inputs must have nontrivial gradients.
        return jnp.sum(jnp.log(jnp.diagonal(factors, axis1=-2, axis2=-1)))

    def reference(initial, replacements, scale):
        _, factors = dense_sequence(initial, indices, replacements, scale)
        return jnp.sum(jnp.log(jnp.diagonal(factors, axis1=-2, axis2=-1)))

    args = initial, replacements, scale
    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))(*args)
    expected, expected_gradients = jax.value_and_grad(reference, argnums=(0, 1, 2))(
        *args
    )
    tolerance = dict(rtol=2e-10 if X64 else 4e-4, atol=2e-11 if X64 else 4e-5)
    np.testing.assert_allclose(value, expected, **tolerance)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, **tolerance)


def test_batched_sequences():
    initial = jnp.asarray(np.random.default_rng(7).normal(size=(2, 3, 2)))
    indices = jnp.asarray([[0, 1, 2], [2, 1, 0]])
    replacements = jnp.ones((2, 3, 2)) * 0.3
    (_, _), history = jax.jit(jax.vmap(run_sequence, in_axes=(0, 0, 0, None)))(
        initial, indices, replacements, jnp.asarray(1.0)
    )
    _, expected = jax.vmap(dense_sequence, in_axes=(0, 0, 0, None))(
        initial, indices, replacements, jnp.asarray(1.0)
    )
    np.testing.assert_allclose(
        history, expected, rtol=2e-11 if X64 else 5e-5, atol=2e-12 if X64 else 5e-6
    )
