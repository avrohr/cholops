"""Fit a line as measurements arrive, using positive rank-one updates.

Ridge regression solves (X.T @ X + ridge * I) @ weights = X.T @ y.
Each new feature vector adds outer(features, features) to this system matrix
and features * target to its right-hand side. We update both and reuse the
factor to obtain the new fit. Here ridge penalizes both the intercept and slope.

Run: JAX_ENABLE_X64=1 uv run --locked python examples/online_least_squares.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from cholops import Cholesky


@jax.jit
def add_measurements(factor, rhs, incoming_features, incoming_targets):
    def step(state, measurement):
        factor, rhs = state
        features, target = measurement
        # The factor represents a system over model coefficients, so its size
        # stays fixed even as the number of observations grows.
        factor = factor.rank1_positive(features)
        rhs = rhs + features * target
        weights = factor.solve(rhs)
        return (factor, rhs), weights

    return jax.lax.scan(step, (factor, rhs), (incoming_features, incoming_targets))


def main():
    locations = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    targets = 0.4 + 1.7 * locations + 0.1 * jnp.sin(4 * locations)
    # A row [1, x] represents the prediction intercept + slope * x.
    features = jnp.stack((jnp.ones_like(locations), locations), axis=1)
    ridge = 0.2
    initial_features, incoming_features = features[:3], features[3:]
    initial_targets, incoming_targets = targets[:3], targets[3:]

    # The positive ridge term makes the system positive definite, including
    # when the observations alone do not determine all coefficients.
    system = initial_features.T @ initial_features + ridge * jnp.eye(2)
    rhs = initial_features.T @ initial_targets
    factor = Cholesky.from_matrix(system)
    initial_weights = factor.solve(rhs)
    (final_factor, final_rhs), weights = add_measurements(
        factor, rhs, incoming_features, incoming_targets
    )

    print("Initial [intercept, slope]:", initial_weights)
    for count, coefficients in enumerate(weights, start=4):
        print(f"After {count} observations: {coefficients}")
    prediction_features = jnp.array([1.0, 3.0])
    print(f"Prediction at x=3: {float(prediction_features @ weights[-1]):.6f}")

    # Rebuild the full regression problem once to check the final result.
    expected_system = features.T @ features + ridge * jnp.eye(2)
    expected_rhs = features.T @ targets
    expected_weights = jnp.linalg.solve(expected_system, expected_rhs)
    tolerance = 1e-10 if jax.config.x64_enabled else 2e-5
    np.testing.assert_allclose(
        final_factor.factor @ final_factor.factor.T,
        expected_system,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(final_rhs, expected_rhs, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        weights[-1], expected_weights, rtol=tolerance, atol=tolerance
    )


if __name__ == "__main__":
    main()
