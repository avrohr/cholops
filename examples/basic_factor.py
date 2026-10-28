"""Modify a small covariance matrix through the Cholesky object API.

We add an observation, replace another, and remove one we no longer need.
The factor is reused after each change. At the end we solve a system and
calculate the two matrix terms in a Gaussian log likelihood.

Run: JAX_ENABLE_X64=1 uv run --locked python examples/basic_factor.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from cholops import Cholesky


def main():
    # Three observations, initially ordered [a, b, c]. The diagonal contains
    # their variances, and the off-diagonal entries their covariances.
    covariance = jnp.array(
        [
            [2.0, 0.3, 0.1],
            [0.3, 1.5, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )
    observations = jnp.array(
        [1.0, -0.5, 0.2],
    )
    original = Cholesky.from_matrix(covariance)

    # Add d. Supply one covariance with each existing observation, followed by
    # d's own variance. Its measured value must also be appended to the data.
    factor = original.append(jnp.array([[0.2], [0.1], [0.05]]), jnp.array([[1.3]]))
    observations = jnp.append(observations, 0.4)

    # Replace b with e at index 1. The cross-covariances follow the retained
    # order [a, c, d]. Both the factor and observations now have order [a, e, c, d].
    factor = factor.replace(1, jnp.array([0.15, 0.1, 0.2]), 1.2)
    observations = observations.at[1].set(-0.3)

    # Remove a from both objects. The surviving order is [e, c, d].
    factor = factor.delete(jnp.asarray([0]))
    observations = observations[1:]

    # These calculations use the stored factor without forming a matrix inverse.
    solution = factor.solve(observations)
    logdet = factor.logdet()
    quadratic = factor.quadratic(observations)
    log_likelihood = -0.5 * (
        logdet + quadratic + observations.size * jnp.log(2 * jnp.pi)
    )

    # Modifications return new objects. The original factor still represents
    # [a, b, c]. Use .factor when an external routine needs the lower triangle.
    lower = factor.factor
    print(f"Original factor shape: {original.factor.shape}")
    print("Final observation order: [e, c, d]")
    print("Solution:", solution)
    print(f"Log determinant: {float(logdet):.6f}")
    print(f"Inverse quadratic form: {float(quadratic):.6f}")
    print(f"Gaussian log likelihood: {float(log_likelihood):.6f}")

    # A small independent check: write down the final covariance explicitly.
    expected = jnp.array([[1.2, 0.1, 0.2], [0.1, 1.0, 0.05], [0.2, 0.05, 1.3]])
    tolerance = 1e-10 if jax.config.x64_enabled else 2e-5
    np.testing.assert_allclose(
        lower @ lower.T, expected, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        solution,
        jnp.linalg.solve(expected, observations),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        logdet, jnp.linalg.slogdet(expected)[1], rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        quadratic,
        observations @ jnp.linalg.solve(expected, observations),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        original.factor @ original.factor.T,
        covariance,
        rtol=tolerance,
        atol=tolerance,
    )


if __name__ == "__main__":
    main()
