"""Fit a GP and compute posterior means, marginal variances and likelihood.

Run: JAX_ENABLE_X64=1 uv run --locked python examples/plain_gp.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from cholops import Cholesky


def kernel(X, Y):
    """Unit-variance squared-exponential kernel, with length scale 0.8."""
    difference = (X[:, None, :] - Y[None, :, :]) / 0.8
    return jnp.exp(-0.5 * jnp.sum(difference**2, axis=-1))


def kernel_diag(X):
    return jnp.ones(X.shape[0], dtype=X.dtype)


def plain_gp(X, y, X_star, noise_var):
    n = y.size
    A = kernel(X, X) + noise_var * jnp.eye(n, dtype=X.dtype)
    factor = Cholesky.from_matrix(A)
    alpha = factor.solve(y)

    # One triangular solve supplies the covariance reduction for all queries.
    K_star = kernel(X, X_star)
    V = factor.whiten(K_star)
    mean = K_star.T @ alpha
    variance = kernel_diag(X_star) - jnp.sum(V * V, axis=0)
    log_likelihood = -0.5 * (y @ alpha + factor.logdet() + n * jnp.log(2 * jnp.pi))
    return mean, variance, log_likelihood


def example_data():
    """The three examples use the same observations and fixed hyperparameters."""
    X = jnp.linspace(-1.8, 2.1, 8)[:, None]
    y = jnp.sin(2 * X[:, 0]) + 0.1 * jnp.cos(5 * X[:, 0])
    X_star = jnp.linspace(-2.0, 2.3, 13)[:, None]
    return X, y, X_star, 0.15


def check(actual, expected):
    tolerance = 2e-9 if jax.config.x64_enabled else 2e-4
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


def main():
    X, y, X_star, noise_var = example_data()
    result = plain_gp(X, y, X_star, noise_var)

    # Independent dense solves check the factor-based calculation.
    A = np.asarray(kernel(X, X)) + noise_var * np.eye(y.size)
    cross = np.asarray(kernel(X, X_star))
    alpha = np.linalg.solve(A, np.asarray(y))
    expected = (
        cross.T @ alpha,
        np.asarray(kernel_diag(X_star))
        - np.sum(cross * np.linalg.solve(A, cross), axis=0),
        -0.5
        * (
            np.asarray(y) @ alpha + np.linalg.slogdet(A)[1] + y.size * np.log(2 * np.pi)
        ),
    )
    check(result, expected)
    print("Posterior means:", result[0])
    print("Latent marginal variances:", result[1])
    print(f"Log marginal likelihood: {float(result[2]):.6f}")
    print("Dense reference checks passed.")


if __name__ == "__main__":
    main()
