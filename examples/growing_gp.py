"""Grow a GP in blocks and reuse its previous regression coefficients.

Run: JAX_ENABLE_X64=1 uv run --locked python examples/growing_gp.py
"""

import jax.numpy as jnp
from plain_gp import check, example_data, kernel, kernel_diag, plain_gp

from cholops import PermutedCholesky


def growing_gp(X, y, stream, X_star, noise_var):
    A = kernel(X, X) + noise_var * jnp.eye(y.size, dtype=X.dtype)
    factor = PermutedCholesky.from_matrix(A)
    alpha = factor.solve(y)
    results = []

    for X_new, y_new in stream:
        # Construct only old-to-new covariances and the new diagonal block.
        K = kernel(X, X_new)
        C = kernel(X_new, X_new) + noise_var * jnp.eye(y_new.size, dtype=X.dtype)
        y = jnp.concatenate((y, y_new))
        factor, alpha = factor.append_solve(K, C, solution=alpha, rhs=y)
        X = jnp.concatenate((X, X_new))

        K_star = kernel(X, X_star)
        V = factor.whiten(K_star)
        mean = K_star.T @ alpha
        variance = kernel_diag(X_star) - jnp.sum(V * V, axis=0)
        log_likelihood = -0.5 * (
            y @ alpha + factor.logdet() + y.size * jnp.log(2 * jnp.pi)
        )
        results.append((mean, variance, log_likelihood))
    return results


def main():
    X, y, X_star, noise_var = example_data()
    # Seed with two observations, then append three two-observation blocks.
    stream = [(X[i : i + 2], y[i : i + 2]) for i in range(2, 8, 2)]
    results = growing_gp(X[:2], y[:2], stream, X_star, noise_var)
    for count, result in zip(range(4, 9, 2), results, strict=True):
        check(result, plain_gp(X[:count], y[:count], X_star, noise_var))
        print(f"{count} observations: log marginal likelihood {float(result[2]):.6f}")
    print("Every block agrees with a fresh GP fit.")


if __name__ == "__main__":
    main()
