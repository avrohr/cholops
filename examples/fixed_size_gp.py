"""Activate reserved GP slots without changing factor or data array shapes.

The caller chooses logical slots. PaddedCholesky masks inactive inputs and
manages physical factor ordering. Hyperparameters stay fixed during streaming.

Run: JAX_ENABLE_X64=1 uv run --locked python examples/fixed_size_gp.py
"""

import jax.numpy as jnp
from plain_gp import check, example_data, kernel, kernel_diag, plain_gp

from cholops import PaddedCholesky


def fixed_size_gp(capacity, d, stream, X_star, noise_var):
    # Finite dummy locations keep kernel evaluations valid before activation.
    X = jnp.zeros((capacity, d), dtype=X_star.dtype)
    y = jnp.zeros(capacity, dtype=X_star.dtype)
    alpha = jnp.zeros(capacity, dtype=X_star.dtype)
    factor = PaddedCholesky.from_factor(
        jnp.eye(capacity, dtype=X_star.dtype), active=jnp.zeros(capacity, dtype=bool)
    )
    results = []

    for indices, X_new, y_new in stream:
        # Full-capacity K needs no caller-side masking or survivor gathering.
        K = kernel(X, X_new)
        C = kernel(X_new, X_new) + noise_var * jnp.eye(y_new.size, dtype=X.dtype)
        y = y.at[indices].set(y_new)
        factor, alpha = factor.activate_solve(indices, K, C, solution=alpha, rhs=y)
        X = X.at[indices].set(X_new)

        K_star = kernel(X, X_star)
        V = factor.whiten(K_star)
        mean = K_star.T @ alpha
        variance = kernel_diag(X_star) - jnp.sum(V * V, axis=0)
        # The Gaussian normalization counts active observations, not capacity.
        log_likelihood = -0.5 * (
            y @ alpha + factor.logdet() + factor.active.sum() * jnp.log(2 * jnp.pi)
        )
        results.append((mean, variance, log_likelihood))
    return results


def main():
    X, y, X_star, noise_var = example_data()
    # Scattered, unsorted logical slots exercise interspersed identity padding.
    slots = jnp.array([[7, 1], [4, 0], [8, 3], [2, 6]])
    stream = [
        (indices, X[i : i + 2], y[i : i + 2])
        for indices, i in zip(slots, range(0, 8, 2), strict=True)
    ]
    results = fixed_size_gp(10, X.shape[1], stream, X_star, noise_var)
    for count, result in zip(range(2, 9, 2), results, strict=True):
        check(result, plain_gp(X[:count], y[:count], X_star, noise_var))
        print(
            f"{count}/10 active slots: log marginal likelihood {float(result[2]):.6f}"
        )
    print("Every block agrees with a fresh GP fit, including the initial block.")


if __name__ == "__main__":
    main()
