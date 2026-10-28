"""Replace a variable using raw factors and keep the right-hand side in order.

The two replacement functions describe the same new variables but return
factors in different orders. This example solves both systems and translates
the answers into the same order for comparison.

Run: JAX_ENABLE_X64=1 uv run --locked python examples/tensor_operations.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from cholops import chol_delete_append, chol_replace, cholesky


def solve(lower, rhs):
    # For A = L @ L.T, solve L @ z = rhs followed by L.T @ solution = z.
    # Raw tensor operations leave these downstream calculations to the caller.
    z = solve_triangular(lower, rhs, lower=True)
    return solve_triangular(lower, z, lower=True, trans="T")


def main():
    # Start with variables [a, b, c, d] and replace b (index 1) with e.
    covariance = jnp.array(
        [
            [2.0, 0.3, 0.1, 0.2],
            [0.3, 1.5, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.05],
            [0.2, 0.1, 0.05, 1.3],
        ]
    )
    lower = cholesky(covariance)
    # Both functions take covariances with the survivors in order [a, c, d].
    cross = jnp.array([0.15, 0.1, 0.2])
    variance = 1.2
    replaced = chol_replace(lower, 1, cross, variance)
    appended = chol_delete_append(
        lower,
        jnp.asarray([1]),
        cross[:, None],
        jnp.array([[variance]]),
    )

    # chol_replace preserves the replaced slot: the result is [a, e, c, d].
    rhs = jnp.array([1.0, -0.3, 0.2, 0.4])
    solution = solve(replaced, rhs)

    # chol_delete_append puts e last: [a, c, d, e]. Rearrange the RHS to match,
    # then bring the solution back to [a, e, c, d] before comparing it.
    append_order = jnp.array([0, 2, 3, 1])
    inverse_order = jnp.argsort(append_order)
    appended_solution = solve(appended, rhs[append_order])
    restored_solution = appended_solution[inverse_order]
    print("chol_replace solution in [a, e, c, d] order:", solution)
    print("chol_delete_append solution in [a, c, d, e] order:", appended_solution)
    print("The latter restored to [a, e, c, d] order:", restored_solution)

    # Check against the explicitly assembled replacement matrix in [a, e, c, d]
    # order. A raw factor carries no labels or permutation metadata.
    expected = jnp.array(
        [
            [2.0, 0.15, 0.1, 0.2],
            [0.15, 1.2, 0.1, 0.2],
            [0.1, 0.1, 1.0, 0.05],
            [0.2, 0.2, 0.05, 1.3],
        ]
    )
    tolerance = 1e-10 if jax.config.x64_enabled else 2e-5
    np.testing.assert_allclose(
        replaced @ replaced.T, expected, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        appended @ appended.T,
        expected[append_order[:, None], append_order],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        solution, jnp.linalg.solve(expected, rhs), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        restored_solution, solution, rtol=tolerance, atol=tolerance
    )


if __name__ == "__main__":
    main()
