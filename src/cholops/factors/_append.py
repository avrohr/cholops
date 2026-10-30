"""Reuse a block factorization to update an existing solution."""

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from ._consumers import _solve


def _append_solution(L, K, solution, rhs):
    """Use the appended factor L and the new block's RHS, in physical order."""
    n, r = K.shape
    if r == 0:
        return solution
    # The projection and Schur factor are already stored in the appended factor.
    V = L[n:, :n].T
    beta = _solve(L[n:, n:], rhs - K.T @ solution)
    if n:
        solution = solution - solve_triangular(
            L[:n, :n], V @ beta, lower=True, trans="T"
        )
    return jnp.concatenate((solution, beta))
