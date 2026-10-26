"""Replacement baseline using two ordinary JAX rank-one scans.

Replace a row and column with a positive update followed by a negative update.
Standard JAX autodiff differentiates both scans. The reverse pass saves column
history, which contributes to the compiler storage reported by the benchmark.
"""

import jax
import jax.numpy as jnp


def _rank1(L, u, sign):
    rows = jnp.arange(L.shape[0])

    def step(x, item):
        i, column = item
        diagonal = column[i]
        radius = jnp.sqrt(diagonal * diagonal + sign * x[i] * x[i])
        cosine, sine = radius / diagonal, x[i] / diagonal
        column = (column + sign * sine * x) / cosine
        column = jnp.where(rows > i, column, jnp.where(rows == i, radius, 0))
        x = jnp.where(rows > i, cosine * x - sine * column, 0)
        return x, column

    # Earlier steps never change a later input column. Carry only the evolving
    # vector and collect output columns instead of repeatedly updating a matrix.
    _, columns = jax.lax.scan(step, u, (rows, L.T))
    return columns.T


def replace(L, index, k, kappa):
    """Index-preserving replacement, with the same inputs as chol_replace."""
    L = jnp.tril(L)
    n = L.shape[0]
    keep = jnp.arange(n - 1) + (jnp.arange(n - 1) >= index)
    target = jnp.zeros(n, L.dtype).at[keep].set(k).at[index].set(kappa)
    difference = (target - L @ L[index]).at[index].multiply(0.5)
    basis = jnp.zeros(n, L.dtype).at[index].set(1)
    # A' = A + e d.T + d e.T = A + u u.T - v v.T.
    # Positive first keeps the intermediate SPD when the target is SPD.
    scale = jnp.sqrt(jnp.asarray(0.5, L.dtype))
    updated = _rank1(L, scale * (difference + basis), 1)
    return _rank1(updated, scale * (difference - basis), -1)
