"""Replacement baseline that rebuilds the covariance and factors it with JAX."""

import jax.numpy as jnp

from benchmarks._support import keep_indices


def replace_matrix(A, index, k, kappa):
    """Replace a covariance row/column, preserving the other indices."""
    keep = keep_indices(A.shape[0], index)
    return A.at[index, keep].set(k).at[keep, index].set(k).at[index, index].set(kappa)


def replace(L, index, k, kappa):
    """Recompute after replacing one variable, using vector/scalar inputs."""
    L = jnp.tril(L)
    A = L @ L.T
    A = replace_matrix(A, index, k, kappa)
    return jnp.linalg.cholesky(A)
