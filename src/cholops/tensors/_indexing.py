"""Principal-submatrix indexing without quadratic Cartesian index tensors."""

import jax
import jax.numpy as jnp
import numpy as np

from ._autodiff import tangent_map


def _indices(indices, n):
    indices = jnp.asarray(indices)
    if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError("indices must be a one-dimensional integer array")
    if indices.size > n:
        raise ValueError("cannot delete more than n variables")
    indices = jnp.sort(indices)
    if not isinstance(indices, jax.core.Tracer):
        values = np.asarray(indices)
        if np.any((values < 0) | (values >= n)) or np.any(np.diff(values) == 0):
            raise ValueError("indices must be distinct and in [0,n)")
    return indices


def _keep_many(n, indices):
    """Map survivor positions to original indices, given sorted deletions."""
    positions = jnp.arange(n - indices.size)
    offsets = indices - jnp.arange(indices.size)
    return positions + jnp.searchsorted(offsets, positions, side="right")


def _keep(n, index):
    positions = jnp.arange(n - 1)
    return positions + (positions >= index)


def _insert(matrix, index, row, column, diagonal):
    """Insert one row/column with linear-sized gather indices.

    Used both for factor assembly and the transpose of principal extraction.
    The inserted diagonal is supplied separately from the off-diagonal vectors.
    """
    n = matrix.shape[0] + 1
    if n == 1:
        return jnp.reshape(diagonal, (1, 1))
    positions = jnp.arange(n)
    source = jnp.clip(positions - (positions > index), 0, n - 2)
    result = jnp.take(jnp.take(matrix, source, axis=0), source, axis=1)
    result = jnp.where((positions == index)[:, None], row[source][None, :], result)
    result = jnp.where((positions == index)[None, :], column[source][:, None], result)
    return jnp.where(
        (positions == index)[:, None] & (positions == index)[None, :],
        diagonal,
        result,
    )


@jax.custom_jvp
def _principal(matrix, index):
    keep = _keep(matrix.shape[0], index)
    return jnp.take(jnp.take(matrix, keep, axis=0), keep, axis=1)


def _principal_pullback(index, cotangent):
    zeros = jnp.zeros((cotangent.shape[0],), dtype=cotangent.dtype)
    return _insert(cotangent, index, zeros, zeros, jnp.zeros((), cotangent.dtype))


@_principal.defjvp
def _principal_jvp(primals, tangents):
    matrix, index = primals
    dmatrix, _ = tangents
    result = _principal(matrix, index)
    return result, tangent_map(_principal_pullback, index, dmatrix, result)
