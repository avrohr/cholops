"""Block deletion and delete-and-append using a positive block update."""

import operator

import jax.numpy as jnp
from jax import Array

from ._indexing import _indices, _keep, _keep_many
from ._validation import _blocks, _factor, _index
from .append import _append_block
from .rank_one import chol_rank1_positive
from .update import chol_update


def _extract(L, keep, indices):
    rows = jnp.take(L, keep, axis=0)
    # Gather rows of the transpose so reverse mode scatters contiguous rows.
    return jnp.take(rows.T, keep, axis=0).T, jnp.take(rows.T, indices, axis=0).T


def _delete_last(L, r):
    m = L.shape[0] - r
    return L[:m, :m]


def chol_delete(L: Array, indices: Array) -> Array:
    """Delete variables at indices:(r,), returning an (n-r,n-r) lower factor.

    Indices must be distinct integers in [0,n), in any order. They refer to
    the input factor. Survivor order is preserved. Use an array of length one
    for a single deletion. Empty indices and deleting all variables are supported.
    Index values may be traced, but their count is static. Traced values are
    not checked at runtime. Supports jit, vmap and first-order JVP and VJP rules.

    Applies chol_update to the retained factor and deleted columns, including
    for a single deletion. Use chol_delete_one to select the rank-one kernel,
    or chol_delete_last for trailing deletion by slicing.
    """
    L = _factor(L)
    indices = _indices(indices, L.shape[0])
    if indices.size == 0:
        return L
    if indices.size == L.shape[0]:
        return L[:0, :0]
    keep = _keep_many(L.shape[0], indices)
    retained, columns = _extract(L, keep, indices)
    return chol_update(retained, columns)


def chol_delete_one(L: Array, index: Array | int) -> Array:
    """Delete one variable using chol_rank1_positive, preserving survivor order.

    index is an integer scalar in [0,n) and may be traced. The result has
    shape (n-1,n-1), including an empty factor when n=1. Traced index values
    are not checked at runtime. Supports jit, vmap and first-order JVP and VJP.
    """
    L = _factor(L)
    index = _index(index, L.shape[0])
    retained, columns = _extract(L, _keep(L.shape[0], index), index[None])
    return chol_rank1_positive(retained, columns[:, 0])


def chol_delete_last(L: Array, r: int) -> Array:
    """Remove the final r variables by retaining the leading lower factor.

    r must be a static integer in [0,n]. The discarded tail is not used.
    Supports jit with static r, vmap and differentiation through the slice.
    """
    L = _factor(L)
    r = operator.index(r)
    if not 0 <= r <= L.shape[0]:
        raise ValueError("r must be in [0,n]")
    return _delete_last(L, r)


def chol_delete_append(L: Array, indices: Array, K: Array, C: Array) -> Array:
    """Delete indices:(r,) and append a block, returning size n-r+s.

    K:(n-r,s) follows the surviving variables in their original order.
    C:(s,s) must be symmetric and the target covariance positive definite.
    Indices must be distinct and in bounds. Their values may be traced, but r
    and s are static. Traced index values are not checked at runtime.
    All floating inputs share a real float32/64 dtype.

    Supports jit, vmap and first-order JVP and VJP rules. Equal block sizes
    preserve the factor shape for lax.scan. Deletion uses chol_update,
    followed by block append with the symmetry-preserving Cholesky reverse rule.
    Use chol_replace_last for equal-sized trailing-block replacement.
    """
    L = chol_delete(L, indices)
    K, C = _blocks(K, C, L)
    return _append_block(L, K, C)
