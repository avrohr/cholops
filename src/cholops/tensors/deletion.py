"""Block deletion and delete-and-append using a positive block update."""

import operator

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff import tangent_map
from ._indexing import _indices, _keep, _keep_many
from ._validation import _blocks, _factor, _index
from .append import _append_block
from .rank_one import chol_rank1_positive
from .update import chol_update


def _extract_forward(L, keep, indices):
    rows = jnp.take(L, keep, axis=0)
    # Pack transposed blocks along their first axis so splitting the result
    # preserves the row-gather layout of both retained and deleted columns.
    return jnp.take(rows.T, jnp.concatenate((keep, indices)), axis=0)


@jax.custom_jvp
def _extract_packed(L, keep, indices):
    """Keep disjoint block cotangents together for one inverse gather."""
    return _extract_forward(L, keep, indices)


def _extract_pullback(saved, cotangent):
    _, indices = saved
    n, m = cotangent.shape
    if m == 0:
        return jnp.zeros((n, n), cotangent.dtype)
    if indices.size == 0:
        return cotangent.T
    positions = jnp.arange(n)
    gaps = jnp.searchsorted(indices, positions, side="left")
    deleted = positions == jnp.take(indices, gaps, mode="clip")
    survivors = positions - gaps
    columns = jnp.where(deleted, m + gaps, survivors)
    # Restore both column groups together, then insert zero rows by a masked
    # gather. Each source entry occurs once, so no scatter-add is necessary.
    rows = jnp.take(cotangent, columns, axis=0).T
    result = jnp.take(rows, jnp.minimum(survivors, m - 1), axis=0)
    return jnp.where(deleted[:, None], 0, result)


@_extract_packed.defjvp
def _extract_jvp(primals, tangents):
    L, keep, indices = primals
    dL, _, _ = tangents
    result = _extract_packed(L, keep, indices)
    return result, tangent_map(
        _extract_pullback,
        (keep, indices),
        dL,
        result,
        lambda saved, dL: _extract_forward(dL, *saved),
    )


def _extract(L, keep, indices):
    packed = _extract_packed(L, keep, indices)
    return packed[: keep.size].T, packed[keep.size :].T


def _delete_last(L, r):
    m = L.shape[0] - r
    return L[:m, :m]


def chol_delete(L: Array, indices: Array) -> Array:
    """Delete variables at indices:(r,), returning an (n-r,n-r) lower factor.

    Indices must be distinct integers in [0,n), in increasing order. They refer to
    the input factor. Survivor order is preserved. Use an array of length one
    for a single deletion. Empty indices and deleting all variables are supported.
    Index values are not checked and may be traced, but their count is static.
    Supports jit, vmap and first-order JVP and VJP rules.

    Applies chol_update to the retained factor and all deleted columns together,
    including for a single deletion. For one variable, prefer chol_delete_one,
    which is usually faster. Use chol_delete_last for trailing deletion by slicing.
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

    Usually faster than chol_delete(L, jnp.array([index])), especially on GPUs.
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
    Indices must be increasing, distinct and in bounds. Values may be traced, but r
    and s are static. Index values are not checked.
    All floating inputs share a real float32/64 dtype.

    Supports jit, vmap and first-order JVP and VJP rules. Equal block sizes
    preserve the factor shape for lax.scan. Deletion uses chol_update,
    followed by block append with the symmetry-preserving Cholesky reverse rule.
    Use chol_replace_last for equal-sized trailing-block replacement.
    """
    L = chol_delete(L, indices)
    K, C = _blocks(K, C, L)
    return _append_block(L, K, C)
