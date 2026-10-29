"""Block deletion and delete-and-append using positive rank-one updates."""

import operator

import jax
import jax.numpy as jnp
from jax import Array, lax

from ._autodiff import tangent_map
from ._indexing import _indices, _keep_many
from ._validation import _blocks, _factor
from .append import _append_block
from .rank_one import _update_bwd, _update_fwd, _update_pushforward


def _extract(L, keep, indices):
    rows = jnp.take(L, keep, axis=0)
    return jnp.take(rows, keep, axis=1), jnp.take(rows, indices, axis=1)


def _delete_fwd(L, indices, keep):
    retained, columns = _extract(L, keep, indices)
    Y, updates = lax.scan(_update_fwd, retained, columns.T)
    return Y, (indices, keep, updates)


def _delete_bwd(saved, cotangent):
    indices, keep, updates = saved

    def step(G, update):
        return _update_bwd(update, G)

    grad_retained, grad_columns = lax.scan(step, cotangent, updates, reverse=True)
    n = keep.size + indices.size
    # Scatter whole rows/columns with one-dimensional indices.
    columns = jnp.zeros((n, keep.size), dtype=cotangent.dtype)
    columns = columns.at[keep].set(grad_retained.T).at[indices].set(grad_columns)
    result = jnp.zeros((n, n), dtype=cotangent.dtype).at[keep].set(columns.T)
    return jnp.tril(result)


def _delete_pushforward(saved, dL):
    indices, keep, updates = saved
    dM, dU = _extract(dL, keep, indices)

    def step(dM, inputs):
        update, du = inputs
        return _update_pushforward(update, (dM, du)), None

    return lax.scan(step, dM, (updates, dU.T))[0]


@jax.custom_jvp
def _delete(L, indices, keep):
    retained, columns = _extract(L, keep, indices)

    def step(M, u):
        return _update_fwd(M, u)[0], None

    return lax.scan(step, retained, columns.T)[0]


@_delete.defjvp
def _delete_jvp(primals, tangents):
    L, indices, keep = primals
    dL, _, _ = tangents
    Y, saved = _delete_fwd(L, indices, keep)
    return Y, tangent_map(_delete_bwd, saved, dL, Y, _delete_pushforward)


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

    Applies r positive rank-one updates to the retained factor, with
    O(r*(n-r)^2) work and reverse-mode storage. Use chol_delete_last for
    trailing deletion by slicing.
    """
    L = _factor(L)
    indices = _indices(indices, L.shape[0])
    if indices.size == 0:
        return L
    if indices.size == L.shape[0]:
        return L[:0, :0]
    keep = _keep_many(L.shape[0], indices)
    return _delete(L, indices, keep)


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
    preserve the factor shape for lax.scan. Deletion uses positive rank-one updates,
    followed by block append with the symmetry-preserving Cholesky reverse rule.
    Use chol_replace_last for equal-sized trailing-block replacement.
    """
    L = chol_delete(L, indices)
    K, C = _blocks(K, C, L)
    return _append_block(L, K, C)
