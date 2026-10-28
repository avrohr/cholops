"""Block deletion and delete-and-append using orthogonal rotations."""

import operator

import jax
import jax.numpy as jnp
from jax import Array, lax

from ._autodiff import tangent_map
from ._indexing import _indices, _keep_many
from ._validation import _blocks, _factor
from .append import _append_block


def _extract(L, keep, indices):
    rows = jnp.take(L, keep, axis=0)
    return jnp.take(rows, keep, axis=1), jnp.take(rows, indices, axis=1)


def _rotate(column, work, index):
    """Eliminate a row of the deleted-column block with one orthogonal rotation."""
    diagonal = column[index]
    row = work[index]
    norm = jnp.linalg.norm(jnp.concatenate((diagonal[None], row)))
    c, s = diagonal / norm, row / norm
    # Right-multiply [column, work] by the orthogonal matrix
    # [[c, -s.T], [s, I - outer(s, s)/(1+c)]].
    # Its pivot row becomes [norm, 0, ..., 0].
    updated = c * column + work @ s
    work = work - ((column + updated) / (1 + c))[:, None] * s[None, :]
    rows = jnp.arange(column.size)
    work = jnp.where((rows > index)[:, None], work, 0)
    updated = jnp.where(rows >= index, updated, 0).at[index].set(norm)
    return work, (updated, c, s)


def _delete_fwd(L, indices, keep):
    retained, columns = _extract(L, keep, indices)

    def step(work, inputs):
        index, column = inputs
        return _rotate(column, work, index)

    _, (factor, c, s) = lax.scan(step, columns, (jnp.arange(keep.size), retained.T))
    Y = factor.T
    return Y, (indices, keep, retained, columns, Y, c, s)


def _delete_bwd(saved, cotangent):
    indices, keep, retained, columns, Y, c, s = saved

    def step(carry, inputs):
        work, grad_work = carry
        index, column, updated, c, s, G = inputs
        # Invert the rotation to recover its input without saving every workspace.
        work = work + ((column + updated) / (1 + c))[:, None] * s[None, :]

        def rotate(column, work):
            work, (updated, _, _) = _rotate(column, work, index)
            return work, updated

        _, pullback = jax.vjp(rotate, column, work)
        grad_column, grad_work = pullback((grad_work, G))
        return (work, grad_work), grad_column

    (_, grad_columns), grad_retained = lax.scan(
        step,
        (jnp.zeros_like(columns), jnp.zeros_like(columns)),
        (jnp.arange(keep.size), retained.T, Y.T, c, s, cotangent.T),
        reverse=True,
    )
    n = keep.size + indices.size
    # Scatter whole rows/columns with one-dimensional indices.
    rows = jnp.zeros((n, keep.size), dtype=cotangent.dtype)
    rows = rows.at[keep].set(grad_retained).at[indices].set(grad_columns.T)
    return jnp.tril(jnp.zeros((n, n), dtype=cotangent.dtype).at[keep].set(rows.T))


def _delete_pushforward(saved, dL):
    indices, keep, retained, columns, Y, c, s = saved
    dM, dU = _extract(dL, keep, indices)

    def step(carry, inputs):
        work, dwork = carry
        index, column, dcolumn, updated, c, s = inputs
        norm = updated[index]
        dnorm = c * dcolumn[index] + s @ dwork[index]
        dc = (dcolumn[index] - c * dnorm) / norm
        ds = (dwork[index] - s * dnorm) / norm
        dy = c * dcolumn + dc * column + dwork @ s + work @ ds
        scale = (column + updated) / (1 + c)
        dscale = (dcolumn + dy - scale * dc) / (1 + c)
        work = work - scale[:, None] * s[None, :]
        dwork = dwork - dscale[:, None] * s[None, :] - scale[:, None] * ds[None, :]
        rows = jnp.arange(column.size)
        work = jnp.where((rows > index)[:, None], work, 0)
        dwork = jnp.where((rows > index)[:, None], dwork, 0)
        dy = jnp.where(rows >= index, dy, 0).at[index].set(dnorm)
        return (work, dwork), dy

    return lax.scan(
        step,
        (columns, dU),
        (jnp.arange(keep.size), retained.T, dM.T, Y.T, c, s),
    )[1].T


@jax.custom_jvp
def _delete(L, indices, keep):
    return _delete_fwd(L, indices, keep)[0]


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

    Rotates the r deleted columns into the retained factor, with O(r*(n-r)^2)
    work and O((n-r)^2 + r*(n-r)) reverse-mode storage. Use chol_delete_last
    for trailing deletion by slicing.
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
    preserve the factor shape for lax.scan. Deletion uses orthogonal rotations,
    followed by block append with the symmetry-preserving Cholesky reverse rule.
    Use chol_replace_last for equal-sized trailing-block replacement.
    """
    L = chol_delete(L, indices)
    K, C = _blocks(K, C, L)
    return _append_block(L, K, C)
