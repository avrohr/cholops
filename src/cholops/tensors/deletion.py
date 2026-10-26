"""Deletion and delete-and-append using positive rank-one updates."""

import jax
import jax.numpy as jnp
from jax import Array, lax

from ._autodiff import tangent_map
from ._indexing import _insert, _principal
from ._validation import _factor, _index, _scalar, _vector
from .append import _append
from .rank_one import _update_bwd, _update_fwd, _update_pushforward


def _delete_fwd(L: Array, index: Array):
    positions = jnp.arange(L.shape[0] - 1)
    keep = positions + (positions >= index)
    retained = _principal(L, index)
    u = jnp.where(keep > index, L[keep, index], 0)
    Y, v, rho, c, b = lax.cond(
        index == L.shape[0] - 1,
        lambda _: (retained, u, jnp.ones_like(u), jnp.ones_like(u), u),
        lambda _: _delete_update(retained, u),
        operand=None,
    )
    return Y, (index, (retained, u, v, rho, c, b, Y))


def _delete_update(retained, u):
    Y, (_, _, v, rho, c, b, _) = _update_fwd(retained, u)
    return Y, v, rho, c, b


def _delete_bwd(saved, cotangent):
    index, update_saved = saved
    n = cotangent.shape[0] + 1
    zeros = jnp.zeros((n - 1,), dtype=cotangent.dtype)
    grad_retained, grad_u = lax.cond(
        index == n - 1,
        lambda _: (jnp.tril(cotangent), zeros),
        lambda _: _update_bwd(update_saved, cotangent),
        operand=None,
    )
    keep = jnp.arange(n - 1)
    keep = keep + (keep >= index)
    # The retained entries and the deleted column are disjoint. Assemble their
    # cotangents together rather than inserting zeros and then scatter-adding.
    column = jnp.where(keep > index, grad_u, 0)
    return _insert(grad_retained, index, zeros, column, jnp.zeros((), cotangent.dtype))


def _delete_pushforward(saved, dL):
    index, update_saved = saved
    retained = _principal(dL, index)
    keep = jnp.arange(dL.shape[0] - 1)
    keep = keep + (keep >= index)
    du = jnp.where(keep > index, dL[keep, index], 0)
    return lax.cond(
        index == dL.shape[0] - 1,
        lambda _: retained,
        lambda _: _update_pushforward(update_saved, (retained, du)),
        operand=None,
    )


@jax.custom_jvp
def _delete(L: Array, index: Array) -> Array:
    if L.shape[0] == 1:
        return L[:0, :0]
    return _delete_fwd(L, index)[0]


@_delete.defjvp
def _delete_jvp(primals, tangents):
    L, index = primals
    dL, _ = tangents
    if L.shape[0] == 1:
        return L[:0, :0], dL[:0, :0]
    Y, saved = _delete_fwd(L, index)
    return Y, tangent_map(_delete_bwd, saved, dL, Y, _delete_pushforward)


def chol_delete(L: Array, index: Array | int) -> Array:
    """Delete a covariance row/column, returning an (n-1,n-1) lower factor.

    index is a scalar integer in [0,n). Check traced indices before calling
    this function, since they are not checked at runtime.
    Survivor order is preserved. Supports jit, vmap and first-order JVP and VJP rules.
    Last-position deletion needs no update solve.
    """
    L = _factor(L)
    index = _index(index, L.shape[0])
    return _delete(L, index)


def chol_delete_append(
    L: Array, index: Array | int, k: Array, kappa: Array | float
) -> Array:
    """Delete index and append a replacement, retaining shape (n,n).

    k has shape (n-1,) in survivor order; kappa is the new covariance diagonal.
    The resulting covariance must be positive definite. Physical ordering moves
    the replacement to the end. Reorder subsequent inputs to match this order.
    Supports jit, vmap, first-order JVP and VJP rules and fixed-size lax.scan sequences.
    index must be in [0,n). Traced indices are not checked at runtime.
    """
    L = _factor(L)
    index = _index(index, L.shape[0])
    k = _vector(k, L.shape[0] - 1, L, "k")
    kappa = _scalar(kappa, L, "kappa")
    return _append(_delete(L, index), k, kappa)
