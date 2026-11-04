"""Positive block updates with QR prefix factors and updated-factor derivatives."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from ._autodiff import tangent_map
from ._validation import _factor


def _update_columns(U, factor):
    U = jnp.asarray(U)
    if U.ndim != 2 or U.shape[0] != factor.shape[0]:
        raise ValueError("U must have shape (n,k); use (n,1) for one column")
    if U.dtype != factor.dtype:
        raise TypeError("U and the factor must have the same dtype")
    return U


def _coefficients(L, U):
    W = solve_triangular(L.T, U, lower=False, trans="T")
    n, rank = W.shape
    leaves = jnp.zeros((n + 1, rank, rank), dtype=W.dtype)
    leaves = leaves.at[0].set(jnp.eye(rank, dtype=W.dtype))
    leaves = leaves.at[1:, 0, :].set(W)

    def combine(a, b):
        return jnp.linalg.qr(jnp.concatenate((a, b), axis=-2), mode="r")

    # R[i].T @ R[i] = I + sum_{j<i} W[j].T @ W[j]. Keeping a square
    # root avoids forming the poorly conditioned Gram matrices in float32.
    R = jax.lax.associative_scan(combine, leaves)[:-1]
    z = solve_triangular(R, W[..., None], lower=False, trans="T")[..., 0]
    c = jax.lax.rsqrt(1 + jnp.sum(z * z, axis=1))
    V = solve_triangular(R, (z * c[:, None])[..., None], lower=False)[..., 0]
    return W, V, c


def _panels(T, A, B):
    n, rank = A.shape
    width = min(n, 8 if rank == 1 else 32)
    padding = (-n) % width
    size = n + padding
    T = jnp.pad(T, ((0, padding), (0, padding)))
    A = jnp.pad(A, ((0, padding), (0, 0)))
    B = jnp.pad(B, ((0, padding), (0, 0)))
    return (
        T.reshape(size, size // width, width).transpose(1, 0, 2),
        A.reshape(size // width, width, rank),
        B.reshape(size // width, width, rank),
    )


def _unpack(blocks, n):
    size = blocks.shape[1]
    return jnp.tril(blocks.transpose(1, 0, 2).reshape(size, size)[:n, :n])


def _panel_prefix(products):
    def prefix(_, values):
        return jnp.concatenate(
            (jnp.zeros_like(values[:1]), jnp.cumsum(values, axis=0)[:-1]), axis=0
        )

    # Keep the transpose as a reverse prefix. Differentiating the slice can
    # otherwise lower to a full-width reduction window on the GPU.
    return tangent_map(
        lambda _, G: prefix((), G[::-1])[::-1], (), products, products, prefix
    )


def _upper_product(T, A, B):
    """Lower triangle of T @ triu(A @ B.T, 1), with quadratic low-rank work."""
    panels, A, B = _panels(T, A, B)
    products = panels @ A
    prefix = _panel_prefix(products)
    local = jnp.triu(A @ B.swapaxes(-1, -2), 1)
    return _unpack(panels @ local + prefix @ B.swapaxes(-1, -2), T.shape[0])


def _lower_product(Y, V, Q, U):
    """Y @ tril(V @ Q.T, -1), using the identity Y @ V = U for the tails."""
    panels, V, Q = _panels(Y, V, Q)
    size, width = panels.shape[1:]
    U = jnp.pad(U, ((0, size - U.shape[0]), (0, 0)))
    tail = U[None] - jnp.cumsum(panels @ V, axis=0)
    # These rows have no columns beyond this panel. Set their exact zero tail.
    ends = (jnp.arange(panels.shape[0]) + 1) * width
    tail = jnp.where(jnp.arange(size)[None, :, None] >= ends[:, None, None], tail, 0)
    local = jnp.tril(V @ Q.swapaxes(-1, -2), -1)
    return _unpack(panels @ local + tail @ Q.swapaxes(-1, -2), Y.shape[0])


def _assemble(L, U, W, V, c):
    # This is L[:,j]*c[j] + (U - sum_{s<j} L[:,s]*W[s]) @ V[j].
    # It avoids adding and subtracting the large current-column contribution.
    Y = jnp.tril(L * c[None, :] + U @ V.T - _upper_product(L, W, V))
    diagonal = jnp.arange(L.shape[0])
    return Y.at[diagonal, diagonal].set(jnp.diag(L) / c)


def _pushforward(Y, U, W, V, c, dL, dU):
    dL = jnp.tril(dL)
    Q = solve_triangular(Y, dU - dL @ W, lower=True)
    # If P = Y^{-1} L, then P has diagonal c and P[i,j] = -V[i] @ W[j]
    # below the diagonal. The two triangular products supply the correction
    # needed to keep dY lower triangular without solving for n right-hand sides.
    return jnp.tril(
        dL * c[None, :]
        + dU @ V.T
        - _upper_product(dL, W, V)
        - _upper_product(Y, Q, V)
        + _lower_product(Y, V, Q, U)
    )


@jax.custom_jvp
def _update_upper(R, U):
    L = R.T
    if not L.shape[0] or not U.shape[1]:
        return L.T
    return _assemble(L, U, *_coefficients(L, U)).T


@_update_upper.defjvp
def _update_jvp(primals, tangents):
    R, U = primals
    dR, dU = tangents
    L = R.T
    if not L.shape[0] or not U.shape[1]:
        return L.T, jnp.triu(dR)
    W, V, c = _coefficients(L, U)
    Y = _assemble(L, U, W, V, c)
    return Y.T, _pushforward(Y, U, W, V, c, dR.T, dU).T


def chol_update(L: Array, U: Array) -> Array:
    """Return a lower factor of L @ L.T + U @ U.T, preserving size and order.

    L:(n,n) is a lower factor with a positive diagonal. U:(n,k) contains the
    update columns. Use (n,1) for one column. Both arrays share a real float32
    or float64 dtype. Empty factors and k=0 are supported. A positive update
    preserves positive definiteness.

    Applies the block together using a triangular solve, small QR prefix factors
    and panel assembly. Derivatives solve with the updated factor. Supports jit,
    vmap and first-order JVP and reverse AD with respect to both inputs.
    """
    L = _factor(L)
    U = _update_columns(U, L)
    return _update_upper(L.T, U).T
