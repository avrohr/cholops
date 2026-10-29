"""Index-preserving block replacement and trailing-block replacement."""

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.linalg import solve_triangular

from ._autodiff import tangent_map
from ._indexing import _insert, _keep, _principal
from ._validation import _factor, _replacement
from .append import _append_block
from .deletion import _delete_last
from .factorization import cholesky
from .rank_one import _backward_rhs, _exclusive_prefix, _update_bwd, _update_fwd


@jax.named_call
def _downdate_forward(L, u):
    v = solve_triangular(L, u, lower=True)
    squares = v * v
    # Every prefix margin is the final SPD margin plus a nonnegative suffix.
    # The subtraction loses accuracy when the squared norm is close to one.
    margin = 1 - jnp.sum(squares)
    suffix = _exclusive_prefix(squares[::-1])[::-1]
    rho = jnp.sqrt(margin + suffix)
    previous = jnp.concatenate((jnp.ones_like(rho[:1]), rho[:-1]))
    c = previous / rho
    b = -(v / rho) / previous
    residual = u[:, None] - _exclusive_prefix(L * v[None, :])
    Y = jnp.tril(L * c[None, :] + residual * b[None, :])
    diagonal = jnp.arange(L.shape[0])
    Y = Y.at[diagonal, diagonal].set(jnp.diag(L) / c)
    return Y, (L, u, v, rho, c, b, Y)


@jax.custom_jvp
def _downdate(L, u):
    return _downdate_forward(L, u)[0]


@jax.named_call
def _downdate_backward(saved, cotangent):
    L, u, v, rho, c, b, Y = saved
    G = jnp.tril(cotangent)
    rhs, B = _backward_rhs(L, u, v, rho, b, Y, G, sign=-1)
    z = solve_triangular(Y, rhs, lower=True, trans="T")
    return (
        jnp.tril(G * c[None, :] - (B + z[:, None]) * v[None, :]),
        z + jnp.sum(G * b[None, :], axis=1),
    )


@_downdate.defjvp
def _downdate_jvp(primals, tangents):
    Y, saved = _downdate_forward(*primals)
    return Y, tangent_map(_downdate_backward, saved, tangents, Y)


def _leading_matrix(retained, leading):
    return jnp.where(
        leading[:, None] & leading[None, :],
        retained,
        jnp.eye(retained.shape[0], dtype=retained.dtype),
    )


@jax.named_call
def _insert_forward(retained, index, k, kappa):
    leading = jnp.arange(k.shape[0]) < index

    def leading_solve(_):
        return solve_triangular(
            _leading_matrix(retained, leading), jnp.where(leading, k, 0), lower=True
        )

    row = lax.cond(index == 0, lambda _: jnp.zeros_like(k), leading_solve, None)
    diagonal = jnp.sqrt(kappa - jnp.vdot(row, row))
    column = jnp.where(leading, 0, (k - retained @ row) / diagonal)
    remainder, negative = _downdate_forward(retained, column)
    result = _insert(
        jnp.where(leading[None, :], retained, remainder), index, row, column, diagonal
    )
    saved = (index, retained, row, diagonal, negative[1:])
    return result, saved


@jax.named_call
def _insert_backward(saved, cotangent):
    index, retained, row, diagonal, negative_tail = saved
    column = negative_tail[0]
    leading = jnp.arange(row.shape[0]) < index
    keep = _keep(cotangent.shape[0], index)
    G = jnp.tril(cotangent)
    middle = _principal(G, index)
    grad_retained, grad_column = _downdate_backward(
        (retained, *negative_tail), jnp.where(leading[None, :], 0, middle)
    )
    grad_retained = grad_retained + jnp.where(leading[None, :], middle, 0)
    grad_column = jnp.where(leading, 0, grad_column + G[keep, index])
    grad_diagonal = G[index, index] - jnp.vdot(grad_column, column) / diagonal
    grad_k = grad_column / diagonal
    grad_retained = grad_retained - grad_k[:, None] * row[None, :]
    grad_row = G[index, keep] - retained.T @ grad_k
    grad_kappa = grad_diagonal / (2 * diagonal)
    grad_row = jnp.where(leading, grad_row - 2 * grad_kappa * row, 0)

    def leading_reverse(_):
        return solve_triangular(
            _leading_matrix(retained, leading), grad_row, lower=True, trans="T"
        )

    leading_rhs = lax.cond(
        index == 0, lambda _: jnp.zeros_like(row), leading_reverse, None
    )
    grad_k = grad_k + leading_rhs
    grad_retained = jnp.tril(grad_retained - leading_rhs[:, None] * row[None, :])
    return grad_retained, grad_k, grad_kappa


def _replace_one_forward(L, index, k, kappa):
    keep = _keep(L.shape[0], index)
    base = _principal(L, index)
    column = jnp.where(keep > index, L[keep, index], 0)
    retained, positive = _update_fwd(base, column)
    Y, (_, _, row, diagonal, negative) = _insert_forward(retained, index, k, kappa)
    return Y, (index, positive, row, diagonal, negative)


def _replace_one_backward(saved, cotangent):
    index, positive, row, diagonal, negative = saved
    dR, dk, dc = _insert_backward(
        (index, positive[-1], row, diagonal, negative), cotangent
    )
    base, column = _update_bwd(positive, dR)
    keep = _keep(cotangent.shape[0], index)
    dL = _insert(
        base,
        index,
        jnp.zeros_like(row),
        jnp.where(keep > index, column, 0),
        jnp.zeros_like(diagonal),
    )
    return dL, dk, dc


@jax.custom_jvp
def _replace_one(L, index, k, kappa):
    return _replace_one_forward(L, index, k, kappa)[0]


@_replace_one.defjvp
def _replace_one_jvp(primals, tangents):
    Y, saved = _replace_one_forward(*primals)
    dL, _, dk, dc = tangents
    return Y, tangent_map(_replace_one_backward, saved, (dL, dk, dc), Y)


def _replace_block(L, start, K, C):
    n, r = L.shape[0], C.shape[0]
    positions = jnp.arange(n - r)
    keep = positions + r * (positions >= start)
    new_columns = jnp.zeros((n, r), L.dtype).at[keep].set(K)
    new_columns = lax.dynamic_update_slice_in_dim(new_columns, C, start, axis=0)
    old_rows = lax.dynamic_slice_in_dim(L, start, r, axis=0)
    difference = new_columns - L @ old_rows.T
    selected = lax.dynamic_update_slice_in_dim(
        jnp.zeros((n, r), L.dtype), jnp.eye(r, dtype=L.dtype), start, axis=0
    )
    U = jnp.concatenate((selected, difference), axis=1)
    # The covariance change is U J U.T, where J = [[-delta_C, I], [I, 0]].
    zero, eye = jnp.zeros((r, r), L.dtype), jnp.eye(r, dtype=L.dtype)
    delta_C = lax.dynamic_slice_in_dim(difference, start, r, axis=0)
    J_inverse = jnp.block([[zero, eye], [eye, delta_C]])
    W = solve_triangular(L, U, lower=True)

    # Each leading submatrix gives a small system for one factor column.
    # These independent 2r-by-2r solves replace the successive full-factor passes.
    outer = W[:, :, None] * W[:, None, :]
    prefix = jnp.concatenate(
        (jnp.zeros_like(outer[:1]), jnp.cumsum(outer, axis=0)[:-1]), axis=0
    )
    B = jnp.linalg.solve(J_inverse + prefix, W[..., None])[..., 0]
    diagonal = jnp.sqrt(1 + jnp.sum(W * B, axis=1))
    H = B / diagonal[:, None]

    # The whitened factor has diagonal `diagonal` and off-diagonal W[i] @ H[j].
    # Multiply it by L in panels, avoiding either an n-by-n matrix product or
    # an n-by-n-by-2r array of partial sums.
    width = min(64, n)
    padding = (-n) % width
    padded_L = jnp.pad(L, ((0, padding), (0, padding)))
    padded_U = jnp.pad(U, ((0, padding), (0, 0)))
    W = jnp.pad(W, ((0, padding), (0, 0)))
    H = jnp.pad(H, ((0, padding), (0, 0)))
    diagonal = jnp.pad(diagonal, ((0, padding),), constant_values=1)
    size, rank = W.shape
    panels = size // width
    factor_panels = padded_L.reshape(size, panels, width).transpose(1, 0, 2)
    W = W.reshape(panels, width, rank)
    H = H.reshape(panels, width, rank)
    local = jnp.tril(W @ H.swapaxes(-1, -2), -1)
    local += jnp.eye(width, dtype=L.dtype) * diagonal.reshape(panels, 1, width)
    tail = padded_U[None] - jnp.cumsum(factor_panels @ W, axis=0)
    result = factor_panels @ local + tail @ H.swapaxes(-1, -2)
    result = jnp.tril(result.transpose(1, 0, 2).reshape(size, size)[:n, :n])
    leading = jnp.arange(n) < start
    return jnp.where(leading[:, None] & leading[None, :], L, result)


def chol_replace(L: Array, start: Array | int, K: Array, C: Array) -> Array:
    """Replace variables start:start+r, preserving their positions and matrix size.

    C:(r,r) describes the new block. K:(n-r,r) has rows for the variables
    before the block, followed by those after it, in their original order.
    Its columns follow the new block's order. start is an integer scalar in
    [0,n-r] and may be traced. r is inferred from C's static shape.
    r=0 leaves the lower factor unchanged, and r=n factors C.
    Floating inputs share a real float32/64 dtype. C must be symmetric and
    the target covariance positive definite.

    One replacement uses a rank-one update and downdate. Larger blocks apply
    the rank-2r covariance change together, with O(n^2*r + n*r^3) work.
    The block algorithm solves with the original factor, so its accuracy remains
    sensitive to that factor's conditioning even if the target is better conditioned.
    Supports jit, vmap and first-order JVP and VJP rules. C uses the
    symmetric-input convention. Use chol_replace_last for trailing replacement.
    """
    L = _factor(L)
    start, K, C = _replacement(start, K, C, L)
    n, r = L.shape[0], C.shape[0]
    if r == 0:
        return L
    C = (C + C.T) / 2
    if r == n:
        return cholesky(C)
    if r == 1:
        return _replace_one(L, start, K[:, 0], C[0, 0])
    return _replace_block(L, start, K, C)


def chol_replace_last(L: Array, K: Array, C: Array) -> Array:
    """Replace the final r variables of an (n,n) lower Cholesky factor.

    K has shape (n-r,r), C has shape (r,r); r is inferred from C's static shape.
    The target is [[A, K], [K.T, C]], with A = L[:n-r,:n-r] @ L[:n-r,:n-r].T.
    C must be symmetric and the target positive definite. All inputs share a
    real float32/64 dtype. Only L's retained leading lower triangle is used;
    the discarded tail and unused upper triangle may contain arbitrary values.

    Returns an (n,n) lower factor, preserving the leading block exactly. r=0
    returns the lower input, and r=n factorizes C. Supports jit, vmap, JVP and
    reverse AD using cholops' symmetry-preserving Cholesky reverse rule.
    The symmetric-input Cholesky convention applies to C and its tangents.
    The Schur subtraction can lose relative accuracy when conditional variances
    are very small.

    Arithmetic costs O((n-r)^2*r + (n-r)*r^2 + r^3), plus writing the result.
    """
    L = _factor(L)
    K, C = jnp.asarray(K), jnp.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square trailing covariance block")
    r = C.shape[0]
    m = L.shape[0] - r
    if m < 0 or K.shape != (m, r):
        raise ValueError("K must have shape (n-r,r), with 0 <= r <= n")
    if K.dtype != L.dtype or C.dtype != L.dtype:
        raise TypeError("L, K and C must have the same dtype")
    return _append_block(_delete_last(L, r), K, C)
