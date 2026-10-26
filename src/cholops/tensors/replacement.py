"""Index-preserving single-variable and trailing-block replacement."""

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.linalg import solve_triangular

from ._autodiff import tangent_map
from ._indexing import _insert, _keep, _principal
from ._validation import _factor, _index, _scalar, _vector
from .append import _append
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
def _replace_forward(L, index, k, kappa):
    keep = _keep(L.shape[0], index)
    base = _principal(L, index)
    removed_column = jnp.where(keep > index, L[keep, index], 0)
    retained, positive = _update_fwd(base, removed_column)
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
    # retained is shared by both rank-one rules; save it once. The covariance
    # inputs and assembled output need not be retained by the integrated VJP.
    saved = (index, positive, row, diagonal, negative[1:])
    return result, saved


@jax.named_call
def _replace_backward(saved, cotangent):
    index, positive, row, diagonal, negative_tail = saved
    retained = positive[-1]
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
    grad_base, grad_removed = _update_bwd(positive, grad_retained)
    grad_L = _insert(
        grad_base,
        index,
        jnp.zeros_like(row),
        jnp.where(keep > index, grad_removed, 0),
        jnp.zeros_like(diagonal),
    )
    return grad_L, grad_k, grad_kappa


@jax.custom_jvp
@jax.named_call
def _replace_interior(L, index, k, kappa):
    return _replace_forward(L, index, k, kappa)[0]


@_replace_interior.defjvp
def _replace_jvp(primals, tangents):
    Y, saved = _replace_forward(*primals)
    dL, _, dk, dkappa = tangents
    return Y, tangent_map(_replace_backward, saved, (dL, dk, dkappa), Y)


def chol_replace(L: Array, index: Array | int, k: Array, kappa: Array | float) -> Array:
    """Replace one covariance row/column without changing any other index.

    L is a real lower Cholesky factor of shape (n,n), n >= 1. k has shape
    (n-1,) in original index order with index omitted; kappa is the new diagonal.
    All floating inputs have the same float32/float64 dtype. The upper triangle
    of L is ignored. The target covariance must be positive definite.

    Supports jit, vmap, runtime scalar indices, and first-order JVP and VJP rules for
    L, k and kappa with quadratic work/storage per operation.
    index must be in [0,n). Traced indices are not checked at runtime.
    Near-singular targets remain sensitive to the downdate margin subtraction.
    Final-position replacement uses a bordering solve without a downdate.
    """
    L = _factor(L)
    index = _index(index, L.shape[0])
    k = _vector(k, L.shape[0] - 1, L, "k")
    kappa = _scalar(kappa, L, "kappa")
    if L.shape[0] == 1:
        return _append(L[:0, :0], k, kappa)
    return lax.cond(
        index == L.shape[0] - 1,
        lambda _: _append(L[:-1, :-1], k, kappa),
        lambda _: _replace_interior(L, index, k, kappa),
        operand=None,
    )


def chol_replace_last(L: Array, K: Array, C: Array) -> Array:
    """Replace the final r variables of an (n,n) lower Cholesky factor.

    K has shape (n-r,r), C has shape (r,r); r is inferred from C's static shape.
    The target is [[A, K], [K.T, C]], with A = L[:n-r,:n-r] @ L[:n-r,:n-r].T.
    C must be symmetric and the target positive definite. All inputs share a
    real float32/64 dtype. Only L's retained leading lower triangle is used;
    the discarded tail and unused upper triangle may contain arbitrary values.

    Returns an (n,n) lower factor, preserving the leading block exactly. r=0
    returns the lower input, and r=n factorizes C. Supports jit, vmap, JVP and
    reverse AD using native JAX rules. JAX's symmetric-input Cholesky
    convention applies to C and its tangents. The Schur subtraction can lose
    relative accuracy when conditional variances are very small.

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
    if r == 0:
        return L
    leading = L[:m, :m]
    V = solve_triangular(leading, K, lower=True) if m else K
    trailing = jnp.linalg.cholesky(C - V.T @ V)
    return jnp.concatenate(
        (
            jnp.concatenate((leading, jnp.zeros_like(K)), axis=1),
            jnp.concatenate((V.T, trailing), axis=1),
        ),
        axis=0,
    )
