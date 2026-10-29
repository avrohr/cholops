"""Positive and signed rank-one updates, diagonal updates and their derivatives."""

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.linalg import solve_triangular

from ._autodiff import tangent_map
from ._validation import _factor, _index, _scalar, _vector


def _rank1(L: Array, u: Array, alpha: Array, start: Array | int = 0) -> Array:
    n = L.shape[0]
    if n == 0:
        return L
    rows = jnp.arange(n)

    def column_step(work, inputs):
        j, column = inputs

        def modify(work):
            diagonal = column[j]
            s = work[j] / diagonal
            c = jnp.sqrt(1 + alpha * s * s)
            updated = (column + (alpha * s) * work) / c
            updated = jnp.where(rows > j, updated, 0)
            updated = updated.at[j].set(diagonal * c)
            residual = jnp.where(rows > j, c * work - s * updated, 0)
            return residual, updated

        return lax.cond(j >= start, modify, lambda work: (work, column), work)

    # Only a vector is carried. AD saves O(n^2) scalar data rather than a
    # sequence of n whole matrix carries, which could require O(n^3) storage.
    _, columns = lax.scan(column_step, u, (rows, L.T))
    return columns.T


def chol_rank1(L: Array, u: Array, alpha: Array | float = 1.0) -> Array:
    """Return chol(L @ L.T + alpha * outer(u, u)).

    L has shape (n, n), u has shape (n,), and alpha is a signed scalar.
    The target must be positive definite. Uses a signed rotation recurrence
    with O(n^2) work, without forming the matrix or taking sqrt(abs(alpha)).
    The weight derivative is therefore defined at alpha=0.

    Supports jit, vmap, JVP and reverse-mode AD on the positive-definite domain.
    Use vmap explicitly for batches.
    """
    L = _factor(L)
    u = _vector(u, L.shape[0], L, "u")
    alpha = _scalar(alpha, L, "alpha")
    return _rank1(L, u, alpha)


def chol_diag_update(L: Array, index: Array | int, delta: Array | float) -> Array:
    """Factor L @ L.T + delta * e_index @ e_index.T, preserving ordering.

    delta is an additive, signed change to the matrix diagonal, not to L's
    diagonal. The target must be positive definite. Earlier factor columns
    are unchanged; the recurrence starts at index, though loop shapes and
    packing still depend on the full n. index is a scalar in [0, n).

    Supports jit, vmap, JVP and reverse-mode AD, including delta=0. No
    derivative is defined for index. Invalid runtime indices give NaNs.
    Invalid concrete Python indices raise.
    """
    L = _factor(L)
    index = _index(index, L.shape[0])
    delta = _scalar(delta, L, "delta")
    u = jax.nn.one_hot(index, L.shape[0], dtype=L.dtype)
    valid = (index >= 0) & (index < L.shape[0])
    result = _rank1(L, u, delta, start=index)
    return result * jnp.where(valid, 1, jnp.asarray(jnp.nan, dtype=result.dtype))


@jax.custom_jvp
def _exclusive_prefix(x: Array) -> Array:
    return jnp.concatenate(
        (jnp.zeros_like(x[..., :1]), jnp.cumsum(x, axis=-1)[..., :-1]), axis=-1
    )


def _prefix_transpose(_, cotangent):
    return _exclusive_prefix(cotangent[..., ::-1])[..., ::-1]


def _prefix_forward(_, tangent):
    return _exclusive_prefix(tangent)


@_exclusive_prefix.defjvp
def _prefix_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = _exclusive_prefix(x)
    # The transpose is another exclusive scan. An explicit rule avoids XLA
    # lowering the differentiated slice/pad as overlapping full-width windows.
    return y, tangent_map(_prefix_transpose, (), dx, y, _prefix_forward)


def _prefix_norm(v: Array) -> Array:
    """Compute sqrt(1 + prefix_sum(v**2)) without squaring large inputs."""

    def combine(a, b):
        scale = jnp.maximum(a, b)
        ratio = jnp.minimum(a, b) / jnp.where(scale == 0, 1, scale)
        return scale * jnp.sqrt(1 + ratio * ratio)

    values = jnp.concatenate((jnp.ones_like(v[:1]), jnp.abs(v)))
    return lax.associative_scan(combine, values)[1:]


@jax.named_call
def _solve(L: Array, u: Array) -> Array:
    return solve_triangular(L, u, lower=True)


@jax.named_call
def _assemble(L: Array, u: Array, v: Array) -> tuple[Array, Array, Array, Array]:
    rho = _prefix_norm(v)
    previous = jnp.concatenate((jnp.ones_like(rho[:1]), rho[:-1]))
    c = previous / rho
    b = (v / rho) / previous
    residual = u[:, None] - _exclusive_prefix(L * v[None, :])
    Y = jnp.tril(L * c[None, :] + residual * b[None, :])
    # Reuse the solve's diagonal residual instead of subtracting it a second time.
    diagonal = jnp.arange(L.shape[0])
    Y = Y.at[diagonal, diagonal].set(jnp.diag(L) / c)
    return Y, rho, c, b


@jax.named_call
def _backward_rhs(
    L: Array, u: Array, v: Array, rho: Array, b: Array, Y: Array, G: Array, *, sign=1
) -> tuple[Array, Array]:
    B = _exclusive_prefix((G * b[None, :])[:, ::-1])[:, ::-1]
    residual = u[:, None] - jnp.cumsum(L * v[None, :], axis=1)
    E = jnp.tril(sign * (residual / rho[None, :]) / rho[None, :], -1)
    return jnp.sum(G * E - Y * B, axis=0), B


@jax.custom_jvp
def _update(L: Array, u: Array) -> Array:
    return _assemble(L, u, _solve(L, u))[0]


def _update_fwd(L: Array, u: Array):
    v = _solve(L, u)
    Y, rho, c, b = _assemble(L, u, v)
    return Y, (L, u, v, rho, c, b, Y)


def _update_bwd(saved, cotangent):
    L, u, v, rho, c, b, Y = saved
    G = jnp.tril(cotangent)
    rhs, B = _backward_rhs(L, u, v, rho, b, Y, G)
    # Reversing and summing the rotation recurrences gives this solve with the
    # UPDATED factor. Differentiating the whitening solve directly instead uses
    # L.T and can lose accuracy even when the updated covariance is well-conditioned.
    z = solve_triangular(Y, rhs, lower=True, trans="T")
    grad_L = jnp.tril(G * c[None, :] - (B + z[:, None]) * v[None, :])
    grad_u = z + jnp.sum(G * b[None, :], axis=1)
    return grad_L, grad_u


def _update_pushforward(saved, tangents):
    L, u, v, rho, c, b, Y = saved
    dL, du = tangents
    dL = jnp.tril(dL)
    # This is the transpose of _update_bwd, with its stable UPDATED-factor solve.
    # A matvec avoids retaining the dense dL*v product merely for a row sum.
    q = solve_triangular(Y, du - dL @ v, lower=True)
    residual = u[:, None] - jnp.cumsum(L * v[None, :], axis=1)
    E = jnp.tril((residual / rho[None, :]) / rho[None, :], -1)
    prefix = _exclusive_prefix(dL * v[None, :] + Y * q[None, :])
    return jnp.tril(dL * c[None, :] + (du[:, None] - prefix) * b[None, :] + E * q)


@_update.defjvp
def _update_jvp(primals, tangents):
    Y, saved = _update_fwd(*primals)
    return Y, tangent_map(_update_bwd, saved, tangents, Y, _update_pushforward)


def chol_rank1_positive(L: Array, u: Array) -> Array:
    """Return chol(L @ L.T + outer(u, u)) with quadratic work and AD storage.

    L has shape (n,n) and u has shape (n,), with a common real float32/float64
    dtype. L must have a positive diagonal. Supports jit, vmap, and first-order
    JVP and VJP rules, including u=0 and empty factors. Use chol_rank1 for
    signed weights and downdates.

    The algorithm rescales intermediate norms to avoid directly squaring
    large values.
    """
    L = _factor(L)
    u = _vector(u, L.shape[0], L, "u")
    return _update(L, u) if L.shape[0] else L
