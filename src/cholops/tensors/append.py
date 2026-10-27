"""Append variables through scalar or block bordering solves."""

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from ._validation import _factor, _finish, _scalar, _valid_factor, _vector


def _append(L: Array, k: Array, kappa: Array) -> Array:
    n = L.shape[0]
    v = solve_triangular(L, k, lower=True) if n else k
    diagonal = jnp.sqrt(kappa - jnp.vdot(v, v))
    result = jnp.zeros((n + 1, n + 1), dtype=L.dtype)
    return result.at[:n, :n].set(L).at[n, :n].set(v).at[n, n].set(diagonal)


def chol_append(L: Array, k: Array, kappa: Array | float) -> Array:
    """Factor [[L @ L.T, k], [k.T, kappa]], returning shape (n+1, n+1).

    k has shape (n,) in the current physical ordering; kappa is scalar.
    The Schur complement kappa - ||solve(L, k)||^2 must be strictly positive.
    Empty (0, 0) factors are supported. Work is O(n^2).
    Supports jit, vmap, JVP and reverse-mode AD. Numerical failure gives NaNs.
    """
    L = _factor(L)
    k = _vector(k, L.shape[0], L, "k")
    kappa = _scalar(kappa, L, "kappa")
    valid = _valid_factor(L) & jnp.all(jnp.isfinite(k)) & jnp.isfinite(kappa)
    return _finish(_append(L, k, kappa), valid)


def _append_block(L: Array, K: Array, C: Array) -> Array:
    """Border a validated lower factor with a covariance block."""
    if C.shape[0] == 0:
        return L
    V = solve_triangular(L, K, lower=True) if L.shape[0] else K
    trailing = jnp.linalg.cholesky(C - V.T @ V)
    return jnp.concatenate(
        (
            jnp.concatenate((L, jnp.zeros_like(K)), axis=1),
            jnp.concatenate((V.T, trailing), axis=1),
        ),
        axis=0,
    )


def chol_append_block(L: Array, K: Array, C: Array) -> Array:
    """Append r variables, returning a lower factor of shape (n+r,n+r).

    The target is [[L @ L.T, K], [K.T, C]], with K:(n,r) in the current
    physical ordering and symmetric C:(r,r). All inputs share a real
    float32/64 dtype. L's unused upper triangle is ignored.
    The target, including the Schur complement C - V.T @ V for V=solve(L,K),
    must be positive definite. No jitter or clipping is applied.

    Preserves the leading lower factor exactly. r=0 returns the lower input;
    n=0 factorizes C. Supports jit, vmap and first-order JVP and reverse AD
    using native JAX rules. JAX's symmetric-input Cholesky convention applies
    to C and its tangents. A non-SPD Schur complement produces NaNs; the
    subtraction can lose accuracy for small conditional variances.

    Arithmetic costs O(n^2*r + n*r^2 + r^3), plus writing the result.
    """
    L = _factor(L)
    K, C = jnp.asarray(K), jnp.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square trailing covariance block")
    if K.shape != (L.shape[0], C.shape[0]):
        raise ValueError("K must have shape (n,r)")
    if K.dtype != L.dtype or C.dtype != L.dtype:
        raise TypeError("L, K and C must have the same dtype")
    return _append_block(L, K, C)
