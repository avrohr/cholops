"""Append variables through a block triangular solve and Schur complement."""

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from ._validation import _blocks, _factor, _finish, _valid_factor
from .factorization import cholesky


def _append(L: Array, k: Array, kappa: Array) -> Array:
    n = L.shape[0]
    v = solve_triangular(L, k, lower=True) if n else k
    diagonal = jnp.sqrt(kappa - jnp.vdot(v, v))
    result = jnp.zeros((n + 1, n + 1), dtype=L.dtype)
    return result.at[:n, :n].set(L).at[n, :n].set(v).at[n, n].set(diagonal)


def _append_block(L: Array, K: Array, C: Array) -> Array:
    if C.shape[0] == 0:
        return L
    V = solve_triangular(L, K, lower=True) if L.shape[0] else K
    trailing = cholesky(C - V.T @ V)
    return jnp.concatenate(
        (
            jnp.concatenate((L, jnp.zeros_like(K)), axis=1),
            jnp.concatenate((V.T, trailing), axis=1),
        ),
        axis=0,
    )


def chol_append(L: Array, K: Array, C: Array) -> Array:
    """Factor [[L @ L.T, K], [K.T, C]], returning shape (n+r, n+r).

    K has shape (n,r) in the factor's order, and C is symmetric with shape (r,r).
    Use (n,1) and (1,1) arrays to append one variable. All inputs share a real
    float32/64 dtype. The target covariance must be positive definite.
    Empty factors are supported, and r=0 returns the lower input.

    Supports jit, vmap and first-order JVP and reverse-mode AD. The Schur
    complement uses cholops' symmetry-preserving Cholesky reverse rule.
    C and its tangents follow the symmetric-input convention of cholesky.
    Work is O(n^2*r + n*r^2 + r^3), plus writing the result.
    Numerical failure gives NaNs.
    """
    L = _factor(L)
    K, C = _blocks(K, C, L)
    valid = _valid_factor(L) & jnp.all(jnp.isfinite(K)) & jnp.all(jnp.isfinite(C))
    return _finish(_append_block(L, K, C), valid)
