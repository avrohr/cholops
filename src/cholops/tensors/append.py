"""Append a row and column through a bordering solve."""

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
