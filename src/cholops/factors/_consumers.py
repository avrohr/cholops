"""Whitening, solves and reductions shared by the factor classes."""

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular


def _rhs(L: Array, rhs: Array) -> Array:
    rhs = jnp.asarray(rhs)
    if rhs.ndim not in (1, 2) or rhs.shape[0] != L.shape[0]:
        raise ValueError("rhs must have shape (n,) or (n,r)")
    if rhs.dtype != L.dtype:
        raise TypeError("rhs and the factor must have the same dtype")
    return rhs


def _matrix_rhs(L: Array, rhs: Array) -> Array:
    rhs = _rhs(L, rhs)
    if rhs.ndim != 2:
        raise ValueError("rhs must have shape (n,r); use (n,1) for one column")
    return rhs


def _whiten(L: Array, rhs: Array) -> Array:
    return solve_triangular(L, rhs, lower=True)


def _solve(L: Array, rhs: Array) -> Array:
    z = _whiten(L, rhs)
    return solve_triangular(L, z, lower=True, trans="T")


def _logdet(L: Array) -> Array:
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))


def _quadratic(L: Array, rhs: Array) -> Array:
    z = _whiten(L, rhs)
    return jnp.sum(z * z, axis=0)
