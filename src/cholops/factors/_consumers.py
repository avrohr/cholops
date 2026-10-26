"""Solves, log determinants and quadratic forms shared by the factor classes."""

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


def _solve(L: Array, rhs: Array) -> Array:
    z = solve_triangular(L, rhs, lower=True)
    return solve_triangular(L, z, lower=True, trans="T")


def _logdet(L: Array) -> Array:
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))


def _quadratic(L: Array, rhs: Array) -> Array:
    z = solve_triangular(L, rhs, lower=True)
    return jnp.sum(z * z, axis=0)
