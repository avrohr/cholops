"""Shared shape, dtype and validity checks for tensor operations."""

import jax.numpy as jnp
from jax import Array


def _factor(L: Array) -> Array:
    L = jnp.asarray(L)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square, unbatched matrix; batch with jax.vmap")
    if L.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("L must have a real float32 or float64 dtype")
    return jnp.tril(L)


def _vector(x: Array, n: int, L: Array, name: str) -> Array:
    x = jnp.asarray(x)
    if x.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {x.shape}")
    if x.dtype != L.dtype:
        raise TypeError(f"{name} and L must have the same dtype")
    return x


def _scalar(x: Array | float, L: Array, name: str) -> Array:
    # Python scalars follow the factor precision; typed arrays are never cast.
    x = jnp.asarray(x, dtype=L.dtype) if isinstance(x, (float, int)) else jnp.asarray(x)
    if x.weak_type:
        x = x.astype(L.dtype)
    if x.shape != ():
        raise ValueError(f"{name} must be a scalar")
    if x.dtype != L.dtype:
        raise TypeError(f"{name} and L must have the same dtype")
    return x


def _index(index: Array | int, n: int) -> Array:
    if n == 0:
        raise ValueError("cannot index an empty factor")
    if isinstance(index, int) and not 0 <= index < n:
        raise ValueError(f"index must be in [0, {n})")
    index = jnp.asarray(index)
    if index.shape != () or not jnp.issubdtype(index.dtype, jnp.integer):
        raise TypeError("index must be an integer scalar")
    return index


def _valid_factor(L: Array) -> Array:
    return jnp.all(jnp.isfinite(L)) & jnp.all(jnp.diag(L) > 0)


def _finish(result: Array, valid: Array) -> Array:
    valid = valid & _valid_factor(result)
    # Keep the failure marker in arithmetic on the result, including its AD path.
    return result * jnp.where(valid, 1, jnp.asarray(jnp.nan, dtype=result.dtype))
