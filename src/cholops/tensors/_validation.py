"""Shared shape, dtype and index checks for tensor operations."""

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


def _blocks(K: Array, C: Array, L: Array) -> tuple[Array, Array]:
    K, C = jnp.asarray(K), jnp.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square covariance block")
    if K.shape != (L.shape[0], C.shape[0]):
        raise ValueError("K must have shape (n,r) and C must have shape (r,r)")
    if K.dtype != L.dtype or C.dtype != L.dtype:
        raise TypeError("L, K and C must have the same dtype")
    return K, C


def _replacement(start, K, C, L):
    """Check inputs for replacing r consecutive variables."""
    K, C = jnp.asarray(K), jnp.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square covariance block")
    n, r = L.shape[0], C.shape[0]
    if r > n or K.shape != (n - r, r):
        raise ValueError("K must have shape (n-r,r), with 0 <= r <= n")
    if K.dtype != L.dtype or C.dtype != L.dtype:
        raise TypeError("L, K and C must have the same dtype")
    return _index(start, n - r + 1), K, C


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
