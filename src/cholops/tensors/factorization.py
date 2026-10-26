"""JAX Cholesky factorization with a symmetry-preserving reverse rule.

The reverse rule symmetrizes its intermediate matrix before triangular solves.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from ._autodiff import tangent_map


def _sym(A: Array) -> Array:
    return (A + jnp.swapaxes(A, -1, -2)) * 0.5


def pullback(L: Array, G: Array) -> Array:
    """Return the symmetric covariance cotangent from a lower-factor cotangent.

    Both arrays have shape (...,n,n). Removes the irrelevant skew component
    before either ill-conditioned triangular solve. Uses cubic work and quadratic
    storage, like the ordinary dense Cholesky pullback.
    """
    P = jnp.tril(jnp.swapaxes(L, -1, -2) @ jnp.tril(G))
    index = jnp.arange(L.shape[-1])
    P = P.at[..., index, index].multiply(0.5)
    Z = solve_triangular(L, _sym(P), lower=True, trans="T")
    Q = solve_triangular(L, jnp.swapaxes(Z, -1, -2), lower=True, trans="T")
    return _sym(jnp.swapaxes(Q, -1, -2))


def _factorize(A):
    A = jnp.asarray(A)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be square in its last two dimensions")
    if A.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("A must have a real float32 or float64 dtype")
    return jnp.linalg.cholesky(A)


@jax.custom_jvp
def cholesky(A: Array) -> Array:
    """Compute a lower Cholesky factor with a symmetry-preserving reverse rule.

    A is a real float32/64 square matrix (or batch); the symmetric part must be
    positive definite. Supports jit and vmap. Differentiation is supported only
    through first order, in both forward and reverse mode.
    """
    return _factorize(A)


@cholesky.defjvp
def _cholesky_jvp(primals, tangents):
    (A,), (dA,) = primals, tangents
    L = _factorize(A)
    return L, tangent_map(pullback, L, dA, L)
