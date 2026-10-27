"""Cholesky factor objects that manage their internal variable ordering."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from ..tensors import (
    chol_append,
    chol_append_block,
    chol_delete,
    chol_delete_append,
    chol_diag_update,
    chol_rank1,
    chol_rank1_positive,
    cholesky,
)
from ..tensors._indexing import _keep
from ..tensors._validation import _factor, _index, _scalar, _vector
from ._consumers import _logdet, _quadratic, _rhs, _solve


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PermutedCholesky:
    """Physical lower factor and inverse permutations, all dynamic PyTree leaves.

    order[physical] is logical; inverse_order[logical] is physical. Construct via
    from_matrix or from_factor. All method inputs and outputs use logical order.
    The stored factor is lower triangular in physical order. Floating arrays
    share a real float32 or float64 dtype, and the represented matrix must remain
    positive definite. Modifications return new objects.
    Supports jit, vmap and first-order differentiation of continuous inputs.
    """

    physical_factor: Array
    order: Array
    inverse_order: Array

    @classmethod
    def from_matrix(cls, matrix: Array) -> "PermutedCholesky":
        """Factor a real (n,n) matrix with the symmetry-preserving reverse rule.

        The symmetric part must be positive definite. Inputs use float32 or
        float64. Batch with vmap.
        """
        matrix = jnp.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("matrix must be unbatched; batch with jax.vmap")
        return cls.from_factor(cholesky(matrix))

    @classmethod
    def from_factor(cls, factor: Array) -> "PermutedCholesky":
        """Wrap an existing (n,n) lower factor in identity logical order.

        This does not change the derivative of the operation that produced
        factor. Use from_matrix to apply cholops' Cholesky reverse rule.
        """
        factor = _factor(factor)
        order = jnp.arange(factor.shape[0])
        return cls(factor, order, order)

    def append(self, k: Array, kappa: Array | float) -> "PermutedCholesky":
        """Append a variable last in logical order, returning size n+1.

        k:(n,) follows the existing logical order. kappa is the new diagonal.
        """
        L = self.physical_factor
        n = L.shape[0]
        k = _vector(k, n, L, "k")
        result = chol_append(L, k[self.order], kappa)
        last = jnp.asarray([n], dtype=self.order.dtype)
        return type(self)(
            result,
            jnp.concatenate((self.order, last)),
            jnp.concatenate((self.inverse_order, last)),
        )

    def append_block(self, K: Array, C: Array) -> "PermutedCholesky":
        """Append r variables last in logical order, returning size n+r.

        K:(n,r) has rows in existing logical order and columns in new-variable
        order. C:(r,r) follows the new-variable order and must be symmetric.
        The target must be positive definite. r is static, inferred from C;
        r=0 and empty factors are supported, with native first-order AD.
        """
        L = self.physical_factor
        n = L.shape[0]
        K, C = jnp.asarray(K), jnp.asarray(C)
        if K.ndim != 2 or K.shape[0] != n:
            raise ValueError("K must have shape (n,r)")
        if K.dtype != L.dtype or C.dtype != L.dtype:
            raise TypeError("L, K and C must have the same dtype")
        result = chol_append_block(L, K[self.order], C)
        last = jnp.arange(n, result.shape[0], dtype=self.order.dtype)
        return type(self)(
            result,
            jnp.concatenate((self.order, last)),
            jnp.concatenate((self.inverse_order, last)),
        )

    def delete(self, index: Array | int) -> "PermutedCholesky":
        """Delete a logical variable, preserving survivor order and returning n-1.

        Remaining logical indices are renumbered from zero.
        """
        L = self.physical_factor
        n = L.shape[0]
        index = _index(index, n)
        physical = self.inverse_order[index]
        result = chol_delete(L, physical)
        survivors = self.order[_keep(n, physical)]
        order = survivors - (survivors > index)
        inverse = self.inverse_order[_keep(n, index)]
        inverse = inverse - (inverse > physical)
        return type(self)(result, order, inverse)

    def replace(
        self, index: Array | int, k: Array, kappa: Array | float
    ) -> "PermutedCholesky":
        """Return a new factor after replacing logical slot index.

        k has shape (n-1,) in logical order with index omitted; scalar kappa
        is the new covariance diagonal. Typed arrays share the factor dtype.
        The target must be positive definite and index must be in [0,n).
        Traced indices are not checked at runtime.
        Logical order and shape are preserved.
        """
        L = self.physical_factor
        n = L.shape[0]
        index = _index(index, n)
        k = _vector(k, n - 1, L, "k")
        kappa = _scalar(kappa, L, "kappa")
        physical = self.inverse_order[index]
        keep = jnp.arange(n - 1) + (jnp.arange(n - 1) >= physical)
        survivors = self.order[keep]
        compact = survivors - (survivors > index)
        result = chol_delete_append(L, physical, k[compact], kappa)
        order = jnp.concatenate((survivors, index[None]))
        inverse = self.inverse_order - (self.inverse_order > physical)
        inverse = inverse.at[index].set(n - 1)
        return type(self)(result, order, inverse)

    def delete_append(
        self, index: Array | int, k: Array, kappa: Array | float
    ) -> "PermutedCholesky":
        """Delete a logical variable and append its replacement last.

        k:(n-1,) follows logical survivor order. kappa is the new diagonal.
        Unlike replace, this changes logical order to survivors then replacement.
        """
        L = self.physical_factor
        n = L.shape[0]
        index = _index(index, n)
        k = _vector(k, n - 1, L, "k")
        physical = self.inverse_order[index]
        survivors = self.order[_keep(n, physical)]
        compact = survivors - (survivors > index)
        result = chol_delete_append(L, physical, k[compact], kappa)
        last = jnp.asarray([n - 1], dtype=self.order.dtype)
        order = jnp.concatenate((compact, last))
        inverse = self.inverse_order[_keep(n, index)]
        inverse = jnp.concatenate((inverse - (inverse > physical), last))
        return type(self)(result, order, inverse)

    def diag_update(
        self, index: Array | int, delta: Array | float
    ) -> "PermutedCholesky":
        """Add signed delta to one logical covariance diagonal entry."""
        index = _index(index, self.physical_factor.shape[0])
        result = chol_diag_update(
            self.physical_factor, self.inverse_order[index], delta
        )
        return type(self)(result, self.order, self.inverse_order)

    def rank1(self, u: Array, alpha: Array | float = 1.0) -> "PermutedCholesky":
        """Add alpha * outer(u, u), with u:(n,) in logical order.

        alpha may be positive, negative or zero. Logical ordering is unchanged.
        """
        L = self.physical_factor
        u = _vector(u, L.shape[0], L, "u")
        result = chol_rank1(L, u[self.order], alpha)
        return type(self)(result, self.order, self.inverse_order)

    def rank1_positive(self, u: Array) -> "PermutedCholesky":
        """Add outer(u, u), with u:(n,) in logical order, using the fast update."""
        L = self.physical_factor
        u = _vector(u, L.shape[0], L, "u")
        result = chol_rank1_positive(L, u[self.order])
        return type(self)(result, self.order, self.inverse_order)

    def solve(self, rhs: Array) -> Array:
        """Solve the covariance system for canonical vector or matrix RHS."""
        rhs = _rhs(self.physical_factor, rhs)
        return _solve(self.physical_factor, rhs[self.order])[self.inverse_order]

    def logdet(self) -> Array:
        """Log determinant of the represented covariance."""
        return _logdet(self.physical_factor)

    def quadratic(self, rhs: Array) -> Array:
        """Covariance inverse quadratic form, separately per matrix RHS column."""
        rhs = _rhs(self.physical_factor, rhs)
        return _quadratic(self.physical_factor, rhs[self.order])
