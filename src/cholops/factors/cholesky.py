"""Cholesky factor objects using the tensor operations."""

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
    chol_replace,
    chol_replace_last,
    cholesky,
)
from ..tensors._validation import _factor
from ._consumers import _logdet, _quadratic, _rhs, _solve


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Cholesky:
    """Canonical lower factor, stored as one dynamic PyTree leaf ``factor``.

    Construct with from_matrix or from_factor. All inputs and outputs follow
    the matrix's row and column order. Floating arrays share a real float32 or
    float64 dtype. The represented matrix must remain positive definite.
    Modifications return new objects, leaving the original unchanged.
    Supports jit, vmap and first-order differentiation.
    """

    factor: Array

    @classmethod
    def from_matrix(cls, matrix: Array) -> "Cholesky":
        """Factor real (n,n) input with the symmetry-preserving reverse rule.

        The symmetric part must be positive definite. Batch with vmap.
        """
        matrix = jnp.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("matrix must be unbatched; batch with jax.vmap")
        return cls(cholesky(matrix))

    @classmethod
    def from_factor(cls, factor: Array) -> "Cholesky":
        """Wrap a real (n,n) lower factor, ignoring its unused upper triangle.

        Does not change the derivative of the function that produced factor.
        Supply a valid factor with a positive diagonal. Empty factors are supported.
        """
        return cls(_factor(factor))

    def solve(self, rhs: Array) -> Array:
        """Solve the covariance system for a canonical (n,) or (n,r) RHS."""
        return _solve(self.factor, _rhs(self.factor, rhs))

    def logdet(self) -> Array:
        """Scalar log determinant of the represented covariance."""
        return _logdet(self.factor)

    def quadratic(self, rhs: Array) -> Array:
        """Inverse quadratic form: scalar for (n,), per column for (n,r)."""
        return _quadratic(self.factor, _rhs(self.factor, rhs))

    def append(self, k: Array, kappa: Array | float) -> "Cholesky":
        """Append covariances k:(n,) and diagonal kappa; return size n+1.

        The target covariance must be positive definite.
        """
        return type(self)(chol_append(self.factor, k, kappa))

    def append_block(self, K: Array, C: Array) -> "Cholesky":
        """Append r variables with K:(n,r), C:(r,r), returning size n+r.

        C must be symmetric and the target covariance positive definite.
        r is static, inferred from C; r=0 and empty factors are supported.
        Uses native JAX first-order differentiation of the Schur complement.
        """
        return type(self)(chol_append_block(self.factor, K, C))

    def delete(self, index: Array | int) -> "Cholesky":
        """Delete one variable, preserving survivor order; return size n-1.

        index must be in [0,n). Check traced indices before calling this method,
        since they are not checked at runtime.
        """
        return type(self)(chol_delete(self.factor, index))

    def replace(self, index: Array | int, k: Array, kappa: Array | float) -> "Cholesky":
        """Replace at index, preserving every other index and the (n,n) shape.

        k:(n-1,) follows survivor order. kappa is the new diagonal.
        The target must be positive definite and index must be in [0,n).
        Traced indices are not checked at runtime.
        """
        return type(self)(chol_replace(self.factor, index, k, kappa))

    def delete_append(
        self, index: Array | int, k: Array, kappa: Array | float
    ) -> "Cholesky":
        """Delete index and append its replacement at the end, keeping size n.

        k:(n-1,) follows survivor order. kappa is the new diagonal.
        The returned factor is ordered as survivors followed by the replacement.
        The target must be positive definite and index must be in [0,n).
        Traced indices are not checked at runtime.
        """
        return type(self)(chol_delete_append(self.factor, index, k, kappa))

    def replace_last(self, K: Array, C: Array) -> "Cholesky":
        """Replace the last r variables with K:(n-r,r), C:(r,r), keeping size n.

        Uses a Schur complement with native JAX Cholesky differentiation.
        C must be symmetric and the target must be positive definite.
        r is static, inferred from C; r=0 and r=n are supported.
        """
        return type(self)(chol_replace_last(self.factor, K, C))

    def diag_update(self, index: Array | int, delta: Array | float) -> "Cholesky":
        """Add signed delta to covariance diagonal index, preserving order.

        The derivative with respect to delta is retained at delta=0.
        The target must be positive definite.
        """
        return type(self)(chol_diag_update(self.factor, index, delta))

    def rank1(self, u: Array, alpha: Array | float = 1.0) -> "Cholesky":
        """Add alpha * outer(u,u), u:(n,), preserving order and size.

        alpha may be positive, negative or zero. The weight derivative is
        retained at zero. The target must be positive definite.
        """
        return type(self)(chol_rank1(self.factor, u, alpha))

    def rank1_positive(self, u: Array) -> "Cholesky":
        """Add outer(u,u), u:(n,), using the optimized positive-update algorithm."""
        return type(self)(chol_rank1_positive(self.factor, u))
