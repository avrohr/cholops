"""Cholesky factor objects using the tensor operations."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from ..tensors import (
    chol_append,
    chol_delete,
    chol_delete_append,
    chol_delete_last,
    chol_diag_update,
    chol_rank1,
    chol_rank1_positive,
    chol_replace,
    chol_replace_last,
    cholesky,
)
from ..tensors._validation import _factor, _vector
from ._append import _append_solution
from ._consumers import _logdet, _matrix_rhs, _quadratic, _rhs, _solve, _whiten


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Cholesky:
    """Canonical lower factor, stored as one dynamic PyTree leaf ``factor``.

    Construct with from_matrix or from_factor. Inputs and solutions returned by
    solve follow the matrix's row and column order. Whitening returns factor
    coordinates. Floating arrays share a real float32 or float64 dtype.
    The represented matrix must remain positive definite.
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

    def whiten(self, rhs: Array) -> Array:
        """Return W:(n,r) solving L @ W = rhs with one triangular solve.

        rhs:(n,r) follows the matrix order and shares the factor dtype.
        Use (n,1) for one column. Rows of W are whitened coordinates.
        Columns retain their input order.
        W.T @ W equals rhs.T @ A^{-1} @ rhs for the represented matrix A.
        Cross products require results from the same factor state.
        Supports jit, vmap and first-order differentiation.
        """
        return _whiten(self.factor, _matrix_rhs(self.factor, rhs))

    def logdet(self) -> Array:
        """Scalar log determinant of the represented covariance."""
        return _logdet(self.factor)

    def quadratic(self, rhs: Array) -> Array:
        """Inverse quadratic form: scalar for (n,), per column for (n,r)."""
        return _quadratic(self.factor, _rhs(self.factor, rhs))

    def append(self, K: Array, C: Array) -> "Cholesky":
        """Append r rows and columns with K:(n,r) and C:(r,r).

        The resulting matrix is [[A, K], [K.T, C]], where A is the
        matrix represented by this factor. C must be symmetric and
        the resulting matrix positive definite.
        Use (n,1) and (1,1) arrays for a single row and column.
        """
        return type(self)(chol_append(self.factor, K, C))

    def append_solve(
        self, K: Array, C: Array, *, solution: Array, rhs: Array
    ) -> tuple["Cholesky", Array]:
        """Append a block and update a solution using the block factorization.

        K:(n,r), C:(r,r), solution:(n,), rhs:(n+r,) share the factor dtype.
        solution must solve the old system for rhs[:n]. Those RHS entries
        must be unchanged. Returns the appended factor and its new solution.
        """
        updated = self.append(K, C)
        n = self.factor.shape[0]
        solution = _vector(solution, n, self.factor, "solution")
        rhs = _vector(rhs, updated.factor.shape[0], self.factor, "rhs")
        coefficients = _append_solution(
            updated.factor, jnp.asarray(K), solution, rhs[n:]
        )
        return updated, coefficients

    def delete(self, indices: Array) -> "Cholesky":
        """Delete indices:(r,), preserving survivor order and returning size n-r.

        Indices must be distinct and in bounds. Their count is static, but
        values may be traced. Traced values are not checked at runtime.
        """
        return type(self)(chol_delete(self.factor, indices))

    def delete_last(self, r: int) -> "Cholesky":
        """Remove the last r variables by slicing. r must be static and in [0,n]."""
        return type(self)(chol_delete_last(self.factor, r))

    def replace(self, start: Array | int, K: Array, C: Array) -> "Cholesky":
        """Replace variables start:start+r, preserving order and shape.

        C:(r,r) describes the new block. K:(n-r,r) has rows for the variables
        before and after it, in their original order. start may be traced and
        must be in [0,n-r]. The target must be positive definite.
        Use (n-1,1) and (1,1) arrays for one replacement.
        """
        return type(self)(chol_replace(self.factor, start, K, C))

    def delete_append(self, indices: Array, K: Array, C: Array) -> "Cholesky":
        """Delete indices:(r,) and append s variables, returning size n-r+s.

        K:(n-r,s) follows survivor order. C:(s,s) must be symmetric and the
        target positive definite. Indices must be distinct and in bounds.
        The result is ordered as survivors followed by the appended block.
        """
        return type(self)(chol_delete_append(self.factor, indices, K, C))

    def replace_last(self, K: Array, C: Array) -> "Cholesky":
        """Replace the last r variables with K:(n-r,r), C:(r,r), keeping size n.

        Uses a Schur complement with the symmetry-preserving Cholesky reverse rule.
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
