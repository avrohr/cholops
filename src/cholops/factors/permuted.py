"""Cholesky factor objects that manage their internal variable ordering."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from ..tensors import (
    chol_append,
    chol_delete,
    chol_delete_append,
    chol_diag_update,
    chol_rank1,
    chol_rank1_positive,
    cholesky,
)
from ..tensors._indexing import _indices, _keep_many
from ..tensors._validation import _blocks, _factor, _index, _replacement, _vector
from ._append import _append_solution
from ._consumers import _logdet, _matrix_rhs, _quadratic, _rhs, _solve, _whiten


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class PermutedCholesky:
    """Cholesky factor with a managed physical ordering.

    order[physical] is logical; inverse_order[logical] is physical. Construct via
    from_matrix or from_factor. Inputs and solutions returned by solve use
    logical order. Whitening returns factor coordinates. physical_factor
    exposes the lower-triangular factor in physical order. Floating arrays
    share a real float32 or float64 dtype, and the represented matrix must remain
    positive definite. Modifications return new objects.
    Supports jit, vmap and first-order differentiation of continuous inputs.
    """

    _upper: Array
    order: Array
    inverse_order: Array

    def __init__(self, physical_factor: Array, order: Array, inverse_order: Array):
        # Persist L.T so CPU solves can read column-major L across JIT calls.
        object.__setattr__(self, "_upper", jnp.swapaxes(physical_factor, -1, -2))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "inverse_order", inverse_order)

    @property
    def physical_factor(self) -> Array:
        """Lower-triangular factor in physical order."""
        return jnp.swapaxes(self._upper, -1, -2)

    @classmethod
    def _from_upper(cls, upper, order, inverse_order):
        """Construct from stored arrays without changing their layout."""
        result = object.__new__(cls)
        object.__setattr__(result, "_upper", upper)
        object.__setattr__(result, "order", order)
        object.__setattr__(result, "inverse_order", inverse_order)
        return result

    def tree_flatten(self):
        return (self._upper, self.order, self.inverse_order), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls._from_upper(*children)

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

    def append(self, K: Array, C: Array) -> "PermutedCholesky":
        """Append r variables last in logical order, returning size n+r.

        K:(n,r) follows the existing logical order along its rows.
        C:(r,r) must be symmetric and the target covariance positive definite.
        Use (n,1) and (1,1) arrays for a single variable.
        """
        L = self.physical_factor
        n = L.shape[0]
        K, C = _blocks(K, C, L)
        # L views _upper.T, and construction below retains the upper result
        # so subsequent CPU operations can reuse its layout.
        result = chol_append(L, K[self.order], C)
        last = jnp.arange(n, n + C.shape[0], dtype=self.order.dtype)
        return type(self)(
            result,
            jnp.concatenate((self.order, last)),
            jnp.concatenate((self.inverse_order, last)),
        )

    def append_solve(
        self, K: Array, C: Array, *, solution: Array, rhs: Array
    ) -> tuple["PermutedCholesky", Array]:
        """Append a block and update a solution, with inputs in logical order.

        K:(n,r), C:(r,r), solution:(n,), rhs:(n+r,) share the factor dtype.
        solution must solve the old system for rhs[:n]. Those RHS entries
        must be unchanged. Returned coefficients follow the new logical order.
        """
        updated = self.append(K, C)
        L = self.physical_factor
        n = L.shape[0]
        solution = _vector(solution, n, L, "solution")
        rhs = _vector(rhs, updated.physical_factor.shape[0], L, "rhs")
        coefficients = _append_solution(
            updated.physical_factor,
            jnp.asarray(K)[self.order],
            solution[self.order],
            rhs[n:],
        )
        return updated, coefficients[updated.inverse_order]

    def delete(self, indices: Array) -> "PermutedCholesky":
        """Delete logical indices:(r,), preserving survivor order.

        Indices must be distinct and in bounds. Remaining logical indices
        are renumbered from zero. The result has size n-r.
        """
        n = self.physical_factor.shape[0]
        indices = _indices(indices, n)
        physical = jnp.sort(self.inverse_order[indices])
        result = chol_delete(self.physical_factor, physical)
        survivors = self.order[_keep_many(n, physical)]
        order = survivors - jnp.searchsorted(indices, survivors)
        return type(self)(result, order, jnp.argsort(order))

    def replace(self, start: Array | int, K: Array, C: Array) -> "PermutedCholesky":
        """Replace logical variables start:start+r using deletion and block append.

        C:(r,r) describes the new block. K:(n-r,r) has rows for the variables
        before and after it, in logical order. start may be traced and must
        be in [0,n-r]. C must be symmetric and the target positive definite.
        Floating inputs share the factor dtype. Use (n-1,1) and (1,1) arrays
        for a single replacement.
        Logical order and shape are preserved.
        """
        L = self.physical_factor
        n = L.shape[0]
        start, K, C = _replacement(start, K, C, L)
        r = C.shape[0]
        indices = start + jnp.arange(r)
        physical = jnp.sort(self.inverse_order[indices])
        survivors = self.order[_keep_many(n, physical)]
        # Locate the physical survivors in K's compact logical row order.
        compact = survivors - r * (survivors >= start)
        result = chol_delete_append(L, physical, K[compact], C)
        order = jnp.concatenate((survivors, indices))
        inverse = self.inverse_order - jnp.searchsorted(physical, self.inverse_order)
        inverse = inverse.at[indices].set(jnp.arange(n - r, n, dtype=inverse.dtype))
        return type(self)(result, order, inverse)

    def delete_append(self, indices: Array, K: Array, C: Array) -> "PermutedCholesky":
        """Delete logical indices:(r,) and append a block of s variables.

        K:(n-r,s) follows logical survivor order. C:(s,s) must be symmetric
        and the target positive definite. Indices must be distinct and in bounds.
        The result has size n-r+s, in survivor order followed by the new block.
        """
        return self.delete(indices).append(K, C)

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

    def whiten(self, rhs: Array) -> Array:
        """Return W:(n,r) solving physical_factor @ W = rhs[order].

        rhs:(n,r) uses logical order and shares the factor dtype. Use (n,1)
        for one column. One triangular solve returns whitened coordinates,
        without an output permutation. Columns retain their input order.
        W.T @ W equals
        rhs.T @ A^{-1} @ rhs for the represented matrix A in logical order.
        Cross products require results from the same factor state.
        Supports jit, vmap and first-order differentiation.
        """
        rhs = _matrix_rhs(self.physical_factor, rhs)
        return _whiten(self.physical_factor, rhs[self.order])

    def logdet(self) -> Array:
        """Log determinant of the represented covariance."""
        return _logdet(self.physical_factor)

    def quadratic(self, rhs: Array) -> Array:
        """Covariance inverse quadratic form, separately per matrix RHS column."""
        rhs = _rhs(self.physical_factor, rhs)
        return _quadratic(self.physical_factor, rhs[self.order])
