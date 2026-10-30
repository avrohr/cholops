"""Fixed-capacity factors with explicit activation of independent identity slots."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from ..tensors import chol_delete_append, cholesky
from ..tensors._indexing import _indices, _keep_many
from ..tensors._validation import _blocks, _vector
from ._consumers import _matrix_rhs, _rhs, _solve
from .permuted import PermutedCholesky


def _activity(active, n):
    active = jnp.asarray(active)
    if active.shape != (n,) or active.dtype != jnp.bool_:
        raise TypeError("active must be a boolean array of shape (N,)")
    return active


def _mask_rows(array, active):
    mask = active.reshape((active.size,) + (1,) * (array.ndim - 1))
    return jnp.where(mask, array, 0)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class PaddedCholesky:
    """A lower factor and logical activity mask, with fixed capacity N.

    Inactive slots must be independent unit entries. Constructors accept an
    already padded matrix or factor and do not check this numerical property.
    Logical slot identities stay fixed as physical ordering changes. All state
    is immutable and consists of dynamic PyTree leaves. Supports jit, vmap,
    fixed-shape scans and first-order forward and reverse differentiation.
    """

    _permuted: PermutedCholesky
    active: Array

    def __init__(self, factor: PermutedCholesky, active: Array):
        object.__setattr__(self, "_permuted", factor)
        object.__setattr__(self, "active", active)

    @property
    def factor(self) -> Array:
        """Lower-triangular factor in physical order."""
        return self._permuted.physical_factor

    @property
    def order(self) -> Array:
        """Map physical positions to logical slots."""
        return self._permuted.order

    @property
    def inverse_order(self) -> Array:
        """Map logical slots to physical positions."""
        return self._permuted.inverse_order

    def tree_flatten(self):
        return (self._permuted, self.active), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @classmethod
    def from_matrix(cls, A: Array, *, active: Array) -> "PaddedCholesky":
        """Factor an already padded (N,N) matrix with logical activity bool[N].

        Inactive rows and columns must have unit diagonal and zero off-diagonal
        entries. Uses the symmetry-preserving Cholesky reverse rule.
        """
        factor = PermutedCholesky.from_matrix(A)
        return cls(factor, _activity(active, factor.physical_factor.shape[0]))

    @classmethod
    def from_factor(cls, L: Array, *, active: Array) -> "PaddedCholesky":
        """Wrap a padded lower factor, initially in logical order.

        Inactive slots must be independent unit entries. The unused upper
        triangle is ignored. This does not repeat factorization.
        """
        factor = PermutedCholesky.from_factor(L)
        return cls(factor, _activity(active, factor.physical_factor.shape[0]))

    def solve(self, rhs: Array) -> Array:
        """Solve the active system for rhs:(N,) or (N,q), ignoring inactive rows.

        The result has the same shape and exactly zero inactive logical rows.
        Ignored RHS entries may be non-finite. Inputs share the factor dtype.
        """
        rhs = _rhs(self.factor, rhs)
        result = self._permuted.solve(_mask_rows(rhs, self.active))
        return _mask_rows(result, self.active)

    def whiten(self, rhs: Array) -> Array:
        """Whiten rhs:(N,q), ignoring inactive logical input rows.

        Returns physical factor coordinates, with zero inactive physical rows.
        Use (N,1) for one column. Ignored input entries may be non-finite.
        """
        rhs = _matrix_rhs(self.factor, rhs)
        result = self._permuted.whiten(_mask_rows(rhs, self.active))
        return _mask_rows(result, self.active[self.order])

    def quadratic(self, rhs: Array) -> Array:
        """Active inverse quadratic form for rhs:(N,) or per column of (N,q).

        Inactive logical input entries are ignored, including non-finite ones.
        """
        rhs = _rhs(self.factor, rhs)
        return self._permuted.quadratic(_mask_rows(rhs, self.active))

    def logdet(self) -> Array:
        """Log determinant of the active system. Unit padding contributes zero."""
        return self._permuted.logdet()

    def _activate(self, indices, K, C):
        L = self.factor
        n = L.shape[0]
        indices = jnp.asarray(indices)
        _indices(indices, n)
        K, C = _blocks(K, C, L)
        r = indices.size
        if C.shape[0] != r:
            raise ValueError("C must have shape (r,r), matching indices:(r,)")
        if r == 0:
            return self, K, C

        # Solve before moving the factor. Inactive rows of the projection are
        # zero, so the same projection also updates the old solution.
        projected = self._permuted.whiten(_mask_rows(K, self.active))
        trailing = cholesky(C - projected.T @ projected)
        physical = self.inverse_order[indices]
        selected = jnp.zeros(n, dtype=bool).at[physical].set(True)
        positions = jnp.arange(n)
        destinations = jnp.where(selected, n, positions - jnp.cumsum(selected))
        permutation = (
            jnp.zeros_like(positions).at[destinations].set(positions, mode="drop")
        )
        permutation = permutation.at[-r:].set(physical)

        # Move only the selected independent entries, keeping all survivors in
        # order. Separate gathers avoid a joint two-axis gather on CPU.
        upper = self._permuted._upper[permutation][:, permutation]
        upper = jax.lax.dynamic_update_slice_in_dim(
            upper, projected[permutation], n - r, axis=1
        )
        upper = jax.lax.dynamic_update_slice(upper, trailing.T, (n - r, n - r))
        order = self.order[permutation]
        inverse_order = jnp.zeros_like(order).at[order].set(positions)
        updated = type(self)(
            PermutedCholesky._from_upper(upper, order, inverse_order),
            self.active.at[indices].set(True),
        )
        return updated, projected, trailing

    def activate(self, indices: Array, K: Array, C: Array) -> "PaddedCholesky":
        """Activate inactive logical indices:(r,), preserving capacity and slots.

        K:(N,r) uses full logical row order. Its columns and both axes of C:(r,r)
        follow the supplied indices order. All currently inactive K rows are
        ignored. C supplies the selected covariance block and must be symmetric.
        The caller must select distinct inactive slots and a positive-definite
        target. Occupancy and independence are not checked. No deletion updates
        are needed. All floating inputs share the factor's float32/64 dtype.
        """
        return self._activate(indices, K, C)[0]

    def activate_solve(
        self, indices: Array, K: Array, C: Array, *, solution: Array, rhs: Array
    ) -> tuple["PaddedCholesky", Array]:
        """Activate slots and reuse the old solution and block factorization.

        Activation inputs follow activate. solution:(N,) solves the old active
        system. rhs:(N,) is the complete new RHS, unchanged on old active slots.
        Old inactive solution entries and still-inactive RHS entries are ignored,
        including non-finite values. Returns zero inactive coefficients.
        """
        L = self.factor
        n = L.shape[0]
        solution = _vector(solution, n, L, "solution")
        rhs = _vector(rhs, n, L, "rhs")
        updated, projected, trailing = self._activate(indices, K, C)
        indices = jnp.asarray(indices)
        if indices.size == 0:
            return updated, _mask_rows(solution, self.active)
        K = _mask_rows(jnp.asarray(K), self.active)
        solution = _mask_rows(solution, self.active)
        rhs = _mask_rows(rhs, updated.active)
        beta = _solve(trailing, rhs[indices] - K.T @ solution)
        correction = solve_triangular(L, projected @ beta, lower=True, trans="T")
        coefficients = solution.at[self.order].add(-correction)
        coefficients = coefficients.at[indices].set(beta)
        return updated, _mask_rows(coefficients, updated.active)

    def replace(self, indices: Array, K: Array, C: Array) -> "PaddedCholesky":
        """Replace arbitrary logical indices:(r,) and mark them active.

        K:(N,r) uses full logical row order. Its columns and both axes of C:(r,r)
        follow indices order. Selected and retained inactive K rows are ignored.
        C supplies the entire selected block. Selected slots may be occupied,
        inactive or mixed. Uses general deletion and block append. The target
        must be positive definite and C symmetric, with the factor's dtype.
        """
        L = self.factor
        n = L.shape[0]
        indices = jnp.asarray(indices)
        _indices(indices, n)
        K, C = _blocks(K, C, L)
        if C.shape[0] != indices.size:
            raise ValueError("C must have shape (r,r), matching indices:(r,)")
        if indices.size == 0:
            return self
        physical = jnp.sort(self.inverse_order[indices])
        survivors = self.order[_keep_many(n, physical)]
        K = _mask_rows(K[survivors], self.active[survivors])
        result = chol_delete_append(L, physical, K, C)
        order = jnp.concatenate((survivors, indices))
        inverse_order = jnp.zeros_like(order).at[order].set(jnp.arange(n))
        return type(self)(
            PermutedCholesky(result, order, inverse_order),
            self.active.at[indices].set(True),
        )

    def deactivate(self, indices: Array) -> "PaddedCholesky":
        """Restore logical indices:(r,) to unit padding, preserving other slots.

        Uses general deletion and appends an independent identity block.
        Capacity and the covariance among remaining observations are unchanged.
        """
        indices = jnp.asarray(indices)
        L = self.factor
        r = indices.size
        updated = self.replace(
            indices, jnp.zeros((L.shape[0], r), L.dtype), jnp.eye(r, dtype=L.dtype)
        )
        return type(self)(updated._permuted, self.active.at[indices].set(False))
