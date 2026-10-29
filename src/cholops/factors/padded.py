"""Fixed-capacity factors with explicit activation of independent identity slots."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from ..tensors import chol_delete_append
from ..tensors._indexing import _indices, _keep_many
from ..tensors._validation import _blocks, _vector
from ..tensors.append import _append_block
from ._append import _append_solution
from ._consumers import _matrix_rhs, _rhs
from .permuted import PermutedCholesky


def _activity(active, n):
    active = jnp.asarray(active)
    if active.shape != (n,) or active.dtype != jnp.bool_:
        raise TypeError("active must be a boolean array of shape (N,)")
    return active


def _mask_rows(array, active):
    mask = active.reshape((active.size,) + (1,) * (array.ndim - 1))
    return jnp.where(mask, array, 0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PaddedCholesky:
    """A PermutedCholesky and a logical activity mask, with fixed capacity N.

    Inactive slots must be independent unit entries. Constructors accept an
    already padded matrix or factor and do not check this numerical property.
    Logical slot identities stay fixed as physical ordering changes. All state
    is immutable and consists of dynamic PyTree leaves. Supports jit, vmap,
    fixed-shape scans and first-order forward and reverse differentiation.
    """

    factor: PermutedCholesky
    active: Array

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
        rhs = _rhs(self.factor.physical_factor, rhs)
        result = self.factor.solve(_mask_rows(rhs, self.active))
        return _mask_rows(result, self.active)

    def whiten(self, rhs: Array) -> Array:
        """Whiten rhs:(N,q), ignoring inactive logical input rows.

        Returns physical factor coordinates, with zero inactive physical rows.
        Use (N,1) for one column. Ignored input entries may be non-finite.
        """
        rhs = _matrix_rhs(self.factor.physical_factor, rhs)
        result = self.factor.whiten(_mask_rows(rhs, self.active))
        return _mask_rows(result, self.active[self.factor.order])

    def quadratic(self, rhs: Array) -> Array:
        """Active inverse quadratic form for rhs:(N,) or per column of (N,q).

        Inactive logical input entries are ignored, including non-finite ones.
        """
        rhs = _rhs(self.factor.physical_factor, rhs)
        return self.factor.quadratic(_mask_rows(rhs, self.active))

    def logdet(self) -> Array:
        """Log determinant of the active system. Unit padding contributes zero."""
        return self.factor.logdet()

    def _modify(self, indices, K, C, *, independent):
        L = self.factor.physical_factor
        n = L.shape[0]
        indices = jnp.asarray(indices)
        _indices(indices, n)
        K, C = _blocks(K, C, L)
        if C.shape[0] != indices.size:
            raise ValueError("C must have shape (r,r), matching indices:(r,)")
        if indices.size == 0:
            return self
        physical = jnp.sort(self.factor.inverse_order[indices])
        keep = _keep_many(n, physical)
        survivors = self.factor.order[keep]
        K = _mask_rows(K[survivors], self.active[survivors])
        if independent:
            # Independent removed columns are zero in every surviving row.
            # Separate row/column gathers preserve the surviving lower factor.
            retained = jnp.take(jnp.take(L, keep, axis=0), keep, axis=1)
            result = _append_block(retained, K, C)
        else:
            result = chol_delete_append(L, physical, K, C)
        order = jnp.concatenate((survivors, indices))
        factor = PermutedCholesky(result, order, jnp.argsort(order))
        return type(self)(factor, self.active.at[indices].set(True))

    def activate(self, indices: Array, K: Array, C: Array) -> "PaddedCholesky":
        """Activate inactive logical indices:(r,), preserving capacity and slots.

        K:(N,r) uses full logical row order. Its columns and both axes of C:(r,r)
        follow the supplied indices order. All currently inactive K rows are
        ignored. C supplies the selected covariance block and must be symmetric.
        The caller must select distinct inactive slots and a positive-definite
        target. Occupancy and independence are not checked. No deletion updates
        are needed. All floating inputs share the factor's float32/64 dtype.
        """
        return self._modify(indices, K, C, independent=True)

    def activate_solve(
        self, indices: Array, K: Array, C: Array, *, solution: Array, rhs: Array
    ) -> tuple["PaddedCholesky", Array]:
        """Activate slots and reuse the old solution and block factorization.

        Activation inputs follow activate. solution:(N,) solves the old active
        system. rhs:(N,) is the complete new RHS, unchanged on old active slots.
        Old inactive solution entries and still-inactive RHS entries are ignored,
        including non-finite values. Returns zero inactive coefficients.
        """
        L = self.factor.physical_factor
        n = L.shape[0]
        solution = _vector(solution, n, L, "solution")
        rhs = _vector(rhs, n, L, "rhs")
        updated = self.activate(indices, K, C)
        m = n - jnp.asarray(indices).size
        order = updated.factor.order
        survivors = order[:m]
        K = _mask_rows(jnp.asarray(K)[survivors], self.active[survivors])
        solution = _mask_rows(solution, self.active)
        rhs = _mask_rows(rhs, updated.active)
        coefficients = _append_solution(
            updated.factor.physical_factor, K, solution[survivors], rhs[order[m:]]
        )
        coefficients = coefficients[updated.factor.inverse_order]
        return updated, _mask_rows(coefficients, updated.active)

    def replace(self, indices: Array, K: Array, C: Array) -> "PaddedCholesky":
        """Replace arbitrary logical indices:(r,) and mark them active.

        K:(N,r) uses full logical row order. Its columns and both axes of C:(r,r)
        follow indices order. Selected and retained inactive K rows are ignored.
        C supplies the entire selected block. Selected slots may be occupied,
        inactive or mixed. Uses general deletion and block append. The target
        must be positive definite and C symmetric, with the factor's dtype.
        """
        return self._modify(indices, K, C, independent=False)

    def deactivate(self, indices: Array) -> "PaddedCholesky":
        """Restore logical indices:(r,) to unit padding, preserving other slots.

        Uses general deletion and appends an independent identity block.
        Capacity and the covariance among remaining observations are unchanged.
        """
        indices = jnp.asarray(indices)
        L = self.factor.physical_factor
        r = indices.size
        updated = self.replace(
            indices, jnp.zeros((L.shape[0], r), L.dtype), jnp.eye(r, dtype=L.dtype)
        )
        return type(self)(updated.factor, self.active.at[indices].set(False))
