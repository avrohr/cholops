"""Differentiable Cholesky operations on dense lower-triangular arrays.

Inputs use a common float32 or float64 dtype. Supply a valid lower factor and
choose modifications that keep the represented matrix positive definite.
Operations support JIT and vmap. Differentiation is supported only through
first order, in both forward and reverse mode.
"""

from .append import chol_append
from .deletion import chol_delete, chol_delete_append
from .factorization import cholesky
from .rank_one import chol_diag_update, chol_rank1, chol_rank1_positive
from .replacement import chol_replace, chol_replace_last

__all__ = [
    "cholesky",
    "chol_append",
    "chol_delete",
    "chol_diag_update",
    "chol_rank1",
    "chol_rank1_positive",
    "chol_delete_append",
    "chol_replace",
    "chol_replace_last",
]
