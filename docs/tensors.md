# Tensor operations

Tensor operations can be imported from `cholops` or `cholops.tensors`:

```python
from cholops.tensors import cholesky, chol_append, chol_replace
```

All factors use conventional dense lower-triangular
storage, and their unused upper triangle is ignored.

Let `A = L @ L.T`, where `L` has shape `(n,n)` and a positive diagonal.
Continuous arrays share a real float32 or float64 dtype. Python scalar
weights and diagonals follow the factor dtype; typed arrays are not silently
promoted. Dimensions are static. Indices and scalar weights may be traced.
Use `jax.vmap` for batches of modifications.

## Operations

| Function | Inputs and result | Differentiation |
| --- | --- | --- |
| `cholesky(A)` | Lower factor of the symmetric part of `A`; accepts `(...,n,n)` | JVP and reverse |
| `chol_append(L, k, kappa)` | Factor of `[[A,k],[k.T,kappa]]`; `k:(n,)`, scalar `kappa`; output `(n+1,n+1)` | JVP and reverse |
| `chol_append_block(L, K, C)` | Factor of `[[A,K],[K.T,C]]`; `K:(n,r)`, symmetric `C:(r,r)`; output `(n+r,n+r)` | JVP and reverse |
| `chol_delete(L, index)` | Remove covariance row/column `index`; survivor order preserved; output `(n-1,n-1)` | JVP and reverse |
| `chol_replace(L, index, k, kappa)` | Replace row/column at the same index; `k:(n-1,)` in survivor order | JVP and reverse |
| `chol_delete_append(L, index, k, kappa)` | Delete `index`, append its replacement last; same inputs as `chol_replace` | JVP and reverse |
| `chol_replace_last(L, K, C)` | Replace final `r` variables; `K:(n-r,r)`, symmetric `C:(r,r)` | JVP and reverse |
| `chol_diag_update(L, index, delta)` | Factor of `A + delta * outer(e_index,e_index)`; signed scalar `delta` | JVP and reverse, including zero |
| `chol_rank1(L, u, alpha=1.0)` | Factor of `A + alpha * outer(u,u)`; `u:(n,)`, signed scalar `alpha` | JVP and reverse, including zero |
| `chol_rank1_positive(L, u)` | Factor of `A + outer(u,u)`; `u:(n,)` | JVP and reverse |

Every operation supports `jit`, `vmap`, first-order `jvp`/`jacfwd`, and
`vjp`/`grad`/`jacrev` on its documented domain. `jax.linearize` can cache a
first-order tangent map for repeated directions. Higher-order derivatives,
including Hessians and differentiating through gradients, have not been
validated for general use. Indices and permutations have no continuous derivative.

`chol_rank1_positive` is the fast positive-only path. `chol_rank1` also accepts
negative weights; its recurrence can be slower than dense recomputation on a GPU. No input-dependent backend switch occurs. A zero
signed weight leaves the factor unchanged but generally has a nonzero
weight derivative.

## Block append

`chol_append_block` adds `r` variables after the existing `n` variables. Rows of
`K` follow the existing factor order; its columns and both axes of `C` follow
the new-variable order. It preserves the leading lower factor exactly, solves
`V = solve(L, K)`, and factors the Schur complement `C - V.T @ V`. This needs
`O(n^2*r + n*r^2 + r^3)` arithmetic, plus writing the enlarged factor.

`r=0` returns the lower input; `n=0` factors `C`. A one-column block is the
block form of `chol_append`. The block uses native JAX first-order derivatives,
including JAX's symmetric-input Cholesky convention for `C` and its tangents.
The Schur complement must be positive definite: no jitter or clipping is added,
and a non-SPD complement gives NaNs in the trailing factor. Small conditional
variances can lose relative accuracy in the subtraction.

The [direct tensor example](../examples/tensor_operations.py) appends two
variables and compares the result with a dense factorization.

## Replacement ordering

Suppose the matrix describes four variables in the order `[a, b, c, d]`, and
you want to replace `b` with a new variable `e`. Since indices start at zero,
pass `index=1`.

- `chol_replace` puts `e` in the old position of `b`. The returned factor
  represents variables in the order `[a, e, c, d]`.
- `chol_delete_append` removes `b` and puts `e` last. The returned factor
  represents variables in the order `[a, c, d, e]`.

If you then solve a system using the returned factor, the right-hand-side entries must follow that same order.

The [direct tensor example](../examples/tensor_operations.py) demonstrates both
replacement orderings, rearranges the right-hand side and compares the solutions
in a common order.

`chol_replace_last` replaces the last `r` rows and columns together. It keeps the factor of the unchanged leading block and computes only the replacement block. This makes small trailing-block replacements fast and avoids the downdates needed for general replacement, improving numerical stability. The resulting matrix must still be positive definite.

## Preconditions and failures

Each operation starts from a valid lower Cholesky factor. The modified matrix must remain positive definite so that it has a Cholesky factor with a positive diagonal. This matters especially for downdates, which subtract from the matrix, and for appends or replacements, whose new entries must be consistent with the retained entries.

The caller is responsible for choosing a valid modification. cholops does not add regularization or adjust the requested update automatically. If a modification produces NaNs or infinities, discard its result and keep the previous factor. Check the proposed entries before trying again. Finite output alone does not guarantee accuracy when the matrix is close to singular.

Incorrect shapes and incompatible dtypes raise errors. Indices must also be within the matrix bounds. A concrete Python integer is checked, but an index passed through `jit` is not checked at runtime. Validate such indices before calling the operation.

Append and delete change the factor’s size. JAX compiles separately for different shapes, so these operations cannot change the size of a factor carried through `lax.scan`. Use fixed-size replacement for such loops.

An empty factor can be extended with append, and deleting the only variable returns an empty factor. There is no valid index to delete or replace in an empty factor.
