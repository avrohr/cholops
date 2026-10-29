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
| `chol_append(L, K, C)` | Factor of `[[A,K],[K.T,C]]`; `K:(n,r)`, symmetric `C:(r,r)`; output `(n+r,n+r)` | JVP and reverse |
| `chol_delete(L, indices)` | Remove `r` variables at `indices:(r,)`; survivor order preserved; output `(n-r,n-r)` | JVP and reverse |
| `chol_delete_last(L, r)` | Remove the last `r` variables by slicing; `r` is static | JVP and reverse |
| `chol_replace(L, start, K, C)` | Replace `start:start+r` at their original positions; `K:(n-r,r)`, symmetric `C:(r,r)` | JVP and reverse |
| `chol_delete_append(L, indices, K, C)` | Delete `indices:(r,)`, append `s` variables with `K:(n-r,s)`, symmetric `C:(s,s)`; output `(n-r+s,n-r+s)` | JVP and reverse |
| `chol_replace_last(L, K, C)` | Replace final `r` variables; `K:(n-r,r)`, symmetric `C:(r,r)` | JVP and reverse |
| `chol_diag_update(L, index, delta)` | Factor of `A + delta * outer(e_index,e_index)`; signed scalar `delta` | JVP and reverse, including zero |
| `chol_rank1(L, u, alpha=1.0)` | Factor of `A + alpha * outer(u,u)`; `u:(n,)`, signed scalar `alpha` | JVP and reverse, including zero |
| `chol_rank1_positive(L, u)` | Factor of `A + outer(u,u)`; `u:(n,)` | JVP and reverse |

`chol_append` always takes block-shaped inputs. For one variable, use `K:(n,1)`
and `C:(1,1)`. An empty block (`r=0`) leaves the factor unchanged.

Deletion takes a one-dimensional integer array, including `[i]` for one variable.
Indices must be distinct and refer to the input before deletion. Their order
does not matter. An empty index array leaves the factor unchanged, and deleting
all variables returns an empty factor. For `chol_delete_append`, rows of `K`
follow survivor order and the appended block may have a different size.

Replacement takes the block's starting index, `start`, and infers its size from
`C.shape[0]`. The rows of `K` describe the variables before the block, then those
after it. Its columns and both axes of `C` follow the new block's order.
`start` may change inside `jit`, while the block size stays fixed. An empty block
leaves the factor unchanged, and replacing all variables factors `C`.

Every operation supports `jit`, `vmap`, first-order `jvp`/`jacfwd`, and
`vjp`/`grad`/`jacrev` on its documented domain. `jax.linearize` can cache a
first-order tangent map for repeated directions. Higher-order derivatives,
including Hessians and differentiating through gradients, have not been
validated for general use. Indices and permutations have no continuous derivative.

`chol_rank1_positive` is the fast positive-only path. `chol_rank1` also accepts
negative weights; its recurrence can be slower than dense recomputation on a GPU. No input-dependent backend switch occurs. A zero
signed weight leaves the factor unchanged but generally has a nonzero
weight derivative.

## Replacement ordering

Suppose the matrix describes four variables in the order `[a, b, c, d]`, and
you want to replace `b` with a new variable `e`. Since indices start at zero,
pass `start=1` to `chol_replace`, or `indices=jnp.array([1])` to
`chol_delete_append`.
Both take the new entries as a column `(3,1)` and the new diagonal as a block
`(1,1)`.

- `chol_replace` puts `e` in the old position of `b`. The returned factor
  represents variables in the order `[a, e, c, d]`.
- `chol_delete_append` removes `b` and puts `e` last. The returned factor
  represents variables in the order `[a, c, d, e]`.

If you then solve a system using the returned factor, the right-hand-side entries must follow that same order.

The [direct tensor example](../examples/tensor_operations.py) demonstrates both
replacement orderings, rearranges the right-hand side and compares the solutions
in a common order.

`chol_replace_last` combines trailing deletion by slicing with block append to replace the last `r` rows and columns, with `r` inferred from `C`. It keeps the factor of the unchanged leading block and computes only the replacement block. This makes small trailing-block replacements fast and avoids the downdates needed for general replacement, improving numerical stability. The resulting matrix must still be positive definite.

## Preconditions and failures

Each operation starts from a valid lower Cholesky factor. The modified matrix must remain positive definite so that it has a Cholesky factor with a positive diagonal. This matters especially for downdates, which subtract from the matrix, and for appends or replacements, whose new entries must be consistent with the retained entries.

The caller is responsible for choosing a valid modification. cholops does not add regularization or adjust the requested update automatically. If a modification produces NaNs or infinities, discard its result and keep the previous factor. Check the proposed entries before trying again. Finite output alone does not guarantee accuracy when the matrix is close to singular.

Incorrect shapes and incompatible dtypes raise errors. Indices must also be within the matrix bounds. A concrete Python integer is checked, but an index passed through `jit` is not checked at runtime. Validate such indices before calling the operation.

Append and delete change the factor’s size. JAX compiles separately for different shapes, so these operations cannot change the size of a factor carried through `lax.scan`. Use fixed-size replacement or delete-and-append with equal block sizes for such loops.

An empty factor can be extended with append, and deleting the only variable returns an empty factor. An empty factor has no valid indices, but accepts an empty deletion.
