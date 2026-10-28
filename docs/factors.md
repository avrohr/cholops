# Cholesky factor objects

Import the factor classes from `cholops` or `cholops.factors`:

```python
from cholops.factors import Cholesky
```

## Cholesky

`Cholesky` stores a matrix's lower-triangular Cholesky factor and provides methods
for modifying it and solving systems with it. The factor follows the same row
and column order as the matrix and is available as `.factor`.

### Construct and use a factor

```python
import jax
import jax.numpy as jnp
from cholops import Cholesky

K = jnp.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
y = jnp.array([1.0, -0.5, 0.2])
factor = Cholesky.from_matrix(K)

# Replace row and column 1. k contains the new entries for indices [0, 2].
updated = factor.replace(1, jnp.array([0.2, 0.1]), 1.2)
L = updated.factor
alpha = updated.solve(y)  # Solve the modified matrix system.
logdet = updated.logdet()
quadratic = updated.quadratic(y)
```

`from_matrix(K)` factors the symmetric part of a real square matrix. It uses
`cholops.cholesky`, including its symmetry-preserving reverse rule. See the
[numerical guide](numerics.md#factorization) for the rule and its limits.

If you already have a lower factor `L`, use `Cholesky.from_factor(L)`. This wraps
`L` without repeating the factorization and ignores its unused upper triangle.
It does not change the backward rule of the function that originally produced
`L`.

### Matrix calculations

The methods below operate on the represented matrix `A = L @ L.T`.

| Method | Inputs | Result |
| --- | --- | --- |
| `solve(rhs)` | Vector `(n,)` or matrix `(n,r)` | Solution of `A @ x = rhs` |
| `logdet()` | None | Log determinant of `A` |
| `quadratic(rhs)` | Vector `(n,)` or matrix `(n,r)` | `rhs.T @ A^{-1} @ rhs` for a vector, or one quadratic form per column |

Inputs share a float32 or float64 dtype. The matrix must be positive definite.
See the [tensor input requirements and failure behavior](tensors.md#preconditions-and-failures)
for what the caller needs to check.

### Modifications

Each method returns a new `Cholesky` object and leaves the original unchanged.

| Method | Effect and inputs |
| --- | --- |
| `append(K, C)` | Add `r` rows and columns, with `K:(n,r)` and symmetric `C:(r,r)` |
| `delete(indices)` | Remove `indices:(r,)`, preserving the order of the remaining entries |
| `delete_last(r)` | Remove the final `r` rows and columns by slicing; `r` is static |
| `replace(index, k, kappa)` | Replace a row and column at the same index, with `k:(n-1,)` and scalar diagonal `kappa` |
| `delete_append(indices, K, C)` | Delete `indices:(r,)` and append `s` variables, with `K:(n-r,s)` and symmetric `C:(s,s)` |
| `replace_last(K, C)` | Replace the final `r` rows and columns, with `K:(n-r,r)` and `C:(r,r)` |
| `diag_update(index, delta)` | Add signed `delta` to one matrix diagonal entry |
| `rank1(u, alpha=1.0)` | Add `alpha * outer(u,u)`, with `u:(n,)` and signed weight `alpha` |
| `rank1_positive(u)` | Add `outer(u,u)` using the optimized positive-update algorithm |

`append` always takes block-shaped arrays, including `(n,1)` and `(1,1)`
for one variable. The rows of `K` follow the existing variable order.

`delete` and `delete_append` take an integer array of distinct indices, including
`jnp.array([i])` for one variable. Indices refer to the input before deletion.
For `replace`, supply `k` in the order of the entries you are keeping.
For `delete_append`, the rows of `K` follow that order. Replacing `b` with `e`
in `[a, b, c, d]` therefore uses entries in the order `[a, c, d]`.
The result has order `[a, e, c, d]` for `replace`, or
`[a, c, d, e]` for `delete_append`. Subsequent inputs must follow that order.
Deletion renumbers the remaining indices from zero.

`replace_last` wraps `delete_append` with equal-sized trailing blocks. It keeps
the leading factor and recomputes only the trailing block.
This makes small trailing-block replacements efficient and avoids the downdate
used by general index-preserving replacement.

`rank1` accepts positive, negative and zero weights. The derivative with respect
to the weight is generally nonzero even at zero. Use `rank1_positive` when you
only need to add `outer(u,u)`.

```python
expanded = factor.append(jnp.array([[0.1], [0.2], [0.1]]), jnp.array([[1.0]]))
reduced = expanded.delete(jnp.array([1]))  # Order is now [0, 2, appended].
updated = factor.rank1_positive(jnp.array([0.1, -0.2, 0.1]))
downdated = updated.rank1(jnp.array([0.1, -0.2, 0.1]), alpha=-0.5)
shifted = downdated.diag_update(0, 0.1)
```

These methods use the [tensor operations](tensors.md) and their derivative
rules. The tensor functions are also available directly.

### JIT, scans and differentiation

A `Cholesky` object is an immutable JAX PyTree, so it can be passed into a
compiled function and returned from it:

```python
@jax.jit
def replace_and_score(factor, index, k, kappa, y):
    factor = factor.replace(index, k, kappa)
    score = 0.5 * (factor.logdet() + factor.quadratic(y))
    return factor, score
```

Use `vmap` to batch operations. Fixed-size modifications can be carried through
`lax.scan`. Append and delete change the factor's shape, so they cannot change
the size of a factor carried through a scan.

Construction, modifications and matrix calculations support first-order
forward and reverse differentiation. For example:

```python
A = jnp.array([[2.0, 0.3], [0.3, 1.5]])
dA = jnp.array([[0.1, -0.02], [-0.02, 0.05]])


def logdet(A):
    return Cholesky.from_matrix(A).logdet()


value, directional_derivative = jax.jvp(logdet, (A,), (dA,))
value, covariance_gradient = jax.value_and_grad(logdet)(A)
# directional_derivative equals jnp.sum(covariance_gradient * dA).
```

See the [basic factor example](../examples/basic_factor.py) for modifications,
solves and Gaussian log likelihoods. The
[online least-squares example](../examples/online_least_squares.py) uses positive
rank-one updates to incorporate new measurements into a regression fit.
Higher-order derivatives have not been validated for general use.

## PermutedCholesky

When replacements are frequent, preserving the factor's original ordering can
add work. `PermutedCholesky` instead deletes the old variable and appends its
replacement at the end of the stored factor. This avoids the trailing downdate
used by index-preserving replacement and can make repeated replacements faster
and more numerically stable.

You still work with the original ordering. For example, replacing `b` with `e`
in `[a, b, c, d]` leaves the stored factor ordered as `[a, c, d, e]`, but the
object presents `[a, e, c, d]` to the caller. It translates indices and rearranges
vectors internally, so `replace`, `solve` and the other methods use the same
input and output ordering as `Cholesky`.

```python
from cholops import PermutedCholesky

managed = PermutedCholesky.from_matrix(K)
managed = managed.replace(1, jnp.array([0.2, 0.1]), 1.2)
alpha = managed.solve(y)  # y and alpha still follow logical order [0, 1, 2].

expanded = managed.append(jnp.array([[0.1], [0.2], [0.1]]), jnp.array([[1.0]]))
updated = managed.rank1_positive(jnp.array([0.1, -0.2, 0.1]))
```

Use this class when you want the replacement algorithm to manage ordering for
you. Use `Cholesky` when you need the lower-triangular factor in the original
matrix order.

### Available methods

`PermutedCholesky` supports all the `Cholesky` methods described above except
`replace_last` and `delete_last`. After replacements, the last logical variables
can be scattered throughout the stored factor, so they do not form a physical
trailing block.

For all supported methods, supply indices and arrays in logical order.
`replace` preserves logical positions. `delete_append` deliberately moves the
replacement to the end of logical order, just as it does on `Cholesky`.
Append adds new logical indices at the end, and delete renumbers the surviving
logical indices from zero. The object updates its internal maps for each operation.

Both `from_matrix` and `from_factor` start with identical logical and physical
ordering. Methods return new objects. JIT, batching, fixed-size scans and
first-order differentiation work as described for `Cholesky`.

See the [streaming Gaussian-process example](../examples/streaming_gp.py) for
a circular observation buffer, predictions and differentiation through the
replacement sequence with respect to a kernel parameter.

### Accessing the stored factor

`physical_factor` is lower triangular in storage order. The two index maps are
`order[physical] = logical` and `inverse_order[logical] = physical`. They are
available when integrating with tensor-level code. Prefer the factory methods
for construction, which initialize consistent maps.

Reordering the rows of this factor into logical order gives a square root of
the represented matrix, but generally not its lower-triangular Cholesky factor.
There is therefore no `.factor` property equivalent to `Cholesky.factor`.

The factor and index maps are dynamic PyTree leaves, so a change in the
permutation alone does not trigger recompilation. The maps are integer indices
and are not differentiable. Reordering a vector takes linear work, which is
included in the [managed-factor benchmarks](../benchmarks/README.md#full-benchmarks).
