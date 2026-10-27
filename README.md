<p align="center">
  <img src="https://raw.githubusercontent.com/avrohr/cholops/main/docs/assets/cholops-logo.svg" alt="cholops logo: an eye with three diagonal squares" width="240">
</p>

# cholops

Cholesky modifications in JAX, with differentiable tensor operations and
immutable factor objects. cholops provides fast appends, deletions and updates
of a dense positive-definite matrix's factor, directly or through a factor
object.

## Install

Install with `pip install cholops`. On Linux with a compatible CUDA 13 driver,
use `pip install "cholops[cuda13]"`. The initial release targets **JAX 0.11.2** and
Python 3.12 or newer.

We recommend using float64; see [JAX's precision settings](https://docs.jax.dev/en/latest/101/default_dtypes.html).

Set the `JAX_ENABLE_X64` shell environment variable before starting Python:

```bash
export JAX_ENABLE_X64=1
```

Alternatively, enable it in Python before creating arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## Example

```python
import jax
import jax.numpy as jnp
from cholops import Cholesky, PermutedCholesky

A = jnp.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.2]])
b = jnp.array([1.0, -0.5, 0.2])

# Create a Cholesky factor from a positive-definite matrix.
factor = Cholesky.from_matrix(A)
factor = factor.append(jnp.array([[0.2], [0.1], [0.05]]), jnp.array([[1.3]]))
L = factor.factor
x = factor.solve(
    jnp.append(b, 0.4)
)  # One RHS entry per variable, including the new one.

# PermutedCholesky replaces by deletion and appending,
# while managing logical ordering transparently.
managed = PermutedCholesky.from_matrix(A)
updated = managed.replace(1, jnp.array([0.2, 0.1]), 1.3)
x_new = updated.solve(b)  # Inputs and outputs remain in logical order.


# Zero-mean Gaussian log likelihood and its covariance/data gradients.
@jax.jit
def log_likelihood(A, b):
    state = Cholesky.from_matrix(A)
    return -0.5 * (state.logdet() + state.quadratic(b) + b.size * jnp.log(2 * jnp.pi))


logp, (grad_A, grad_b) = jax.value_and_grad(log_likelihood, argnums=(0, 1))(A, b)
```

See the [factor guide](https://github.com/avrohr/cholops/blob/main/docs/factors.md) for more information. The executable
examples explain each step and check their results against dense calculations:

- [Basic factor operations](https://github.com/avrohr/cholops/blob/main/examples/basic_factor.py): add, replace and remove
  observations, then solve a system and calculate a Gaussian log likelihood.
- [Streaming Gaussian process](https://github.com/avrohr/cholops/blob/main/examples/streaming_gp.py): maintain a fixed-size
  observation window with `PermutedCholesky` and differentiate through the
  replacement sequence with respect to a kernel parameter.
- [Online least squares](https://github.com/avrohr/cholops/blob/main/examples/online_least_squares.py): update a regression
  fit with `rank1_positive` as measurements arrive.
- [Direct tensor operations](https://github.com/avrohr/cholops/blob/main/examples/tensor_operations.py): compare replacement
  orderings and arrange the right-hand side for each returned factor.

## Tensor operations

The factor objects use the following tensor operations, which can also be
imported directly from `cholops`:

| Operation | Description |
| --- | --- |
| `cholesky` | Factor a positive-definite matrix. |
| `chol_append` | Append a block of covariance rows and columns. |
| `chol_delete` | Delete one covariance row and column, preserving survivor order. |
| `chol_replace` | Replace one covariance row and column at the same index. |
| `chol_delete_append` | Delete one variable and append its replacement last. |
| `chol_replace_last` | Replace a trailing block of variables. |
| `chol_diag_update` | Add a signed value to one covariance diagonal entry. |
| `chol_rank1` | Apply a rank-one update with a signed weight. |
| `chol_rank1_positive` | Apply a positive rank-one update using the optimized path. |

See the [tensor guide](https://github.com/avrohr/cholops/blob/main/docs/tensors.md) for shapes, ordering and preconditions.
All operations support `jit`, `vmap` and first-order forward and reverse
differentiation. Higher-order derivatives have not been validated for general use.

## Performance

We benchmark a single index-preserving `chol_replace` call against a naive JAX
implementation and full matrix refactorization on an NVIDIA H100. All methods
receive the existing factor and replacement covariances.

![Single index-preserving replacement: absolute milliseconds on H100](https://raw.githubusercontent.com/avrohr/cholops/main/benchmarks/figures/replacement.svg)

Replacing the first index of a **16,384 × 16,384** matrix gives the following
speedups after compilation:

| Precision | Forward: vs recompute / naive replacement | Value + gradient: vs recompute / naive replacement |
| --- | ---: | ---: |
| float32 | 15.7× / 21.3× | 22.7× / 30.3× |
| float64 | 10.2× / 16.1× | 15.6× / 22.9× |

[Recomputation](https://github.com/avrohr/cholops/blob/main/benchmarks/_recompute.py) forms `L @ L.T`, changes its row/column
and calls native JAX Cholesky. The [naive replacement](https://github.com/avrohr/cholops/blob/main/benchmarks/_naive.py)
baseline uses two signed rank-one scans with standard JAX autodiff.

See [the complete benchmark results](https://github.com/avrohr/cholops/blob/main/benchmarks/README.md) for
the measurement protocol, additional timings and memory usage.

## Numerical limitations

Near-singular covariances can produce inaccurate factors and gradients.
See [numerical limits](https://github.com/avrohr/cholops/blob/main/docs/numerics.md).
