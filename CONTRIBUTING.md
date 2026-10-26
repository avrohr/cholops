# Contributing

## Setup and checks

Use Python 3.12 or newer and the locked environment:

```sh
uv sync --locked --all-groups
uv lock --check
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 uv run --locked python -m pytest
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --locked python -m pytest
```

Run each precision in a separate process, selecting x64 before collection.
Examples inherit that setting:

```sh
JAX_ENABLE_X64=1 uv run --locked python examples/basic_factor.py
JAX_ENABLE_X64=1 uv run --locked python examples/streaming_gp.py
JAX_ENABLE_X64=1 uv run --locked python examples/online_least_squares.py
JAX_ENABLE_X64=1 uv run --locked python examples/tensor_operations.py
```

The default suite is the small CI layer. Keep each case within a few seconds,
including compilation. Small numerical and gradient regressions belong here.
The manual release suite adds larger matrices, long sequences, high-precision
references and benchmark comparisons:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 uv run --locked python -m pytest --full
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --locked python -m pytest --full
JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 uv run --locked python -m pytest --full --require-gpu
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 uv run --locked python -m pytest --full --require-gpu
```

CUDA requires the configured GPU environment (`uv sync --locked --all-groups
--extra cuda13` on supported Linux hosts). CPU full runs explicitly skip the
GPU-only storage check. `--require-gpu` prevents accidental CPU validation.
Use `--full -m full` to run only additional release tests. See
[testing guide](TESTING.md) and the [benchmark guide](benchmarks/README.md).

## Design

The public API is exported from `cholops` and grouped into `cholops.factors`
and `cholops.tensors`. Leading-underscore modules are private.

`factors/` contains the two factor classes and their shared matrix calculations.
The classes own factor storage and, for `PermutedCholesky`, permutation state.
`tensors/` contains factorization, append, rank-one updates, deletion and
replacement. Each algorithm stays with its derivative rules. Private modules
provide shared validation, indexing and JAX differentiation support.

Dependencies run from factor objects to tensor operations. Tensor code does not
import the factor classes. Reuse the tensor implementations rather than
duplicating them in containers.

State is immutable. No hidden precision changes, jitter, clipping, retry,
occupancy tracking, memory pool or buffer donation. Test meaningful numerical
behavior against independent dense or high-precision references. A small factor
residual alone does not establish accurate gradients. Preserve signed
zero-weight derivatives and visible numerical failures.

Keep supported transformations and shape/order explicit in docstrings
and the [API guides](docs/tensors.md). Any new numerical algorithm needs both
correctness evidence and representative measurements before performance claims.
