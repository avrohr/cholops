# Testing

The tests check factors, solves and derivatives against dense recomputation and
independent high-precision references. This guide describes the test groups and
how to run them. See [contributing](CONTRIBUTING.md) for environment setup.

## Running tests

Run float32 and float64 in separate processes:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 uv run --locked python -m pytest
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --locked python -m pytest
```

The default suite runs in CI on Python 3.12 and 3.13. Additional tests are marked
`full` and run before releases:

| Option | What runs |
| --- | --- |
| No option | Small functional, numerical and derivative tests. |
| `--full` | Default tests plus larger matrices, longer sequences and benchmark checks. |
| `--full -m full` | Only the additional tests. |
| `--full --require-gpu` | Full suite, failing immediately if no GPU is active. |

For CUDA, install the `cuda13` extra and set `JAX_PLATFORMS=cuda`. Run both
precisions as on CPU. GPU storage checks are skipped on CPU; inspect the test
summary for failures and skips.

## What is checked

- **Operations and factor objects:** reconstruction of the expected covariance,
  solves, log determinants, quadratic forms and logical ordering after updates.
- **JAX transformations:** JIT, batching, runtime indices, first-order JVPs and
  VJPs, and combinations such as differentiation through replacement scans.
- **Derivative accuracy:** comparison with dense derivatives, JVP/VJP duality,
  and independent Decimal calculations in [tests/reference](tests/reference).
- **Sequences and larger matrices:** accumulated factor and gradient errors over
  repeated replacements, plus scan lengths absent from the small CI cases.
- **GPU storage:** compiler-reported temporary memory for deletion gradients at
  several matrix sizes.
- **Benchmark comparisons:** matching inputs, outputs and derivatives across
  implementations, including the naive replacement baseline.

Tolerances are chosen per test and precision. Higher-order derivatives are not
part of the tested API. See [numerical limits](docs/numerics.md) for limitations
that passing these tests does not remove.

## Writing tests

Each test should establish a desired behavior and have a name that explains
what a failure means. Compare numerical results with an independent calculation.
For derivatives, choose an objective that depends on the inputs being tested.

Keep setup and assertions direct. Split unrelated checks into separate tests,
and use parameter names that identify the failing case. Check logical results
and documented behavior without requiring a particular internal arrangement.
Keep numerical regression cases and their precision-specific tolerances when
consolidating tests.

