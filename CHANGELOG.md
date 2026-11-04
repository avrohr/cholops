# Changelog

## 0.2.0

- Append and replace blocks of variables, delete multiple variables by index,
  or drop a trailing block with `chol_delete_last`.
- Use `PaddedCholesky` to activate and deactivate observations in a fixed-size
  factor. Streaming updates of the same shape can reuse compiled JAX code.
- `append_solve` and `activate_solve` reuse an existing solution when adding
  observations. `whiten` applies the inverse lower factor to right-hand sides.
- Apply positive rank-k updates with `chol_update(L, U)` or `factor.update(U)`.
  `chol_delete_one(L, i)` selects the positive rank-one kernel for one deletion.

**Updating from 0.1.0.** Append, replace and delete-and-append now take matrix
blocks, including for a single variable. Pass `k[:, None]` for the old vector
and `jnp.asarray(kappa, dtype=k.dtype).reshape(1, 1)` for the old scalar.
Replacement takes the block's starting index. Delete and delete-and-append
take an index array such as `jnp.array([i])`. The same changes apply to the
corresponding factor methods.

## 0.1.0

- First release
