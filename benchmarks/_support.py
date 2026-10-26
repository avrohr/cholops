"""Shared fixtures, numerical diagnostics and sequence checkpointing."""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _leaf_error(a, b):
    finite = jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b))
    error = jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1)
    maximum = jnp.max(jnp.abs(a - b), initial=0)
    return finite, error, maximum


def error_record(actual, expected):
    """Report the largest leaf error, with reference norms floored at one."""
    leaves = [
        _leaf_error(a, b)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    ]
    jax.block_until_ready(leaves)
    finite = all(
        bool(f) and np.isfinite(float(e)) and np.isfinite(float(m))
        for f, e, m in leaves
    )
    return {
        "finite": finite,
        "max_leaf_normalized_error": max(float(e) for _, e, _ in leaves)
        if finite
        else None,
        "max_absolute_error": max(float(m) for _, _, m in leaves) if finite else None,
    }


def keep_indices(n, index):
    p = jnp.arange(n - 1)
    return p + (p >= index)


@jax.jit
def _relative_leaf_error(actual, expected):
    denominator = jnp.maximum(jnp.linalg.norm(expected), jnp.finfo(expected.dtype).tiny)
    return jnp.linalg.norm(actual - expected) / denominator


def accuracy_record(actual, expected):
    """Also report relative error so small reference values remain visible."""
    record = error_record(actual, expected)
    errors = [
        _relative_leaf_error(a, b)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    ]
    jax.block_until_ready(errors)
    relative = max(float(e) for e in errors)
    record["finite"] = bool(record["finite"] and np.isfinite(relative))
    record["max_leaf_relative_error"] = relative if np.isfinite(relative) else None
    return record


def fixture(n, position, dtype, seed, steps=16):
    """Create features for a Gram covariance and a moving replacement sequence."""
    rng = np.random.default_rng(seed)
    rank = min(n, 16)
    X = jnp.asarray(rng.normal(size=(n, rank)) / np.sqrt(rank), dtype)
    incoming = jnp.asarray(rng.normal(size=(steps, rank)) / np.sqrt(rank), dtype)
    start = {"first": 0, "middle": n // 2, "penultimate": n - 2, "last": n - 1}[
        position
    ]
    direction = -1 if position in ("penultimate", "last") else 1
    indices = jnp.asarray((start + direction * np.arange(steps)) % n, jnp.int32)
    return X, incoming, jnp.asarray(0.7, dtype), indices


def scan_sequence(step, carry, requests, checkpoint):
    """Run updates, optionally recomputing four-step blocks during reverse AD."""
    if checkpoint == "none":
        return jax.lax.scan(step, carry, requests)
    # Checkpoint steps and blocks to trade recomputation for less saved state.
    # Reverse AD retains block boundaries and reconstructs intermediate steps.
    steps = requests[0].shape[0]
    block = min(4, steps)
    if steps % block:
        raise ValueError("block checkpointing requires a step count divisible by four")
    requests = jax.tree.map(
        lambda x: x.reshape((steps // block, block, *x.shape[1:])), requests
    )

    def run_block(carry, xs):
        return jax.lax.scan(jax.checkpoint(step), carry, xs)

    carry, scores = jax.lax.scan(jax.checkpoint(run_block), carry, requests)
    return carry, scores.reshape((steps,))
