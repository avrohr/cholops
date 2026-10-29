"""Canonical block replacement matches the target matrix and its derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import Cholesky, chol_replace, cholesky

DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
TOL = 3e-10 if jax.config.x64_enabled else 8e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def reference(L, start, K, C):
    n, r = L.shape[0], C.shape[0]
    indices = start + jnp.arange(r)
    keep = jnp.setdiff1d(jnp.arange(n), indices, size=n - r)
    A = jnp.tril(L) @ jnp.tril(L).T
    A = A.at[keep[:, None], indices].set(K)
    A = A.at[indices[:, None], keep].set(K.T)
    A = A.at[indices[:, None], indices].set((C + C.T) / 2)
    return jnp.linalg.cholesky(A)


def fixture(n, start, r):
    keep = np.r_[np.arange(start), np.arange(start + r, n)]
    rng = np.random.default_rng(79)
    X = jnp.asarray(rng.normal(size=(n, 3)) / 3, DTYPE)
    new = jnp.asarray(rng.normal(size=(r, 3)) / 3, DTYPE)
    L = cholesky(jnp.eye(n, dtype=DTYPE) + X @ X.T)
    return L, X[keep] @ new.T, jnp.eye(r, dtype=DTYPE) + new @ new.T


def evaluate(fn, args):
    directions = tuple(
        jnp.cos(jnp.arange(x.size, dtype=DTYPE)).reshape(x.shape) / 20 for x in args
    )
    value, tangent = jax.jit(lambda *a: jax.jvp(fn, a, directions))(*args)
    G = jnp.sin(jnp.arange(value.size, dtype=DTYPE)).reshape(value.shape)
    gradients = jax.jit(lambda *a: jax.vjp(fn, *a)[1](G))(*args)
    return value, tangent, gradients


@pytest.mark.parametrize(
    "n,start,r",
    [
        (0, 0, 0),
        (1, 0, 1),
        (6, 6, 0),
        (6, 3, 1),
        (6, 0, 2),
        (6, 2, 2),
        (6, 4, 2),
        (6, 0, 6),
        (70, 62, 4),
    ],
)
def test_block_replacement_order_and_all_derivatives(n, start, r):
    L, K, C = fixture(n, start, r)
    L = L.at[jnp.triu_indices(n, 1)].set(jnp.nan)
    actual = evaluate(lambda *a: chol_replace(a[0], start, *a[1:]), (L, K, C))
    expected = evaluate(lambda *a: reference(a[0], start, *a[1:]), (L, K, C))
    close(actual, expected)
    np.testing.assert_array_equal(actual[2][2], actual[2][2].T)
    first = start if r else n
    np.testing.assert_array_equal(
        actual[0][:first, :first], jnp.tril(L)[:first, :first]
    )
    close(Cholesky.from_factor(L).replace(start, K, C).factor, expected[0])


def test_runtime_block_start_with_shared_inputs_and_checkpointing():
    starts = jnp.array([0, 2, 4], jnp.int32)
    args = fixture(6, 0, 2)

    def candidate(starts, L, K, C):
        return jax.vmap(jax.checkpoint(chol_replace), (None, 0, None, None))(
            L, starts, K, C
        )

    def dense(starts, L, K, C):
        return jax.vmap(reference, (None, 0, None, None))(L, starts, K, C)

    def run(fn):
        directions = jax.tree.map(lambda x: jnp.cos(x) / 20, args)

        @jax.jit
        def transformed(starts, *args):
            def operation(*a):
                return fn(starts, *a)

            value, tangent = jax.jvp(operation, args, directions)
            G = jnp.sin(jnp.arange(value.size, dtype=DTYPE)).reshape(value.shape)
            return value, tangent, jax.vjp(operation, *args)[1](G)

        return transformed(starts, *args)

    close(run(candidate), run(dense))


def test_block_replacement_does_not_mix_old_and_new_covariances():
    L = jnp.diag(jnp.array([0.1, 0.1, 1.0], DTYPE))
    K = jnp.array([[0.1, 0.2]], DTYPE)
    C = jnp.array([[1.0, 0.8], [0.8, 1.0]], DTYPE)
    # Inserting C[0,1] while keeping the old variance 0.01 at slot 1 would
    # give a negative 2x2 determinant: 1*0.01 - 0.8**2.
    close(
        evaluate(lambda *a: chol_replace(a[0], 0, *a[1:]), (L, K, C)),
        evaluate(lambda *a: reference(a[0], 0, *a[1:]), (L, K, C)),
    )
