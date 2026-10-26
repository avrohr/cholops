import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import chol_replace_last

DTYPE = np.float64 if jax.config.x64_enabled else np.float32
TOL = 2e-10 if jax.config.x64_enabled else 4e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def dense(L, K, C):
    m = K.shape[0]
    leading = jnp.tril(L)[:m, :m]
    return jnp.linalg.cholesky(jnp.block([[leading @ leading.T, K], [K.T, C]]))


@pytest.mark.parametrize("n,r", [(0, 0), (8, 0), (8, 1), (8, 3), (8, 8)])
def test_block_factor_jvp_and_vjp(n, r):
    rng = np.random.default_rng(48)
    X = rng.normal(size=(n, 4))
    A = X @ X.T + np.eye(n)
    L = jnp.asarray(np.linalg.cholesky(A), DTYPE)
    new = rng.normal(size=(r, 4))
    K = jnp.asarray(X[: n - r] @ new.T, DTYPE)
    C = jnp.asarray(new @ new.T + np.eye(r), DTYPE)
    args = L, K, C
    dL = jnp.asarray(rng.normal(size=L.shape), DTYPE)
    dK = jnp.asarray(rng.normal(size=K.shape), DTYPE)
    dC = jnp.asarray(rng.normal(size=C.shape), DTYPE)
    dC = (dC + dC.T) / 2
    G = jnp.asarray(rng.normal(size=L.shape), DTYPE)
    candidate = jax.jit(chol_replace_last)
    reference = jax.jit(dense)
    actual = candidate(*args)
    close(actual, reference(*args))
    np.testing.assert_array_equal(actual[: n - r, : n - r], L[: n - r, : n - r])
    close(
        jax.jvp(candidate, args, (dL, dK, dC)), jax.jvp(reference, args, (dL, dK, dC))
    )
    close(jax.vjp(candidate, *args)[1](G), jax.vjp(reference, *args)[1](G))


def test_discarded_tail_and_upper_triangle_are_ignored():
    L = jnp.eye(6, dtype=DTYPE).at[4:, :].set(jnp.nan)
    L = L.at[:, 4:].set(jnp.nan).at[jnp.triu_indices(6, 1)].set(jnp.nan)
    K = jnp.full((4, 2), 0.05, dtype=DTYPE)
    C = jnp.eye(2, dtype=DTYPE)
    result = jax.jit(chol_replace_last)(L, K, C)
    close(result, dense(L, K, C))
    gradient = jax.jit(jax.grad(lambda L: chol_replace_last(L, K, C).sum()))(L)
    np.testing.assert_array_equal(gradient[4:], 0)
    np.testing.assert_array_equal(gradient[:, 4:], 0)
    np.testing.assert_array_equal(gradient[jnp.triu_indices(6, 1)], 0)


def test_batched_block_replacement_matches_dense():
    L = jnp.stack((jnp.eye(5, dtype=DTYPE), 2 * jnp.eye(5, dtype=DTYPE)))
    K = jnp.full((3, 2), 0.1, dtype=DTYPE)
    C = jnp.eye(2, dtype=DTYPE)
    close(
        jax.jit(jax.vmap(chol_replace_last, (0, None, None)))(L, K, C),
        jax.jit(jax.vmap(dense, (0, None, None)))(L, K, C),
    )


def test_invalid_block_replacement_is_visible_and_bad_shapes_raise():
    L = jnp.eye(5, dtype=DTYPE)
    K = jnp.full((3, 2), 0.1, dtype=DTYPE)
    C = jnp.eye(2, dtype=DTYPE)
    bad = jax.jit(chol_replace_last)(L, K, -C)
    assert not np.isfinite(bad[3:, 3:]).all()
    with pytest.raises(ValueError):
        chol_replace_last(L, K[:, :1], C)
