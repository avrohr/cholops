import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import chol_append, chol_replace_last

DTYPE = np.float64 if jax.config.x64_enabled else np.float32
TOL = 2e-10 if jax.config.x64_enabled else 4e-5


def close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)


def dense(L, K, C):
    m = K.shape[0]
    leading = jnp.tril(L)[:m, :m]
    return jnp.linalg.cholesky(jnp.block([[leading @ leading.T, K], [K.T, C]]))


@pytest.mark.parametrize("operation", [chol_append, chol_replace_last])
@pytest.mark.parametrize("m,r", [(0, 0), (5, 0), (5, 1), (5, 3), (0, 3), (2, 5)])
def test_block_factor_jvp_and_vjp(operation, m, r):
    rng = np.random.default_rng(48)
    n = m if operation is chol_append else m + r
    X = rng.normal(size=(n, 4))
    A = X @ X.T + np.eye(n)
    L = jnp.asarray(np.linalg.cholesky(A), DTYPE)
    new = rng.normal(size=(r, 4))
    K = jnp.asarray(X[:m] @ new.T, DTYPE)
    C = jnp.asarray(new @ new.T + np.eye(r), DTYPE)
    args = L, K, C
    dL = jnp.asarray(rng.normal(size=L.shape), DTYPE)
    dK = jnp.asarray(rng.normal(size=K.shape), DTYPE)
    dC = jnp.asarray(rng.normal(size=C.shape), DTYPE)
    # Asymmetric directions also test the symmetric-input convention for C.
    G = jnp.asarray(rng.normal(size=(m + r, m + r)), DTYPE)
    candidate = jax.jit(operation)
    reference = jax.jit(dense)
    actual = candidate(*args)
    close(actual, reference(*args))
    np.testing.assert_array_equal(actual[:m, :m], L[:m, :m])
    close(
        jax.jvp(candidate, args, (dL, dK, dC)), jax.jvp(reference, args, (dL, dK, dC))
    )
    gradients = jax.vjp(candidate, *args)[1](G)
    close(gradients, jax.vjp(reference, *args)[1](G))
    np.testing.assert_array_equal(gradients[2], gradients[2].T)


@pytest.mark.parametrize("operation", [chol_append, chol_replace_last])
def test_unused_factor_entries_are_ignored(operation):
    n = 4 if operation is chol_append else 6
    L = jnp.eye(n, dtype=DTYPE).at[4:, :].set(jnp.nan)
    L = L.at[:, 4:].set(jnp.nan).at[jnp.triu_indices(n, 1)].set(jnp.nan)
    K = jnp.full((4, 2), 0.05, dtype=DTYPE)
    C = jnp.eye(2, dtype=DTYPE)
    result = jax.jit(operation)(L, K, C)
    close(result, dense(L, K, C))
    gradient = jax.jit(jax.grad(lambda L: operation(L, K, C).sum()))(L)
    np.testing.assert_array_equal(gradient[4:], 0)
    np.testing.assert_array_equal(gradient[:, 4:], 0)
    np.testing.assert_array_equal(gradient[jnp.triu_indices(n, 1)], 0)


@pytest.mark.parametrize("operation", [chol_append, chol_replace_last])
def test_batched_blocks_and_shared_input_gradients(operation):
    L = jnp.stack((jnp.eye(5, dtype=DTYPE), 2 * jnp.eye(5, dtype=DTYPE)))
    m = 5 if operation is chol_append else 3
    K = jnp.full((m, 2), 0.1, dtype=DTYPE)
    C = jnp.eye(2, dtype=DTYPE)
    candidate = jax.vmap(operation, (0, None, None))
    reference = jax.vmap(dense, (0, None, None))
    close(jax.jit(candidate)(L, K, C), jax.jit(reference)(L, K, C))
    close(
        jax.jit(jax.grad(lambda *a: jnp.sin(candidate(*a)).sum(), (0, 1, 2)))(L, K, C),
        jax.jit(jax.grad(lambda *a: jnp.sin(reference(*a)).sum(), (0, 1, 2)))(L, K, C),
    )


def test_invalid_block_replacement_is_visible_and_bad_shapes_raise():
    L = jnp.eye(5, dtype=DTYPE)
    K = jnp.full((3, 2), 0.1, dtype=DTYPE)
    C = jnp.eye(2, dtype=DTYPE)
    bad = jax.jit(chol_replace_last)(L, K, -C)
    assert not np.isfinite(bad[3:, 3:]).all()
    with pytest.raises(ValueError):
        chol_replace_last(L, K[:, :1], C)
