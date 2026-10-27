import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import chol_append, chol_append_block, chol_replace_last, tensors

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


@pytest.mark.parametrize("n,r", [(0, 0), (0, 3), (5, 0), (5, 1), (5, 3)])
def test_block_append_factor_jvp_and_vjp(n, r):
    rng = np.random.default_rng(49)
    X = rng.normal(size=(n + r, 4)) / 2
    A = jnp.asarray(X @ X.T + np.eye(n + r), DTYPE)
    L = jnp.linalg.cholesky(A[:n, :n])
    args = L, A[:n, n:], A[n:, n:]
    # Include asymmetric C directions and unused upper-factor directions.
    tangents = tuple(jnp.asarray(rng.normal(size=a.shape), DTYPE) for a in args)
    G = jnp.asarray(rng.normal(size=A.shape), DTYPE)
    candidate = jax.jit(chol_append_block)
    assert tensors.chol_append_block is chol_append_block
    actual = candidate(*args)
    close(actual, jnp.linalg.cholesky(A))
    np.testing.assert_array_equal(actual[:n, :n], L)
    close(jax.jvp(candidate, args, tangents), jax.jvp(dense, args, tangents))
    close(jax.vjp(candidate, *args)[1](G), jax.vjp(dense, *args)[1](G))
    if r == 1:
        close(actual, chol_append(L, args[1][:, 0], args[2][0, 0]))


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


@pytest.mark.parametrize(
    "L_shape,K_shape,C_shape",
    [
        ((3,), (3, 2), (2, 2)),
        ((3, 2), (3, 2), (2, 2)),
        ((1, 3, 3), (3, 2), (2, 2)),
        ((3, 3), (3, 2), (2,)),
        ((3, 3), (3, 2), (2, 1)),
        ((3, 3), (2, 2), (2, 2)),
        ((3, 3), (4, 2), (2, 2)),
        ((3, 3), (3, 1), (2, 2)),
        ((3, 3), (3,), (1, 1)),
        ((3, 3), (4, 0), (0, 0)),
    ],
)
def test_block_append_rejects_bad_shapes(L_shape, K_shape, C_shape):
    args = [jnp.ones(shape, DTYPE) for shape in (L_shape, K_shape, C_shape)]
    with pytest.raises(ValueError):
        chol_append_block(*args)


@pytest.mark.parametrize("argument", [0, 1, 2])
@pytest.mark.parametrize(
    "dtype",
    [jnp.int32, jnp.complex64, jnp.float16]
    + ([jnp.float32] if jax.config.x64_enabled else []),
)
def test_block_append_rejects_incompatible_dtypes(argument, dtype):
    args = [jnp.eye(3, dtype=DTYPE), jnp.zeros((3, 2), DTYPE), jnp.eye(2, dtype=DTYPE)]
    args[argument] = args[argument].astype(dtype)
    with pytest.raises(TypeError):
        chol_append_block(*args)


def test_block_append_ignores_upper_triangle_and_batches_all_inputs():
    L = jnp.array([[1.2, jnp.nan], [0.3, 1.1]], DTYPE)
    K = jnp.array([[0.1, -0.2], [0.3, 0.1]], DTYPE)
    C = jnp.array([[1.5, 0.2], [0.2, 1.2]], DTYPE)
    args = L, K, C
    close(chol_append_block(*args), dense(*args))
    gradient = jax.jit(jax.grad(lambda L: chol_append_block(L, K, C).sum()))(L)
    np.testing.assert_array_equal(gradient[jnp.triu_indices(2, 1)], 0)
    batched = tuple(jnp.stack((a, 1.1 * a)) for a in args)
    close(jax.jit(jax.vmap(chol_append_block))(*batched), jax.vmap(dense)(*batched))


@pytest.mark.parametrize(
    "cross,C",
    [(0.0, [[0.0]]), (0.0, [[-1.0]]), (0.0, [[1.0, 2.0], [2.0, 1.0]]), (1.0, [[1.0]])],
)
def test_block_append_non_spd_schur_complement_is_visible(cross, C):
    C = jnp.asarray(C, DTYPE)
    L = jnp.eye(3, dtype=DTYPE)
    K = jnp.full((3, C.shape[0]), cross, DTYPE)
    result = jax.jit(chol_append_block)(L, K, C)
    assert not np.isfinite(result[3:, 3:]).all()
    np.testing.assert_array_equal(result[:3, :3], L)


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
