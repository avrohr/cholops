"""Positive rank-one updates and deletion, including first-order derivatives."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import chol_delete, chol_delete_append, chol_rank1_positive

DTYPE = np.float64 if jax.config.x64_enabled else np.float32
TOL = 2e-10 if jax.config.x64_enabled else 3e-5


def close(actual, expected, tol=TOL):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def fixture(n):
    rng = np.random.default_rng(492)
    X = rng.normal(size=(n, min(n, 5))) / 3
    L = np.linalg.cholesky(X @ X.T + np.eye(n))
    return jnp.asarray(L, dtype=DTYPE), rng


def dense_rank1(L, u):
    L = jnp.tril(L)
    return jnp.linalg.cholesky(L @ L.T + jnp.outer(u, u))


def edit_matrix(A, index, k=None, kappa=None):
    keep = jnp.arange(A.shape[0] - 1)
    keep += keep >= index
    A = A[keep[:, None], keep[None, :]]
    if k is not None:
        A = jnp.pad(A, ((0, 1), (0, 1)))
        A = A.at[-1, :-1].set(k).at[:-1, -1].set(k).at[-1, -1].set(kappa)
    return A


def dense_edit(L, index, k=None, kappa=None):
    L = jnp.tril(L)
    return jnp.linalg.cholesky(
        edit_matrix(
            L @ L.T,
            index[0],
            None if k is None else k[:, 0],
            None if kappa is None else kappa[0, 0],
        )
    )


def evaluate(fn, args, argnums):
    out = jax.jit(fn)(*args)
    weights = jnp.sin(jnp.arange(out.size, dtype=out.dtype) + 0.3).reshape(out.shape)
    value_grad = jax.jit(
        jax.value_and_grad(lambda *xs: jnp.sum(fn(*xs) * weights), argnums=argnums)
    )(*args)
    tangents = tuple(
        jnp.cos(jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape) * 0.03
        if i in argnums
        else jnp.zeros(x.shape, dtype=jax.dtypes.float0)
        for i, x in enumerate(args)
    )
    tangent = jax.jit(lambda *xs: jax.jvp(fn, xs, tangents)[1])(*args)
    return out, value_grad, tangent


@pytest.mark.parametrize("n", [0, 1, 7, pytest.param(32, marks=pytest.mark.full)])
@pytest.mark.parametrize("scale", [0.0, 0.2, 3.0])
def test_positive_update_values_jvp_and_vjp(n, scale):
    L, rng = fixture(n)
    u = jnp.asarray(rng.normal(size=n) * scale, dtype=DTYPE)
    args = L, u
    close(
        evaluate(chol_rank1_positive, args, (0, 1)), evaluate(dense_rank1, args, (0, 1))
    )


@pytest.mark.parametrize(
    "n,index",
    [(1, 0), (7, 0), (7, 3), (7, 6)],
    ids=["singleton", "first", "middle", "last"],
)
@pytest.mark.parametrize("replacement", [False, True])
def test_deletion_and_delete_append_values_jvp_and_vjp(n, index, replacement):
    L, _ = fixture(n)
    index = jnp.asarray(index)
    args = (L, index[None])
    argnums = (0,)
    if replacement:
        args += (
            jnp.linspace(0.01, 0.1, n - 1, dtype=DTYPE)[:, None],
            jnp.asarray([[2.0]], DTYPE),
        )
        argnums = (0, 2, 3)
    fn = chol_delete_append if replacement else chol_delete
    close(evaluate(fn, args, argnums), evaluate(dense_edit, args, argnums))


def test_runtime_indices_batched_ad_and_unused_triangle():
    L, _ = fixture(7)
    L = L.at[jnp.triu_indices(7, 1)].set(jnp.nan)
    factors = jnp.stack((L, 1.2 * L, 0.8 * L))
    indices = jnp.asarray([0, 3, 6])
    k = jnp.zeros(6, dtype=DTYPE)
    scalar = jnp.asarray(2.0, DTYPE)
    args = factors, indices[:, None], k[:, None], scalar.reshape(1, 1)
    actual = evaluate(jax.vmap(chol_delete_append, (0, 0, None, None)), args, (0, 2, 3))
    expected = evaluate(jax.vmap(dense_edit, (0, 0, None, None)), args, (0, 2, 3))
    close(actual, expected)


@pytest.mark.parametrize("delta", [1e-2, 1e-6, 1e-10])
def test_positive_update_that_improves_conditioning(delta):
    # These are the retained factor and deleted column of a 3x3 correlation
    # factor. The updated covariance approaches I as the old factor degenerates.
    q = np.sqrt(1 + delta * delta)
    c = 1 / np.sqrt(2)
    L = jnp.asarray([[delta / q, 0], [c, c]], dtype=DTYPE)
    u = jnp.asarray([1 / q, 0], dtype=DTYPE)
    close(
        evaluate(chol_rank1_positive, (L, u), (0, 1)),
        evaluate(dense_rank1, (L, u), (0, 1)),
    )


def test_prefix_norm_does_not_square_large_whitened_coordinates():
    small = 1e-200 if jax.config.x64_enabled else 1e-20
    L = jnp.asarray([[small]], dtype=DTYPE)
    u = jnp.ones(1, dtype=DTYPE)
    result, (_, (grad_L, grad_u)), tangent = evaluate(
        chol_rank1_positive, (L, u), (0, 1)
    )
    weight = np.sin(0.3)
    np.testing.assert_allclose(result, [[1]], rtol=TOL)
    np.testing.assert_allclose(grad_L, [[small * weight]], rtol=TOL, atol=0)
    np.testing.assert_allclose(grad_u, [weight], rtol=TOL)
    np.testing.assert_allclose(tangent, [[0.03]], rtol=TOL)


def test_fixed_size_sequence_and_gradients():
    n, steps = 8, 16
    rng = np.random.default_rng(23)
    features = jnp.asarray(rng.normal(size=(n, 3)), dtype=DTYPE).at[::3].set(0)
    incoming = jnp.asarray(rng.normal(size=(steps, 3)), dtype=DTYPE).at[::4].set(0)
    indices = jnp.asarray([0, 7, 3, 1, 5, 0, 6, 2] * 2)

    def sequence(X, incoming, scale, *, recompute):
        A = jnp.eye(n) + scale * (X @ X.T)
        initial = A if recompute else jnp.linalg.cholesky(A)

        def step(carry, request):
            state, X = carry
            index, new = request
            keep = jnp.arange(n - 1) + (jnp.arange(n - 1) >= index)
            survivors = X[keep]
            k = scale * (survivors @ new)
            kappa = 1 + scale * jnp.vdot(new, new)
            if recompute:
                state = edit_matrix(state, index, k, kappa)
                L = jnp.linalg.cholesky(state)
            else:
                state = chol_delete_append(
                    state,
                    jnp.asarray([index]),
                    k[:, None],
                    kappa.reshape(1, 1),
                )
                L = state
            score = jnp.log(jnp.diag(L)).sum()
            return (state, jnp.concatenate((survivors, new[None]))), (L, score)

        return jax.lax.scan(step, (initial, X), (indices, incoming))[1]

    args = features, incoming, jnp.asarray(0.3, DTYPE)
    candidate = partial(sequence, recompute=False)
    reference = partial(sequence, recompute=True)
    close(jax.jit(candidate)(*args), jax.jit(reference)(*args))
    weights = jnp.linspace(1, 2, steps, dtype=DTYPE)
    results = [
        jax.jit(
            jax.value_and_grad(lambda *xs: weights @ fn(*xs)[1], argnums=(0, 1, 2))
        )(*args)
        for fn in (candidate, reference)
    ]
    close(*results)


def test_domain_errors_are_not_repaired():
    L, _ = fixture(3)
    out = jax.jit(chol_delete_append)(
        L,
        jnp.asarray([1]),
        jnp.reshape(jnp.zeros(2, DTYPE), (-1, 1)),
        jnp.asarray([[-1.0]]),
    )
    assert np.isnan(out[-1, -1])
    with pytest.raises(ValueError):
        chol_delete(L, jnp.asarray([-1]))
