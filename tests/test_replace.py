"""Ordering, independent covariance checks, and first-order reverse derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cholops import chol_delete_append, chol_replace
from cholops.tensors.replacement import _downdate

DTYPE = np.float64 if jax.config.x64_enabled else np.float32
TOL = 3e-10 if jax.config.x64_enabled else 8e-5


def close(a, b, tol=TOL):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=tol, atol=tol)


def edit(A, index, k, kappa):
    keep = jnp.arange(A.shape[0] - 1)
    keep = keep + (keep >= index)
    return A.at[index, keep].set(k).at[keep, index].set(k).at[index, index].set(kappa)


def dense(L, index, k, kappa):
    L = jnp.tril(L)
    return jnp.linalg.cholesky(edit(L @ L.T, index, k, kappa))


def fixture(n, index):
    rng = np.random.default_rng(318)
    X = rng.normal(size=(n, 3)) / 2
    new = rng.normal(size=3) / 2
    A = X @ X.T + np.eye(n)
    keep = np.delete(np.arange(n), index)
    return tuple(
        jnp.asarray(a, dtype=DTYPE)
        for a in (np.linalg.cholesky(A), X[keep] @ new, 1 + new @ new)
    )


@pytest.mark.parametrize(
    "n,index",
    [
        pytest.param(1, 0, id="singleton"),
        pytest.param(2, 0, id="two-first"),
        pytest.param(2, 1, id="two-last"),
        *[pytest.param(7, i, id=f"seven-index-{i}") for i in (0, 3, 5, 6)],
        *[
            pytest.param(32, i, marks=pytest.mark.full, id=f"large-index-{i}")
            for i in (0, 16, 30, 31)
        ],
    ],
)
def test_replacement_and_all_continuous_vjps(n, index):
    L, k, kappa = fixture(n, index)
    args = L, jnp.asarray(index), k, kappa
    W = jnp.sin(jnp.arange(n * n, dtype=L.dtype) + 0.2).reshape((n, n))
    for compiled in (False, True):

        def evaluate(fn):
            def f(L, i, k, c):
                return jnp.sum(fn(L, i, k, c) * W)

            grad = jax.value_and_grad(f, argnums=(0, 2, 3))
            return (
                (jax.jit(fn)(*args), jax.jit(grad)(*args))
                if compiled
                else (fn(*args), grad(*args))
            )

        close(evaluate(chol_replace), evaluate(dense))
    expected = np.asarray(L) @ np.asarray(L).T
    keep = np.delete(np.arange(n), index)
    expected[index, keep] = expected[keep, index] = np.asarray(k)
    expected[index, index] = kappa
    out = jax.jit(chol_replace)(*args)
    close(out, np.linalg.cholesky(expected.astype(np.float64)))
    close(out @ out.T, expected)
    # Index preservation includes an exactly unchanged leading factor.
    np.testing.assert_array_equal(out[:index, :index], L[:index, :index])
    shifted = jax.jit(chol_delete_append)(
        L, jnp.asarray([index]), k[:, None], jnp.asarray(kappa).reshape(1, 1)
    )
    order = np.r_[keep, index]
    close(shifted @ shifted.T, expected[np.ix_(order, order)])


def test_dynamic_batch_shared_inputs_and_ignored_upper_triangle():
    L, _, _ = fixture(7, 0)
    L = L.at[jnp.triu_indices(7, 1)].set(jnp.nan)
    factors = jnp.stack((L, 1.2 * L, 0.9 * L))
    indices = jnp.array([0, 3, 6])
    k = jnp.linspace(0.01, 0.1, 6, dtype=L.dtype)
    c = jnp.asarray(2.0, dtype=L.dtype)

    def evaluate(fn):
        fn = jax.vmap(fn, (0, 0, None, None))
        return jax.jit(
            jax.value_and_grad(
                lambda L, k, c: jnp.sum(fn(L, indices, k, c)), argnums=(0, 1, 2)
            )
        )(factors, k, c)

    close(evaluate(chol_replace), evaluate(dense))


@pytest.mark.parametrize("margin", [0.5, 0.001, 1.0])
def test_downdate_boundary_and_vjp(margin):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(7, 3)) / 4
    L = jnp.asarray(np.linalg.cholesky(np.eye(7) + X @ X.T), DTYPE)
    v = rng.normal(size=7)
    v *= np.sqrt((1 - margin) / (v @ v))
    u = L @ jnp.asarray(v, DTYPE)
    W = jnp.asarray(rng.normal(size=(7, 7)), DTYPE)

    def evaluate(fn):
        return jax.jit(
            jax.value_and_grad(lambda L, u: jnp.sum(fn(L, u) * W), argnums=(0, 1))
        )(L, u)

    def reference(L, u):
        L = jnp.tril(L)
        return jnp.linalg.cholesky(L @ L.T - jnp.outer(u, u))

    close(_downdate(L, u), reference(L, u), tol=TOL * 5)
    close(evaluate(_downdate), evaluate(reference), tol=TOL * 5)


def test_sequence_all_states_and_gradients():
    n, steps = 8, 32
    rng = np.random.default_rng(2)
    X = jnp.asarray(rng.normal(size=(n, 3)) / 2, DTYPE).at[::3].set(0)
    incoming = jnp.asarray(rng.normal(size=(steps, 3)) / 2, DTYPE).at[::4].set(0)
    indices = jnp.asarray([0, 7, 3, 1, 5, 0, 6, 2] * 4)

    def sequence(X, incoming, scale, method):
        L = jnp.linalg.cholesky(jnp.eye(n) + scale * (X @ X.T))

        def step(carry, request):
            X, L = carry
            index, new = request
            keep = jnp.arange(n - 1) + (jnp.arange(n - 1) >= index)
            k, c = scale * (X[keep] @ new), 1 + scale * jnp.vdot(new, new)
            X = X.at[index].set(new)
            if method == "recompute":
                L = jnp.linalg.cholesky(jnp.eye(n) + scale * (X @ X.T))
            else:
                L = chol_replace(L, index, k, c)
            y = X[:, 0]
            z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
            score = (jnp.log(jnp.diag(L)).sum() + 0.5 * jnp.vdot(z, z)) / n
            return (X, L), (L, score)

        return jax.lax.scan(step, (X, L), (indices, incoming))[1]

    args = X, incoming, jnp.asarray(0.7, DTYPE)
    close(
        jax.jit(lambda *xs: sequence(*xs, "replace"))(*args),
        jax.jit(lambda *xs: sequence(*xs, "recompute"))(*args),
    )

    def vg(method):
        return jax.jit(
            jax.value_and_grad(
                lambda *xs: jnp.mean(sequence(*xs, method)[1]), argnums=(0, 1, 2)
            )
        )(*args)

    close(vg("replace"), vg("recompute"))


def test_invalid_replacement_is_visible_and_bad_inputs_raise():
    L = jnp.eye(4, dtype=DTYPE)
    out = jax.jit(chol_replace)(L, jnp.asarray(1), jnp.ones(3, dtype=DTYPE), 0.1)
    assert not np.isfinite(out).all()
    with pytest.raises(ValueError):
        chol_replace(L, -1, jnp.zeros(3, dtype=DTYPE), 1.0)
    with pytest.raises(ValueError):
        chol_replace(L, 1, jnp.zeros(4, dtype=DTYPE), 1.0)


@pytest.mark.parametrize("index", [0, 4, 6, 7])
def test_conditioned_replacement_vjp_against_decimal(index):
    from decimal import Decimal, localcontext

    from tests.reference.decimal_ad import (
        decimal_matrix,
        matmul,
        pullback,
        transpose,
    )

    # Eight correlated points make both modification and bordering derivatives
    # sensitive, while keeping the reference cheap. The precision-specific noise
    # keeps the rounded target inside the positive-definite domain.
    n = 8
    x = np.linspace(-0.3, 0.4, n)
    noise = 1e-8 if jax.config.x64_enabled else 1e-3
    A = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2) + noise * np.eye(n)
    L = np.linalg.cholesky(A).astype(DTYPE)
    keep = np.delete(np.arange(n), index)
    k = np.exp(-0.5 * (x[keep] - 0.17) ** 2).astype(DTYPE)
    c = np.asarray(1 + noise, DTYPE)
    G = np.random.default_rng(59).normal(size=(n, n)).astype(DTYPE)
    with localcontext() as ctx:
        ctx.prec = 80
        dL = decimal_matrix(L)
        target = matmul(dL, transpose(dL))
        for j, value in zip(keep, k, strict=True):
            target[index][j] = target[j][index] = Decimal.from_float(float(value))
        target[index][index] = Decimal.from_float(float(c))
        factor = [[Decimal(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                value = target[i][j] - sum(
                    factor[i][a] * factor[j][a] for a in range(j)
                )
                factor[i][j] = value.sqrt() if i == j else value / factor[j][j]
        # The Decimal factor is rounded once for the existing exact-input local
        # adjoint reference, with errors far below the tested precision budgets.
        Q = pullback(np.asarray(factor, dtype=np.float64), G)
        qold = Q.copy()
        qold[index, :] = qold[:, index] = 0
        expected = (np.tril(2 * qold @ L), 2 * Q[keep, index], Q[index, index])
    actual = jax.jit(
        jax.grad(
            lambda L, k, c: jnp.sum(chol_replace(L, index, k, c) * G),
            argnums=(0, 1, 2),
        )
    )(jnp.asarray(L), jnp.asarray(k), jnp.asarray(c))
    for a, b in zip(actual, expected, strict=True):
        relative = np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b)
        assert relative < (2e-6 if jax.config.x64_enabled else 2e-3)
