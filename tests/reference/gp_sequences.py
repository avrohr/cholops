"""GP covariance fixtures and identical sequence objectives for stability tests."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from cholops import chol_delete_append, chol_replace_last


@dataclass(frozen=True)
class Case:
    scenario: str
    n: int
    steps: int
    pattern: str
    block: int

    @property
    def methods(self):
        return (
            ("solve_scan", "recompute")
            if self.pattern == "fifo"
            else ("schur", "solve_scan", "recompute")
        )


def kernel(x, z, theta, scenario):
    variance, length = jnp.exp(theta[0]), jnp.exp(theta[1])
    difference = (x[:, None] - z[None, :]) / length
    if scenario == "se_clustered":
        return variance * jnp.exp(-0.5 * difference**2)
    distance = jnp.sqrt(jnp.asarray(3, dtype=x.dtype)) * jnp.abs(difference)
    return variance * (1 + distance) * jnp.exp(-distance)


def covariance(x, theta, scenario):
    return kernel(x, x, theta, scenario) + jnp.exp(theta[2]) * jnp.eye(
        x.shape[0], dtype=x.dtype
    )


def fixture(case, dtype, noise, seed=0):
    rng = np.random.default_rng(seed)
    if case.scenario == "se_clustered":
        # Close pairs span [0,1]; incoming locations also lie near this grid.
        centers = np.linspace(0, 1, (case.n + 1) // 2)
        x = np.repeat(centers, 2)[: case.n]
        x += 1e-4 * rng.normal(size=x.shape)
        incoming = rng.choice(centers, (case.steps, case.block))
        incoming += 1e-4 * rng.normal(size=incoming.shape)
        length = 0.5
    else:
        x = np.linspace(0, 1, case.n)
        incoming = rng.uniform(0, 1, (case.steps, case.block))
        length = 0.3
    # Non-smooth observation residuals excite small-eigenvalue directions.
    y = np.sin(6 * x) + 0.05 * rng.normal(size=x.shape)
    new_y = np.sin(6 * incoming) + 0.05 * rng.normal(size=incoming.shape)
    theta = np.log(np.asarray([1.0, length, noise]))
    return tuple(jnp.asarray(a, dtype=dtype) for a in (x, incoming, y, new_y, theta))


def nll(L, y):
    whitened = solve_triangular(L, y, lower=True)
    return (jnp.log(jnp.diag(L)).sum() + 0.5 * jnp.vdot(whitened, whitened)) / y.size


def step_function(case, method, theta, *, trace):
    keep_count = case.n - case.block

    def step(carry, request):
        L, x, y = carry
        new_x, new_y = request
        keep_x = x[case.block :] if case.pattern == "fifo" else x[:keep_count]
        keep_y = y[case.block :] if case.pattern == "fifo" else y[:keep_count]
        target_x = jnp.concatenate((keep_x, new_x))
        target_y = jnp.concatenate((keep_y, new_y))
        if method == "recompute":
            L = jnp.linalg.cholesky(covariance(target_x, theta, case.scenario))
        elif method == "schur":
            K = kernel(keep_x, new_x, theta, case.scenario)
            C = covariance(new_x, theta, case.scenario)
            L = chol_replace_last(L, K, C)
        else:
            # Replacing index n-r repeatedly removes every old trailing point
            # before the next block boundary; intermediate states remain valid.
            index = 0 if case.pattern == "fifo" else keep_count
            positions = jnp.arange(case.n - 1)
            survivors = positions + (positions >= index)
            for j in range(case.block):
                retained_x = x[survivors]
                k = kernel(retained_x, new_x[j : j + 1], theta, case.scenario)[:, 0]
                kappa = jnp.exp(theta[0]) + jnp.exp(theta[2])
                L = chol_delete_append(
                    L,
                    jnp.asarray([index]),
                    k[:, None],
                    kappa.reshape(1, 1),
                )
                x = jnp.concatenate((retained_x, new_x[j : j + 1]))
        score = nll(L, target_y)
        return (L, target_x, target_y), (L, score) if trace else score

    return step


def function(case, method, *, trace=False):
    def evaluate(initial_x, incoming_x, initial_y, incoming_y, theta):
        initial = jnp.linalg.cholesky(covariance(initial_x, theta, case.scenario))
        carry = initial, initial_x, initial_y
        requests = incoming_x, incoming_y
        step = step_function(case, method, theta, trace=trace)
        if trace:
            _, outputs = jax.lax.scan(step, carry, requests)
            return initial, *outputs
        # Bound saved dense states across time, using the same policy everywhere.
        block = min(4, case.steps)
        requests = jax.tree.map(
            lambda a: a.reshape((case.steps // block, block, *a.shape[1:])), requests
        )

        def run_block(carry, requests):
            return jax.lax.scan(jax.checkpoint(step), carry, requests)

        _, scores = jax.lax.scan(jax.checkpoint(run_block), carry, requests)
        scores = scores.reshape(case.steps)
        weights = 1 + jnp.arange(case.steps, dtype=scores.dtype) / case.steps
        return jnp.mean(scores * weights)

    return evaluate


def coordinate_history(case, args):
    x, incoming, y, new_y, _ = map(np.asarray, args)
    history = [(x, y)]
    for a, b in zip(incoming, new_y, strict=True):
        keep = (
            slice(case.block, None)
            if case.pattern == "fifo"
            else slice(None, case.n - case.block)
        )
        x, y = np.concatenate((x[keep], a)), np.concatenate((y[keep], b))
        history.append((x, y))
    return history
