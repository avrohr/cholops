"""Ablate initial/trailing Cholesky reverse rules on the frozen GP fixtures."""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from cholops.tensors import cholesky
from tests.reference.gp_sequences import covariance, kernel, nll, step_function


def function(case, method, rule, *, from_factor=False):
    """Use the gp_sequences fixtures with a chosen Cholesky reverse rule.

    rule is native, initial, trailing or both. For recompute, both selects the
    cholops rule at every recomputation. from_factor exposes the downstream factor
    cotangent for independent local arithmetic diagnostics.
    """
    initial_chol = cholesky if rule in ("initial", "both") else jnp.linalg.cholesky
    trailing_chol = cholesky if rule in ("trailing", "both") else jnp.linalg.cholesky

    def evaluate(initial_x, incoming_x, initial_y, incoming_y, theta, initial=None):
        if not from_factor:
            initial = initial_chol(covariance(initial_x, theta, case.scenario))
        carry = initial, initial_x, initial_y
        original_step = step_function(case, method, theta, trace=False)

        def step(carry, request):
            if method == "solve_scan":
                return original_step(carry, request)
            L, x, y = carry
            new_x, new_y = request
            keep = (
                slice(case.block, None)
                if case.pattern == "fifo"
                else slice(None, case.n - case.block)
            )
            target_x = jnp.concatenate((x[keep], new_x))
            target_y = jnp.concatenate((y[keep], new_y))
            if method == "recompute":
                L = trailing_chol(covariance(target_x, theta, case.scenario))
            else:
                m = case.n - case.block
                leading = L[:m, :m]
                K = kernel(x[:m], new_x, theta, case.scenario)
                C = covariance(new_x, theta, case.scenario)
                V = solve_triangular(leading, K, lower=True)
                R = trailing_chol(C - V.T @ V)
                L = jnp.block([[leading, jnp.zeros_like(K)], [V.T, R]])
            return (L, target_x, target_y), nll(L, target_y)

        block = min(4, case.steps)
        requests = jax.tree.map(
            lambda a: a.reshape((case.steps // block, block, *a.shape[1:])),
            (incoming_x, incoming_y),
        )

        def run_block(carry, requests):
            return jax.lax.scan(jax.checkpoint(step), carry, requests)

        _, scores = jax.lax.scan(jax.checkpoint(run_block), carry, requests)
        scores = scores.reshape(case.steps)
        weights = 1 + jnp.arange(case.steps, dtype=scores.dtype) / case.steps
        return jnp.mean(scores * weights)

    return evaluate
