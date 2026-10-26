import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.reference.gp_sequences import Case, covariance, fixture, function, nll

pytestmark = pytest.mark.full


@pytest.mark.parametrize("scenario", ["se_clustered", "matern32_uniform"])
@pytest.mark.parametrize("pattern,block", [("fifo", 1), ("tail", 1), ("tail", 3)])
def test_gp_sequence_and_all_gradients(scenario, pattern, block):
    case = Case(scenario, 9, 16, pattern, block)
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    args = fixture(case, dtype, 1e-2)

    def reference(initial_x, incoming_x, initial_y, incoming_y, theta):
        if pattern == "fifo":
            all_x = jnp.concatenate((initial_x, incoming_x[:, 0]))
            all_y = jnp.concatenate((initial_y, incoming_y[:, 0]))
            xs = jax.vmap(lambda t: jax.lax.dynamic_slice_in_dim(all_x, t + 1, case.n))(
                jnp.arange(case.steps)
            )
            ys = jax.vmap(lambda t: jax.lax.dynamic_slice_in_dim(all_y, t + 1, case.n))(
                jnp.arange(case.steps)
            )
        else:
            xs = jnp.concatenate(
                (
                    jnp.broadcast_to(initial_x[:-block], (case.steps, case.n - block)),
                    incoming_x,
                ),
                axis=1,
            )
            ys = jnp.concatenate(
                (
                    jnp.broadcast_to(initial_y[:-block], (case.steps, case.n - block)),
                    incoming_y,
                ),
                axis=1,
            )
        factors = jax.vmap(
            lambda x: jnp.linalg.cholesky(covariance(x, theta, scenario))
        )(xs)
        return factors, jax.vmap(nll)(factors, ys)

    def objective(*args):
        _, scores = reference(*args)
        return jnp.mean(
            scores * (1 + jnp.arange(case.steps, dtype=scores.dtype) / case.steps)
        )

    factors, scores = jax.jit(reference)(*args)
    value_grad = jax.jit(jax.value_and_grad(objective, (0, 1, 2, 3, 4)))(*args)
    tol = 2e-9 if jax.config.x64_enabled else 5e-4
    for method in case.methods:
        _, actual_factors, actual_scores = jax.jit(function(case, method, trace=True))(
            *args
        )
        actual_grad = jax.jit(
            jax.value_and_grad(function(case, method), (0, 1, 2, 3, 4))
        )(*args)
        for a, b in zip(
            jax.tree.leaves((actual_factors, actual_scores, actual_grad)),
            jax.tree.leaves((factors, scores, value_grad)),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, rtol=tol, atol=tol)
