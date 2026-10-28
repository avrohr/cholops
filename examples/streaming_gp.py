"""Keep a Gaussian process over the most recent observations.

A circular buffer replaces the oldest observation when a new one arrives.
PermutedCholesky manages the factor's internal ordering, so locations and
measured values can remain in the same buffer slots. We differentiate a mean
window loss through the entire replacement sequence with respect to the kernel
length scale. The windows overlap, so this is not the joint likelihood of the
whole stream.

Run: JAX_ENABLE_X64=1 uv run --locked python examples/streaming_gp.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from cholops import PermutedCholesky

NOISE_VARIANCE = 0.1


def kernel(x, y, log_length_scale):
    """Unit-variance squared-exponential kernel between two location arrays."""
    length_scale = jnp.exp(log_length_scale)
    distances = (x[:, None] - y[None, :]) / length_scale
    return jnp.exp(-0.5 * distances**2)


@jax.jit
def window_losses(log_length_scale, initial_x, initial_y, incoming_x, incoming_y):
    capacity = initial_x.size
    covariance = kernel(initial_x, initial_x, log_length_scale)
    covariance = covariance + NOISE_VARIANCE * jnp.eye(capacity)
    # Construct inside the differentiated function so the gradient includes
    # the initial factorization as well as every subsequent modification.
    factor = PermutedCholesky.from_matrix(covariance)
    positions = jnp.arange(capacity - 1)

    def step(state, request):
        x, y, factor = state
        index, new_x, new_y = request
        # Skip the outgoing slot. K's rows follow the remaining logical slots,
        # regardless of how PermutedCholesky stores its factor internally.
        keep = positions + (positions >= index)
        K = kernel(x[keep], new_x[None], log_length_scale)
        C = jnp.array([[1.0 + NOISE_VARIANCE]], dtype=K.dtype)
        factor = factor.replace(index[None], K, C)
        x = x.at[index].set(new_x)
        y = y.at[index].set(new_y)
        loss = 0.5 * (
            factor.logdet() + factor.quadratic(y) + capacity * jnp.log(2 * jnp.pi)
        )
        return (x, y, factor), loss

    # The initial data are chronological, so slots 0, 1, ... expire in order.
    # Wrapping around reuses each slot after another full window of arrivals.
    indices = jnp.arange(incoming_x.size) % capacity
    state, losses = jax.lax.scan(
        step, (initial_x, initial_y, factor), (indices, incoming_x, incoming_y)
    )
    return losses.mean(), (state, losses)


def dense_window_losses(log_length_scale, initial_x, initial_y, incoming_x, incoming_y):
    """Reference only: rebuild each window covariance instead of updating it."""
    x, y = initial_x, initial_y
    losses = []
    for step in range(incoming_x.size):
        index = step % x.size
        x = x.at[index].set(incoming_x[step])
        y = y.at[index].set(incoming_y[step])
        covariance = kernel(x, x, log_length_scale) + NOISE_VARIANCE * jnp.eye(x.size)
        loss = 0.5 * (
            jnp.linalg.slogdet(covariance)[1]
            + y @ jnp.linalg.solve(covariance, y)
            + x.size * jnp.log(2 * jnp.pi)
        )
        losses.append(loss)
    losses = jnp.stack(losses)
    return losses.mean(), losses


def main():
    initial_x = jnp.linspace(0.0, 2.5, 6)
    incoming_x = jnp.linspace(3.0, 6.5, 8)
    initial_y = jnp.sin(initial_x)
    incoming_y = jnp.sin(incoming_x) + 0.05 * jnp.cos(3 * incoming_x)
    log_length_scale = jnp.log(0.8)

    # has_aux keeps the final buffer and factor alongside the scalar loss.
    # Only the first argument is differentiated here: the log length scale.
    (mean_loss, (state, losses)), gradient = jax.value_and_grad(
        window_losses, has_aux=True
    )(log_length_scale, initial_x, initial_y, incoming_x, incoming_y)
    x, y, factor = state

    # Predictions use x and y in buffer order. There is no need to inspect or
    # apply the factor's internal permutation, or to sort the buffer by time.
    prediction_x = jnp.array([6.75, 7.0])
    weights = factor.solve(y)
    prediction = kernel(prediction_x, x, log_length_scale) @ weights
    print("Final locations in buffer order:", x)
    print("Negative log likelihood after each arrival:", losses)
    print(f"Mean window loss: {float(mean_loss):.6f}")
    print(f"Derivative with respect to log length scale: {float(gradient):.6f}")
    print("Prediction locations:", prediction_x)
    print("Posterior means:", prediction)

    # Check the whole loss trajectory and its parameter gradient against dense
    # recomputation. Checking only that the gradient is finite is insufficient.
    (expected_mean, expected_losses), expected_gradient = jax.value_and_grad(
        dense_window_losses, has_aux=True
    )(log_length_scale, initial_x, initial_y, incoming_x, incoming_y)
    tolerance = 1e-9 if jax.config.x64_enabled else 3e-4
    np.testing.assert_allclose(losses, expected_losses, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(mean_loss, expected_mean, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=tolerance, atol=tolerance
    )
    covariance = kernel(x, x, log_length_scale) + NOISE_VARIANCE * jnp.eye(x.size)
    np.testing.assert_allclose(
        weights, jnp.linalg.solve(covariance, y), rtol=tolerance, atol=tolerance
    )


if __name__ == "__main__":
    main()
