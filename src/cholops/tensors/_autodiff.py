"""Custom JAX rules for first-order differentiation.

Reuses each operation's reverse rule under JIT, batching and checkpointing.
Only first-order differentiation is supported.
"""

import jax
import jax.numpy as jnp
from jax.experimental import hijax


class _TangentMap(hijax.HiPrim):
    def __init__(self, pullback, residuals, tangents, out_aval, pushforward=None):
        self.in_avals = jax.tree.map(jax.typeof, (residuals, tangents))
        self.out_aval = out_aval
        self.params = dict(
            pullback=pullback,
            pushforward=pushforward,
            in_avals=self.in_avals,
            out_aval=out_aval,
        )
        super().__init__()

    def expand(self, residuals, tangents):
        if self.pushforward is not None:
            return self.pushforward(residuals, tangents)
        # Differentiating the linear pullback also handles masks introduced by
        # batching runtime conditionals. Its primal at zero is discarded.
        zero = jnp.zeros(self.out_aval.shape, self.out_aval.dtype)
        _, pushforward = jax.vjp(lambda G: self.pullback(residuals, G), zero)
        return pushforward(tangents)[0]

    def transpose(self, cotangent, residuals, tangent_accums):
        gradients = self.pullback(residuals, hijax.instantiate_zeros(cotangent))

        def accumulate(accumulator, value):
            if isinstance(accumulator, hijax.GradAccum):
                accumulator.accum(value)

        jax.tree.map(accumulate, tangent_accums, gradients)

    def batch(self, axis_data, args, dims):
        residuals, tangents = args
        residual_dims, tangent_dims = dims

        def pullback(residuals, G):
            gradients = jax.vmap(
                self.pullback, in_axes=(residual_dims, 0), axis_size=axis_data.size
            )(residuals, G)
            # A shared tangent is broadcast in the JVP, so its adjoint sums
            # across the batch; mapped tangents retain their original axis.
            return jax.tree.map(
                lambda g, d: jnp.sum(g, axis=0) if d is None else jnp.moveaxis(g, 0, d),
                gradients,
                tangent_dims,
                is_leaf=lambda x: x is None,
            )

        out_aval = self.out_aval.update(shape=(axis_data.size, *self.out_aval.shape))
        pushforward = None
        if self.pushforward is not None:

            def pushforward(residuals, tangents):
                return jax.vmap(
                    self.pushforward,
                    in_axes=(residual_dims, tangent_dims),
                    axis_size=axis_data.size,
                )(residuals, tangents)

        operation = _TangentMap(pullback, residuals, tangents, out_aval, pushforward)
        return operation(residuals, tangents), 0


def tangent_map(pullback, residuals, tangents, output, pushforward=None):
    """Apply the transpose of a linear pullback, retaining its explicit adjoint."""
    return _TangentMap(pullback, residuals, tangents, jax.typeof(output), pushforward)(
        residuals, tangents
    )
