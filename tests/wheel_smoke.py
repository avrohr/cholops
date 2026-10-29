"""Run outside the checkout with only the installed wheel and its dependencies."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cholops
from cholops import factors, tensors
from cholops.factors import Cholesky, PermutedCholesky


def main():
    assert Path(cholops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    # Both grouped imports and the original root imports expose the same objects.
    for namespace in (factors, tensors):
        for name in namespace.__all__:
            assert getattr(namespace, name) is getattr(cholops, name)
    A = jnp.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.2]])
    args = A, jnp.array([0.2, 0.1]), jnp.asarray(1.3), jnp.array([1.0, -0.5, 0.2])
    tangents = jax.tree.map(lambda x: jnp.ones_like(x) * 0.03, args)

    def reference(A, k, kappa, rhs):
        keep = jnp.array([0, 2])
        matrix = (A + A.T) * 0.5
        matrix = matrix.at[1, keep].set(k).at[keep, 1].set(k).at[1, 1].set(kappa)
        solution = jnp.linalg.solve(matrix, rhs)
        return jnp.linalg.slogdet(matrix)[1] + rhs @ solution + jnp.sum(solution)

    def evaluate(fn):
        return (
            jax.jit(jax.value_and_grad(fn, argnums=(0, 1, 2, 3)))(*args),
            jax.jit(lambda *xs: jax.jvp(fn, xs, tangents))(*args),
        )

    expected = evaluate(reference)
    tolerance = 2e-10 if jax.config.x64_enabled else 5e-5
    for cls in (Cholesky, PermutedCholesky):

        def score(A, k, kappa, rhs):
            factor = cls.from_matrix(A)
            factor = factor.replace(1, k[:, None], kappa.reshape(1, 1))
            return factor.logdet() + factor.quadratic(rhs) + jnp.sum(factor.solve(rhs))

        actual = evaluate(score)
        for got, want in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
    print("Installed wheel: both factor classes, JIT, JVP and VJP passed.")


if __name__ == "__main__":
    main()
