# Numerical accuracy and differentiation

cholops modifies Cholesky factors without recomputing them from scratch. It also
provides derivative rules for using those modifications inside optimization
problems. This guide explains the numerical choices behind these operations
and what they mean for the accuracy of your results.

For function arguments and input requirements, see the [tensor guide](tensors.md).
For construction and use of factor objects, see the [factor guide](factors.md).

## Conditioning and precision

A matrix is ill-conditioned when small changes in its entries can cause large
changes in a solution or derivative. For example, a covariance matrix can
become ill-conditioned when two variables are almost perfectly correlated.
Rounding errors then matter more, even if the matrix is still positive definite
and its Cholesky factor contains only finite values.

Use float64 when float32 does not give enough accuracy for your problem. Enable
JAX's 64-bit mode before creating your arrays, and construct the inputs in
float64. More precision helps, but there is still a limit to how
ill-conditioned a matrix can be handled reliably.

Downdates subtract from a matrix and can make it cease to be positive definite.
Appends and replacements can also produce an invalid matrix if the new entries
are incompatible with the retained entries. Even before that boundary is
crossed, computing a small difference between nearly equal quantities can lose
accuracy. This occurs, for example, when calculating a new diagonal entry
during append.

The caller must choose modifications that leave the matrix positive definite.
cholops does not add jitter, clip values or retry with a different algorithm.
If an operation returns NaNs or infinities, discard that result and check the
proposed modification. If your model needs regularization, apply it explicitly
when defining the matrix. See [preconditions and failures](tensors.md#preconditions-and-failures)
for the full input requirements.

## Factorization

`cholops.cholesky(A)` computes the same lower factor as JAX's Cholesky function,
using the symmetric part of the real input matrix. The difference is in how
derivatives are calculated.

During reverse differentiation, a gradient with respect to the factor must be
converted into a gradient with respect to the matrix. This requires triangular
solves, which can amplify rounding errors when the factor is ill-conditioned.
The cholops rule makes an intermediate matrix symmetric before those solves.
It removes an antisymmetric component that does not contribute to the desired
derivative, rather than letting it pass through the sensitive solves and cancel
later. This improves the calculation without changing the mathematical
derivative or forming a matrix inverse.

Both factor classes use this rule in `from_matrix`. Block append and
trailing-block replacement also use it when factoring the Schur complement.
If you instead call `from_factor`, the supplied factor keeps the derivative
history of whichever function produced it. Wrapping it does not replace that
function's derivative rule.

## Updating and replacing entries

A rank-one update changes a matrix by adding `outer(u, u)`. The positive-update
algorithm reuses the old factor and needs quadratic work in the matrix size,
compared with cubic work for a full factorization. Its derivative calculation
also has quadratic cost.

The derivative rule solves with the **updated factor**. This matters when an
update improves the matrix's conditioning: solving with the old factor could
still amplify errors associated with the old, poorly conditioned matrix. The
forward calculation also uses scaled norms to avoid unnecessarily squaring
large intermediate values. These choices improve robustness but do not remove
all sources of rounding error or overflow.

Deletion uses a positive rank-one update to obtain the factor of the remaining
matrix. Delete-and-append then extends that factor with the new variable.
Replacing a variable at its original index needs an additional downdate of the
trailing factor, unless the variable is already last. Avoiding this downdate is
one reason to use [PermutedCholesky](factors.md#permutedcholesky) for repeated
replacements. Replacing a trailing block with `replace_last` also avoids this
downdate by keeping the leading factor and factoring only the new remainder.

`chol_rank1` uses a separate algorithm because it must handle positive, negative
and zero weights. The weight enters the calculation directly, so its derivative
is retained even at zero. A zero weight leaves the matrix unchanged, but a
small change in that weight generally changes the result. Diagonal updates use
the same algorithm, starting at the affected column.

## Differentiating through a calculation

The operations support first-order forward and reverse differentiation.

For the optimized updates and replacements, cholops supplies derivative rules
instead of asking JAX to differentiate every step of the factor construction.
The forward and reverse rules describe the same linear derivative and retain
the efficient solves used by the algorithms. They work with `jit`, `vmap` and
fixed-size `lax.scan` loops. Higher-order derivatives, including Hessians, have
not been validated for general use.

The custom rules use an experimental JAX interface, which is why the package
pins its JAX version. They do not change global JAX differentiation rules.
For the formulas behind the update and replacement rules, see the
[adjoint derivation](adjoints.md).

## Accuracy of a complete loss or gradient

Factor methods such as `solve`, `logdet` and `quadratic` work directly with the
stored factor. They use triangular solves, diagonal entries and sums, without
reconstructing the matrix or forming its inverse.

For example, differentiating a Gaussian-process likelihood follows a chain:

`model parameters -> covariance matrix -> factor -> likelihood`.

The custom rules improve the factorization and modification steps in that
chain. Other steps can still lose accuracy. In particular, large contributions
to a gradient can nearly cancel, leaving a small result whose relative error is
large. A factor that accurately reconstructs the matrix therefore does not, by
itself, establish that the final parameter gradient is accurate.

Check the quantities your application actually uses, including gradients when
optimizing a model. Comparing with a fresh factorization is useful for finding
errors accumulated over repeated modifications. Agreement at the same
precision is not an independent guarantee of accuracy, since both calculations
can suffer from the same conditioning problem. The [testing guide](../TESTING.md)
describes the dense and high-precision references used in this library.

For long differentiable sequences, reverse mode may save intermediate factors
at each step. JAX checkpointing can reduce that memory use by recomputing parts
of the sequence during the reverse pass.
