"""Build the computations and reference results measured by benchmarks.run.

The workload factories are operations and sequences.
Each returns named phases: one implementation in one differentiation mode.
Phases in the same comparison group use equivalent inputs and outputs.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from benchmarks._naive import replace as naive_replace
from benchmarks._recompute import replace as recompute_replace
from benchmarks._recompute import replace_matrix
from benchmarks._support import (
    accuracy_record,
    fixture,
    keep_indices,
    scan_sequence,
)
from cholops import (
    Cholesky,
    PermutedCholesky,
    chol_append,
    chol_delete,
    chol_delete_append,
    chol_diag_update,
    chol_rank1,
    chol_rank1_positive,
    chol_replace,
    chol_replace_last,
    cholesky,
)

OPERATIONS = (
    "append",
    "delete",
    "replace",
    "delete_append",
    "replace_last",
    "diag_update",
    "rank1",
    "rank1_positive",
)
IMPLEMENTATIONS = ("canonical", "managed", "recompute")


def _factor_array(state):
    return state.factor if isinstance(state, Cholesky) else state


def _score(L, rhs):
    """Gaussian negative log likelihood per variable, omitting the constant."""
    z = solve_triangular(L, rhs, lower=True)
    return (2 * jnp.sum(jnp.log(jnp.diag(L))) + jnp.sum(z * z)) / (2 * L.shape[0])


def _phase(function, args, expected, group, implementation):
    """Bundle a timed callable with its inputs and untimed accuracy check."""
    return {
        "function": function,
        "args": args,
        "expected": expected,
        "validate": lambda actual: accuracy_record(actual, expected),
        "phase_kind": next(mode for mode in MODES if group.endswith(mode)),
        "comparison_group": group,
        "implementation": implementation,
        "description": (
            "Equivalent computations with runtime indices and no buffer donation"
        ),
    }


MODES = ("forward", "full_vjp", "jvp")


def _direction(x):
    return (
        0.03
        * jnp.cos(jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape)
        / jnp.sqrt(max(x.size, 1))
    )


def _transformed(fn, args, mode, argnums, loss):
    """Select the primal, value-and-gradient, or directional-JVP computation.

    argnums identifies continuous inputs. Indices remain runtime arguments but
    are held fixed during differentiation.
    """
    if mode == "forward":
        return fn, args
    if mode == "full_vjp":
        return jax.value_and_grad(lambda *a: loss(fn(*a)), argnums=argnums), args
    tangents = tuple(jax.tree.map(_direction, args[i]) for i in argnums)

    def push(*inputs):
        primals, directions = inputs[:-1], inputs[-1]

        def continuous(*values):
            supplied = list(primals)
            for i, value in zip(argnums, values, strict=True):
                supplied[i] = value
            return fn(*supplied)

        return jax.jvp(continuous, tuple(primals[i] for i in argnums), directions)

    return push, (*args, tangents)


def _factor_loss(state):
    # Nonuniform weights exercise derivatives throughout the returned factor.
    out = _factor_array(state)
    weights = jnp.cos(jnp.arange(out.size, dtype=out.dtype).reshape(out.shape) + 0.3)
    return jnp.sum(out * weights) / max(out.size, 1)


def _add_operation_phases(
    phases,
    group,
    tensor,
    raw_args,
    reference,
    *,
    argnums,
    modes=MODES,
    implementations=("tensor", "recompute"),
):
    for mode in modes:
        oracle, oracle_args = _transformed(
            reference, raw_args, mode, argnums, _factor_loss
        )
        expected = jax.jit(oracle)(*oracle_args)
        jax.block_until_ready(expected)
        label = f"{group}_{mode}"
        for name in implementations:
            functions = {"tensor": tensor, "recompute": reference}
            if group == "replace":
                functions["naive"] = lambda L, p, i: naive_replace(L, i, *p)
            fn, args = _transformed(
                functions[name], raw_args, mode, argnums, _factor_loss
            )
            phases[f"{name}_{label}"] = _phase(fn, args, expected, label, name)


def _operation(name, L, params, index):
    if name == "delete":
        return chol_delete(L, index[None])
    if name == "delete_append":
        return chol_delete_append(
            L, index[None], params[0][:, None], params[1].reshape(1, 1)
        )
    if name in ("replace", "diag_update"):
        functions = {
            "replace": chol_replace,
            "diag_update": chol_diag_update,
        }
        return functions[name](L, index, *params)
    functions = {
        "append": chol_append,
        "replace_last": chol_replace_last,
        "rank1": chol_rank1,
        "rank1_positive": chol_rank1_positive,
    }
    return functions[name](L, *params)


def _reference(name, L, params, index):
    if name == "replace":
        return recompute_replace(L, index, *params)
    L = jnp.tril(L)
    A = L @ L.T
    keep = keep_indices(L.shape[0], index)
    if name == "append":
        K, C = params
        A = jnp.block([[A, K], [K.T, (C + C.T) / 2]])
    elif name == "delete":
        A = A[keep[:, None], keep[None, :]]
    elif name == "delete_append":
        A = replace_matrix(A, index, *params)
        order = jnp.concatenate((keep, index[None]))
        A = A[order[:, None], order[None, :]]
    elif name == "replace_last":
        K, C = params
        m = A.shape[0] - C.shape[0]
        A = jnp.block([[A[:m, :m], K], [K.T, (C + C.T) / 2]])
    elif name == "diag_update":
        A = A.at[index, index].add(params[0])
    else:
        u = params[0]
        alpha = params[1] if name == "rank1" else 1
        A = A + alpha * jnp.outer(u, u)
    return jnp.linalg.cholesky(A)


def operations(
    n,
    position,
    dtype,
    seed=0,
    *,
    operation=None,
    modes=MODES,
    implementations=("tensor", "recompute"),
    steps=16,
):
    """Compare modifications starting from a factor.

    The dense reference reconstructs L @ L.T inside the measured call.
    Sequence benchmarks instead retain the covariance between steps.
    """
    del steps
    options = dict(modes=modes, implementations=implementations)
    X, incoming, scale, indices = fixture(n, position, dtype, seed)
    index, new = indices[0], incoming[0]
    A = scale * (X @ X.T) + jnp.eye(n, dtype=dtype)
    L = cholesky(A)
    k, c = scale * (X @ new), 1 + scale * jnp.vdot(new, new)
    u = jnp.sin(jnp.arange(n, dtype=dtype) + 0.3) / jnp.sqrt(n)
    # Use four replacement columns, or the whole matrix in smaller smoke cases.
    r = min(4, n)
    tail = incoming[:r]
    params = {
        "append": (k[:, None], c.reshape(1, 1)),
        "delete": (),
        "replace": (k[keep_indices(n, index)], c),
        "delete_append": (k[keep_indices(n, index)], c),
        "replace_last": (
            scale * X[:-r] @ tail.T,
            jnp.eye(r, dtype=dtype) + scale * tail @ tail.T,
        ),
        "diag_update": (jnp.asarray(-0.2, dtype),),
        "rank1": (u, jnp.asarray(-0.2, dtype)),
        "rank1_positive": (u,),
    }
    result = {}
    for name in OPERATIONS:
        if operation is not None and name != operation:
            continue
        _add_operation_phases(
            result,
            name,
            lambda L, p, i, name=name: _operation(name, L, p, i),
            (L, params[name], index),
            lambda L, p, i, name=name: _reference(name, L, p, i),
            argnums=(0, 1),
            **options,
        )
    return result


def _sequence(implementation):
    """Build a complete sequence, including initialization and per-step losses."""

    def run(X, incoming, scale, indices):
        n = X.shape[0]
        A = scale * (X @ X.T) + jnp.eye(n, dtype=X.dtype)
        if implementation == "canonical":
            state = Cholesky.from_matrix(A)
        elif implementation == "managed":
            state = PermutedCholesky.from_matrix(A)
        else:
            state = A

        def step(carry, request):
            X, state = carry
            i, new = request
            keep = keep_indices(n, i)
            k, c = scale * (X[keep] @ new), 1 + scale * jnp.vdot(new, new)
            X = X.at[i].set(new)
            rhs = X[:, 0]
            if implementation in ("canonical", "managed"):
                if implementation == "managed":
                    state = state.replace(i[None], k[:, None], c.reshape(1, 1))
                else:
                    state = state.replace(i, k, c)
                value = (state.logdet() + state.quadratic(rhs)) / (2 * n)
            else:
                # Retain the covariance, avoiding reconstruction from L.
                state = replace_matrix(state, i, k, c)
                L = jnp.linalg.cholesky(state)
                value = _score(L, rhs)
            return (X, state), value

        return scan_sequence(step, (X, state), (indices, incoming), "block")[1]

    return run


def _matched_losses(functions, args, group, argnums, modes=MODES):
    """Compare losses and derivatives against covariance recomputation.

    Reverse mode differentiates the mean when a sequence returns many losses.
    """
    result = {}
    for mode in modes:
        transformed = {
            name: _transformed(fn, args, mode, argnums, jnp.mean)
            for name, fn in functions.items()
        }
        oracle, oracle_args = transformed["recompute"]
        expected = jax.jit(oracle)(*oracle_args)
        jax.block_until_ready(expected)
        for name, (fn, inputs) in transformed.items():
            label = f"{group}_{mode}"
            result[f"{name}_{label}"] = _phase(
                fn,
                inputs,
                expected,
                label,
                name,
            )
    return result


def sequences(
    n,
    position,
    dtype,
    seed=0,
    *,
    operation=None,
    modes=MODES,
    implementations=IMPLEMENTATIONS,
    steps=16,
):
    """Compare full replacement sequences starting from features and a scale.

    Each method builds its initial state inside the measured call. Recomputation
    retains the covariance, while update methods retain a factor. All methods
    use the same four-step checkpoint blocks and evaluate losses in logical
    variable order. Gradients include the initial and incoming features and scale.
    """
    del operation
    args = fixture(n, position, dtype, seed, steps=steps)
    functions = {name: _sequence(name) for name in implementations}
    return _matched_losses(functions, args, "sequence", (0, 1, 2), modes)
