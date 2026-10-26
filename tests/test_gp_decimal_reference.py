"""Check the high-precision references before using them to judge cholops."""

import numpy as np
import pytest

from tests.reference.decimal_ad import gp, sequence
from tests.reference.decimal_gp import reference
from tests.reference.gp_sequences import Case

pytestmark = pytest.mark.full


@pytest.mark.parametrize("scenario", ["se_clustered", "matern32_uniform"])
def test_decimal_gradient_and_precision_convergence(scenario):
    x = np.linspace(0, 1, 6)
    y = np.sin(7 * x) + np.linspace(-0.1, 0.1, 6)
    theta = np.log([1.0, 0.5, 1e-6])
    score, gradient, factor = reference(x, y, theta, scenario, digits=60)
    score100, gradient100, factor100 = reference(x, y, theta, scenario, digits=100)
    np.testing.assert_allclose(score, score100, rtol=1e-14)
    np.testing.assert_allclose(gradient, gradient100, rtol=1e-14)
    np.testing.assert_allclose(factor, factor100, rtol=1e-14)
    for i in range(3):
        direction = np.eye(3)[i] * 1e-4
        plus = reference(x, y, theta + direction, scenario, digits=80)[0]
        minus = reference(x, y, theta - direction, scenario, digits=80)[0]
        np.testing.assert_allclose(
            (plus - minus) / 2e-4, gradient[i], rtol=1e-5, atol=1e-7
        )


def sequence_inputs(scenario):
    rng = np.random.default_rng(12)
    case = Case(scenario, 6, 4, "tail", 2)
    args = (
        np.linspace(0.05, 0.95, 6),
        rng.uniform(0.02, 0.98, (4, 2)),
        rng.normal(size=6),
        rng.normal(size=(4, 2)),
        np.log([1.0, 0.3, 1e-3]),
    )
    return case, args


@pytest.mark.parametrize("scenario", ["se_clustered", "matern32_uniform"])
def test_sequence_reference_converges_with_precision(scenario):
    case, args = sequence_inputs(scenario)
    value, grads = sequence(case, args, 80)
    v100, g100 = sequence(case, args, 100)
    assert value == v100
    for g, g1 in zip(grads, g100, strict=True):
        np.testing.assert_array_equal(g, g1)


@pytest.mark.parametrize("scenario", ["se_clustered", "matern32_uniform"])
def test_sequence_reference_gradients_match_finite_differences(scenario):
    case, args = sequence_inputs(scenario)
    _, grads = sequence(case, args, 80)
    rng = np.random.default_rng(12)
    h = 1e-5
    input_names = (
        "initial locations",
        "incoming locations",
        "initial values",
        "incoming values",
        "kernel parameters",
    )
    for i, a in enumerate(args):
        direction = rng.normal(size=a.shape)
        direction /= np.linalg.norm(direction)
        plus = list(args)
        minus = list(args)
        plus[i] = a + h * direction
        minus[i] = a - h * direction
        fd = (sequence(case, plus)[0] - sequence(case, minus)[0]) / (2 * h)
        derivative = np.vdot(grads[i], direction)
        np.testing.assert_allclose(
            derivative,
            fd,
            rtol=2e-5,
            atol=1e-6,
            err_msg=f"Sequence derivative with respect to {input_names[i]}",
        )


@pytest.mark.parametrize("scenario", ["se_clustered", "matern32_uniform"])
def test_independent_gp_references_agree(scenario):
    _, args = sequence_inputs(scenario)
    value, (_, _, parameter_gradient) = gp(args[0], args[2], args[4], scenario)
    expected_value, expected_gradient, _ = reference(
        args[0], args[2], args[4], scenario
    )
    assert value == expected_value
    np.testing.assert_array_equal(parameter_gradient, expected_gradient)
