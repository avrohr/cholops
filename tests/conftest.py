"""Small checks by default; release coverage is an explicit opt-in."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        help="Include release numerical, large-matrix and benchmark tests",
    )
    parser.addoption(
        "--require-gpu",
        action="store_true",
        help="Fail instead of silently running release validation on CPU",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-gpu"):
        import jax

        if jax.default_backend() != "gpu":
            raise pytest.UsageError(
                "--require-gpu needs an active CUDA device (set JAX_PLATFORMS=cuda)"
            )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--full"):
        selected, deselected = [], []
        for item in items:
            (deselected if item.get_closest_marker("full") else selected).append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)


@pytest.fixture(autouse=True)
def release_gpu_requirement(request):
    if request.node.get_closest_marker("gpu"):
        import jax

        if jax.default_backend() != "gpu":
            pytest.skip("GPU-specific release check; run --full --require-gpu on CUDA")
