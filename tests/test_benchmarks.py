"""Check benchmark comparisons, coverage and reporting."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from benchmarks._support import accuracy_record
from benchmarks.suite import CORE, cases
from benchmarks.workloads import MODES, OPERATIONS, operations, sequences


def test_relative_error_diagnostic_preserves_small_scales():
    record = accuracy_record(jnp.asarray([2e-8]), jnp.asarray([1e-8]))
    assert record["max_leaf_relative_error"] == pytest.approx(1.0)
    assert record["max_leaf_normalized_error"] < 1e-7


def test_campaign_coverage_and_unique_cases():
    campaign = list(
        cases([512, 1024, 2048, 4096, 8192, 16384], ["float32", "float64"], 3)
    )
    curves = [c for c in campaign if c.get("operation") in CORE]
    signed = [c for c in campaign if c.get("operation") in ("diag_update", "rank1")]
    sequences = [c for c in campaign if c["workload"] == "sequences"]
    assert len(campaign) == 720
    assert len(curves) == 648
    assert len(signed) == len(sequences) == 36
    assert {c["n"] for c in signed} == {4096}
    assert {c["n"] for c in sequences} == {2048, 8192}
    assert all(c["steps"] == 16 for c in sequences)
    assert all(c["position"] == "first" for c in campaign)
    for group in (curves, signed, sequences):
        assert {c["mode"] for c in group} == set(MODES)
        assert {c["dtype"] for c in group} == {"float32", "float64"}
        assert {c["repeat"] for c in group} == {0, 1, 2}
    for case in campaign:
        expected = (
            ["canonical", "managed", "recompute"]
            if case["workload"] == "sequences"
            else ["tensor", "recompute"]
            + (["naive"] if case["operation"] == "replace" else [])
        )
        assert case["implementations"] == expected
    keys = [json.dumps(c, sort_keys=True) for c in campaign]
    assert len(set(keys)) == len(keys)


def test_campaign_size_selection():
    campaign = list(cases([512], ["float32"], 1))
    assert len(campaign) == 18
    assert {c["operation"] for c in campaign} == set(CORE)
    assert all(c["n"] == 512 and c["dtype"] == "float32" for c in campaign)


@pytest.mark.full
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", MODES)
def test_operation_workload(operation, mode):
    dtype = "float64" if jax.config.x64_enabled else "float32"
    phases = operations(7, "middle", dtype, operation=operation, modes=(mode,))
    check_phases(phases)


@pytest.mark.full
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("position", ["first", "penultimate"])
def test_sequence_workload(mode, position):
    dtype = "float64" if jax.config.x64_enabled else "float32"
    check_phases(sequences(7, position, dtype, modes=(mode,)))


def check_phases(phases):
    assert phases
    for name, phase in phases.items():
        actual = jax.jit(phase["function"])(*phase["args"])
        accuracy = phase["validate"](actual)
        assert accuracy["finite"], name
        # Small, well-conditioned fixtures: tolerances belong to numerical tests,
        # not the performance runner or its workload definitions.
        sequence = phase["comparison_group"].startswith("sequence_")
        tolerance = (
            (2e-9 if sequence else 5e-10)
            if jax.config.x64_enabled
            else (2e-3 if sequence else 7e-4)
        )
        assert accuracy["max_leaf_relative_error"] < tolerance, (
            name,
            accuracy,
        )


@pytest.mark.full
@pytest.mark.parametrize("position", ["first", "middle", "penultimate", "last"])
@pytest.mark.parametrize("mode", MODES)
def test_naive_replacement_comparator(position, mode):
    dtype = "float64" if jax.config.x64_enabled else "float32"
    check_phases(
        operations(
            7,
            position,
            dtype,
            operation="replace",
            modes=(mode,),
            implementations=("tensor", "naive", "recompute"),
        )
    )


@pytest.mark.full
@pytest.mark.parametrize("mode", MODES)
def test_block_append_comparators(mode):
    dtype = "float64" if jax.config.x64_enabled else "float32"
    phases = operations(
        7,
        "first",
        dtype,
        operation="append_block",
        modes=(mode,),
        implementations=("tensor", "sequential", "recompute"),
    )
    assert {phase["implementation"] for phase in phases.values()} == {
        "tensor",
        "sequential",
        "recompute",
    }
    check_phases(phases)


def test_block_append_runner_accepts_sequential_baseline(monkeypatch, tmp_path):
    from benchmarks.run import arguments

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.run",
            "--backend",
            "cpu",
            "--dtype",
            "float32",
            "--n",
            "8",
            "--position",
            "first",
            "--workload",
            "operations",
            "--operation",
            "append_block",
            "--implementations",
            "tensor",
            "sequential",
            "recompute",
            "--output",
            str(tmp_path),
        ],
    )
    args = arguments()
    assert args.operation == "append_block"
    assert args.implementations == ["tensor", "sequential", "recompute"]


@pytest.fixture
def benchmark_output(tmp_path):
    # Exercise the real runner and plot reader, with a deliberately poor
    # comparison diagnostic. Numerical assertions are tested separately above.
    output = tmp_path / "case-00000"
    script = """
from benchmarks import workloads
from benchmarks.run import main
workloads.accuracy_record = lambda actual, expected: {
    "finite": False, "max_leaf_relative_error": None,
    "max_leaf_normalized_error": None, "max_absolute_error": None,
}
main()
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--backend",
            "cpu",
            "--dtype",
            "float32",
            "--n",
            "2",
            "--position",
            "first",
            "--workload",
            "operations",
            "--operation",
            "append",
            "--mode",
            "forward",
            "--implementations",
            "tensor",
            "recompute",
            "--samples",
            "3",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "plan.json").write_text(json.dumps({"cases": [{}], "samples": 3}))
    return tmp_path


@pytest.mark.full
def test_timings_are_not_filtered_by_numerical_diagnostics(benchmark_output):
    from benchmarks.plot import read_campaign

    _, rows, coverage = read_campaign(benchmark_output)
    assert coverage[0]["status"] == "ok"
    assert {row["implementation"] for row in rows} == {"tensor", "recompute"}
    assert all(not row["accuracy"]["finite"] for row in rows)
    assert all(len(row["samples_s"]) == 3 for row in rows)


@pytest.mark.full
def test_mixed_sources_require_an_audit_for_each_case(benchmark_output):
    from benchmarks.plot import read_campaign

    # Accepting a known source version is insufficient: it must match this case.
    tmp_path = benchmark_output
    output = tmp_path / "case-00000"
    other = tmp_path / "case-00001"
    shutil.copytree(output, other)
    metadata = json.loads((other / "metadata.json").read_text())
    original = metadata["source_sha256"].copy()
    metadata["source_sha256"]["benchmarks/workloads.py"] = "different-source"
    (other / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "plan.json").write_text(json.dumps({"cases": [{}, {}], "samples": 3}))
    with pytest.raises(ValueError, match="per-case source audit"):
        read_campaign(tmp_path)
    audit = {
        "source_variants": [original, metadata["source_sha256"]],
        "case_variants": {"0": 0, "1": 1},
    }
    audit_path = tmp_path / "source-audit.json"
    audit_path.write_text(json.dumps(audit))
    _, rows, _ = read_campaign(tmp_path)
    assert len(rows) == 4
    audit["case_variants"]["1"] = 0
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Source does not match audited case 1"):
        read_campaign(tmp_path)
