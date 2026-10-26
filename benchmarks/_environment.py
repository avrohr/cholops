"""Record the code, software, hardware and compiler storage behind each result."""

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jaxlib
import numpy as np


def command_output(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def metadata(args):
    """Capture source hashes as well as Git state, including uncommitted edits."""
    root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for folder in (root / "src", root / "benchmarks"):
        for path in sorted(folder.rglob("*.py")):
            source_hashes[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for name in ("pyproject.toml", "uv.lock"):
        source_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "devices": [str(d) for d in jax.devices()],
        "device_kinds": [d.device_kind for d in jax.devices()],
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "cpu_model": command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if sys.platform == "darwin"
        else command_output(["lscpu"]),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm",
                "--format=csv,noheader",
            ]
        ),
        "git_revision": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "-C", str(root), "status", "--porcelain"]),
        "source_sha256": source_hashes,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("JAX_", "XLA_", "SLURM_", "OMP_", "OPENBLAS_", "MKL_"))
        },
        "timing_contract": (
            "Inputs stay on the device and are not donated. Timing includes dispatch "
            "and synchronized execution. Compilation is recorded separately. "
            "Implementation order is randomized within each round."
        ),
    }


def compiled_memory(executable):
    """Read compiler storage estimates in bytes, not observed device allocations."""
    memory = executable.memory_analysis()
    if memory is None:
        raise RuntimeError("compiler memory accounting is unavailable")
    return {
        key: getattr(memory, key)
        for key in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
            "generated_code_size_in_bytes",
        )
    }
