"""Write a benchmark plan and optionally run each case in a fresh process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CORE = (
    "append",
    "delete",
    "rank1_positive",
    "replace",
    "delete_append",
    "replace_last",
)


def cases(sizes, dtypes, repeats):
    """Yield operation curves and fixed workloads for comparisons between versions."""
    for dtype in dtypes:
        for n in sizes:
            operations = CORE + (("diag_update", "rank1") if n == 4096 else ())
            for repeat in range(repeats):
                for mode in ("forward", "full_vjp", "jvp"):
                    for operation in operations:
                        yield dict(
                            workload="operations",
                            operation=operation,
                            n=n,
                            dtype=dtype,
                            position="first",
                            mode=mode,
                            repeat=repeat,
                            steps=16,
                            implementations=["tensor", "recompute"]
                            + (["naive"] if operation == "replace" else []),
                        )
                    if n in (2048, 8192):
                        yield dict(
                            workload="sequences",
                            n=n,
                            dtype=dtype,
                            position="first",
                            mode=mode,
                            repeat=repeat,
                            steps=16,
                            implementations=["canonical", "managed", "recompute"],
                        )


def command(case, backend, samples, memory_budget_gib, output):
    """Run a case with this Python environment and a repeat-specific timing order."""
    args = [
        sys.executable,
        "-m",
        "benchmarks.run",
        "--backend",
        backend,
        "--samples",
        str(samples),
        "--memory-budget-gib",
        str(memory_budget_gib),
        "--order-seed",
        str(1701 + case["repeat"]),
        "--output",
        str(output),
    ]
    for key, value in case.items():
        args += ["--" + key.replace("_", "-")]
        args += [str(x) for x in value] if isinstance(value, list) else [str(value)]
    return args


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float64"),
        default=["float32", "float64"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--memory-budget-gib", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the new plan in fresh child processes. By default, only write it",
    )
    args = parser.parse_args()
    if (
        min(args.sizes) < 2
        or args.repeats < 1
        or args.samples < 3
        or args.memory_budget_gib <= 0
    ):
        parser.error(
            "require sizes >= 2, repeats >= 1, samples >= 3 and positive memory budget"
        )
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.dtypes)) != len(
        args.dtypes
    ):
        parser.error("sizes and dtypes must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = list(cases(args.sizes, args.dtypes, args.repeats))
    payload = {
        **vars(args),
        "output": str(args.output),
        "cases": plan,
        "schema_version": 1,
    }
    (args.output / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Planned {len(plan)} processes in {args.output / 'plan.json'}", flush=True)
    if not args.execute:
        return
    for i, case in enumerate(plan):
        output = args.output / f"case-{i:05d}"
        with (args.output / f"case-{i:05d}.log").open("w") as stream:
            result = subprocess.run(
                command(
                    case, args.backend, args.samples, args.memory_budget_gib, output
                ),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        record = {
            "case": i,
            "returncode": result.returncode,
            "status": "ok"
            if result.returncode == 0
            else "skipped_memory"
            if result.returncode == 77
            else "failed",
        }
        with (args.output / "execution.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if result.returncode not in (0, 77):
            raise SystemExit(f"Case {i} failed. Inspect its case log.")


if __name__ == "__main__":
    main()
