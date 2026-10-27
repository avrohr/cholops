"""Run one benchmark case and save timings, diagnostics and environment details."""

import argparse
import gc
import json
import os
import time
from pathlib import Path


def native(value):
    """Convert scalar arrays and paths to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(x) for x in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--position", choices=["first", "middle", "penultimate", "last"], required=True
    )
    parser.add_argument(
        "--workload",
        choices=["operations", "sequences"],
        default="sequences",
    )
    parser.add_argument(
        "--operation",
        choices=[
            "append",
            "append_block",
            "delete",
            "replace",
            "delete_append",
            "replace_last",
            "diag_update",
            "rank1",
            "rank1_positive",
        ],
    )
    parser.add_argument("--mode", choices=["forward", "full_vjp", "jvp"])
    parser.add_argument("--implementations", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--order-seed", type=int, default=1701)
    parser.add_argument("--memory-budget-gib", type=float, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 2 or args.samples < 3 or args.memory_budget_gib <= 0 or args.repeat < 0:
        parser.error(
            "require n >= 2, samples >= 3, positive budget and nonnegative repeat"
        )
    if args.steps < 4 or args.steps % 4:
        parser.error("steps must be a positive multiple of four")
    allowed = (
        {"tensor", "recompute"}
        if args.workload == "operations"
        else {"canonical", "managed", "recompute"}
    )
    if args.operation == "replace":
        allowed.add("naive")
    if args.operation == "append_block":
        allowed.add("sequential")
    if args.implementations is not None and (
        not set(args.implementations) <= allowed
        or "recompute" not in args.implementations
        or len(set(args.implementations)) != len(args.implementations)
    ):
        parser.error(
            "implementations must be unique and valid for the workload. "
            "Include recompute"
        )
    if args.operation and args.workload != "operations":
        parser.error("--operation requires --workload operations")
    return args


def main():
    args = arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Configure JAX before importing it so each process uses the requested device.
    os.environ["JAX_ENABLE_X64"] = str(int(args.dtype == "float64"))
    os.environ["JAX_PLATFORMS"] = "cuda" if args.backend == "gpu" else "cpu"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    dump = output / "xla"
    flags = (
        f"--xla_dump_to={dump} --xla_dump_hlo_as_text "
        "--xla_dump_hlo_module_re=jit_rounds_.*"
    )
    if args.dump_hlo:
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flags}".strip()
    import jax
    import numpy as np

    from benchmarks import workloads
    from benchmarks._environment import compiled_memory, metadata

    if jax.default_backend() != args.backend:
        raise RuntimeError("requested backend is not active")
    jax.config.update("jax_enable_compilation_cache", False)
    args.output = str(output)
    meta = metadata(args)
    meta.update(
        completed=False,
        timing_contract=(
            "Compile and record numerical diagnostics outside timing. "
            "Within each comparison group, "
            "two shuffled warmup rounds precede independently shuffled matched "
            "timing rounds. Every call synchronizes its entire output. "
            "Arguments stay on the device and are not donated. The schedule is saved."
        ),
        accuracy_policy="diagnostic_only",
        profiled_timings=False,
        donation=False,
        warmup_rounds=2,
        schedules={},
    )
    meta_path = output / "metadata.json"

    def save_meta():
        meta_path.write_text(json.dumps(native(meta), indent=2) + "\n")

    def event(stage, **fields):
        record = native({"stage": stage, **fields})
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    save_meta()
    budget = args.memory_budget_gib * 2**30

    def memory_skip(reason):
        meta.update(status="skipped_memory", reason=reason)
        save_meta()
        event("skipped_memory", reason=reason)
        raise SystemExit(77)

    # Conservative preparation allowance, including dense reference derivatives.
    # The sequence adjoint stores block-boundary states even with checkpointing.
    matrices = max(16, 12 + args.steps // 4 + 4) if args.workload == "sequences" else 16
    if args.n**2 * np.dtype(args.dtype).itemsize * matrices + 2**30 > budget:
        memory_skip("preparation estimate exceeds memory budget")
    start = time.perf_counter()
    options = {"operation": args.operation, "steps": args.steps}
    if args.mode:
        options["modes"] = (args.mode,)
    if args.implementations:
        options["implementations"] = tuple(args.implementations)
    phases = getattr(workloads, args.workload)(
        args.n, args.position, np.dtype(args.dtype), seed=args.seed, **options
    )
    # Compare implementations of the same computation within each timing group.
    groups = {}
    for name, phase in phases.items():
        required = {
            "function",
            "args",
            "validate",
            "comparison_group",
            "implementation",
        }
        if not required <= phase.keys():
            raise ValueError(f"phase {name} missing {required - phase.keys()}")
        groups.setdefault(phase["comparison_group"], []).append(name)
    if not groups:
        raise ValueError("factory returned no phases")
    for group, names in groups.items():
        implementations = [phases[n]["implementation"] for n in names]
        if len(set(implementations)) != len(names):
            raise ValueError(f"duplicate implementation in {group}")
    # Shared fixture arrays count once toward the preparation memory budget.
    leaves = {
        id(x): x
        for p in phases.values()
        for x in jax.tree.leaves((p["args"], p.get("expected", ())))
        if hasattr(x, "nbytes")
    }
    fixture_bytes = sum(x.nbytes for x in leaves.values())
    jax.block_until_ready(tuple(leaves.values()))
    if fixture_bytes + 2**30 > budget:
        memory_skip("prepared fixtures exceed memory budget")
    meta.update(
        prepare_s=time.perf_counter() - start,
        prepared_argument_bytes=fixture_bytes,
        phases=list(phases),
        comparison_groups=groups,
    )
    save_meta()
    rng = np.random.default_rng(args.order_seed)
    rows = []
    for group, names in groups.items():
        compiled_phases = {}
        records = {}
        compile_order = list(rng.permutation(names))
        schedule = {
            "compile_order": compile_order,
            "warmup_rounds": [],
            "timed_rounds": [],
        }
        meta["schedules"][group] = schedule
        save_meta()
        # Compile and record the first call separately from warm timings.
        for name in compile_order:
            phase = phases[name]
            event("compile_start", phase=name, comparison_group=group)

            def wrapped(*inputs, _function=phase["function"]):
                return _function(*inputs)

            wrapped.__name__ = f"rounds_{name}"
            start = time.perf_counter()
            lowered = jax.jit(wrapped).lower(*phase["args"])
            lower_s = time.perf_counter() - start
            start = time.perf_counter()
            compiled = lowered.compile()
            compile_s = time.perf_counter() - start
            memory = compiled_memory(compiled)
            guarded = (
                fixture_bytes
                + memory["temp_size_in_bytes"]
                + memory["output_size_in_bytes"]
                + 2**30
            )
            row = {
                "phase": name,
                "phase_kind": phase.get("phase_kind"),
                "comparison_group": group,
                "implementation": phase["implementation"],
                "description": phase.get("description"),
                "workload": args.workload,
                "operation": args.operation,
                "steps": args.steps if args.workload == "sequences" else None,
                "n": args.n,
                "position": args.position,
                "dtype": args.dtype,
                "repeat": args.repeat,
                "order_seed": args.order_seed,
                "capture": "none",
                "lower_s": lower_s,
                "compile_s": compile_s,
                "memory": memory,
                "guarded_working_set_bytes": guarded,
            }
            row["compiled_working_set_bytes"] = (
                sum(
                    memory[k]
                    for k in (
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "temp_size_in_bytes",
                    )
                )
                - memory["alias_size_in_bytes"]
            )
            if args.dump_hlo:
                (output / f"{name}.hlo.txt").write_text(compiled.as_text())
            if guarded > budget:
                event("memory_guard_failed", **row)
                memory_skip(f"{name} exceeds compiled memory budget")
            start = time.perf_counter()
            actual = compiled(*phase["args"])
            jax.block_until_ready(actual)
            row["first_call_s"] = time.perf_counter() - start
            # Diagnostics run after synchronization and outside the timed call.
            accuracy = native(phase["validate"](actual))
            row["accuracy"] = accuracy
            del actual, lowered
            compiled_phases[name] = compiled
            records[name] = row
            event("phase_ready", phase=name, memory=memory, accuracy=accuracy)
        # Every implementation gets two warmups before any timed rounds begin.
        for _ in range(2):
            order = list(rng.permutation(names))
            schedule["warmup_rounds"].append(order)
            for name in order:
                jax.block_until_ready(compiled_phases[name](*phases[name]["args"]))
        # Synchronize the whole output to include execution, not just dispatch.
        samples = {name: [] for name in names}
        for _ in range(args.samples):
            order = list(rng.permutation(names))
            schedule["timed_rounds"].append(order)
            for name in order:
                start = time.perf_counter()
                jax.block_until_ready(compiled_phases[name](*phases[name]["args"]))
                samples[name].append(time.perf_counter() - start)
        for name in names:
            row = records[name]
            row.update(
                samples_s=samples[name],
                median_s=float(np.median(samples[name])),
                p90_s=float(np.percentile(samples[name], 90)),
                cold_one_call_s=row["lower_s"] + row["compile_s"] + row["first_call_s"],
                status="ok",
            )
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(native(row)) + "\n")
            rows.append(row)
        save_meta()
        event("group_complete", comparison_group=group, phases=len(names))
        del compiled, compiled_phases, records
        gc.collect()
    meta.update(
        completed=True,
        status="ok",
        phase_count=len(rows),
        xla_dump_files=sorted(
            str(p.relative_to(output)) for p in dump.glob("*") if p.is_file()
        ),
    )
    save_meta()
    event("complete", phases=len(rows))


if __name__ == "__main__":
    main()
