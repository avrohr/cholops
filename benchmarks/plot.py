"""Absolute timing and compiler-storage plots from a declared benchmark campaign."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from benchmarks.suite import CORE

NAMES = {
    "tensor": "cholops",
    "naive": "Naive replacement",
    "canonical": "Cholesky",
    "managed": "PermutedCholesky",
    "recompute": "JAX recomputation",
}
COLORS = {
    "tensor": "#177ca4",
    "naive": "#ae5d9b",
    "canonical": "#d88824",
    "managed": "#18866b",
    "recompute": "#4b5563",
}


def identity(metadata):
    """Select hardware and JAX settings that must agree across measured cases."""
    result = {key: metadata[key] for key in ("jax", "jaxlib", "device_kinds", "gpu")}
    cpu = metadata.get("cpu_model") or "unknown"
    # Linux lscpu also reports changing frequency and vulnerability status.
    model = [
        line.strip() for line in cpu.splitlines() if line.startswith("Model name:")
    ]
    result["cpu_model"] = model[0] if model else cpu
    affinity = metadata.get("cpu_affinity")
    result["cpu_threads"] = len(affinity) if affinity else metadata.get("cpu_count")
    return json.dumps(result, sort_keys=True)


def read_campaign(root, allow_partial=False):
    """Load planned cases and check completeness, timing samples and provenance."""
    plan = json.loads((root / "plan.json").read_text())
    records, coverage, identities, sources = [], [], set(), {}
    for i, case in enumerate(plan["cases"]):
        folder = root / f"case-{i:05d}"
        path = folder / "metadata.json"
        metadata = json.loads(path.read_text()) if path.exists() else {}
        status = metadata.get("status", "incomplete")
        if metadata.get("completed") and status == "ok":
            measured = [
                json.loads(line)
                for line in (folder / "results.jsonl").read_text().splitlines()
            ]
            if len(measured) != metadata["phase_count"]:
                raise ValueError(f"Incomplete results: {folder}")
            actual = metadata["arguments"]
            if any(actual.get(k) != value for k, value in case.items()):
                raise ValueError(f"Results do not match planned case: {folder}")
            identities.add(identity(metadata))
            sources[str(i)] = json.dumps(metadata["source_sha256"], sort_keys=True)
            for row in measured:
                if row["status"] != "ok":
                    raise ValueError(f"Incomplete measurement: {folder}")
                samples = row["samples_s"]
                if len(samples) != plan["samples"] or not all(
                    np.isfinite(x) and x > 0 for x in samples
                ):
                    raise ValueError(f"Invalid timing samples: {folder}")
                row["median_s"] = float(np.median(samples))
                row["case"] = i
                records.append(row)
        coverage.append(
            {
                "case": i,
                **case,
                "status": status,
                "reason": metadata.get("reason", ""),
                "skipped_implementations": json.dumps(
                    metadata.get("skipped_implementations", [])
                ),
            }
        )
    if len(identities) > 1:
        raise ValueError(
            "Hardware or software differs. Report these campaigns separately"
        )
    if len(set(sources.values())) > 1:
        # A partial rerun may use new code for some operations. The audit maps
        # each case to the source hashes that produced its measurements.
        path = root / "source-audit.json"
        if not path.exists():
            raise ValueError("Sources differ without a per-case source audit")
        audit = json.loads(path.read_text())
        variants = [json.dumps(v, sort_keys=True) for v in audit["source_variants"]]
        for case, source in sources.items():
            variant = audit["case_variants"][case]
            if source != variants[variant]:
                raise ValueError(f"Source does not match audited case {case}")
    if not allow_partial and any(
        c["status"] not in ("ok", "skipped_memory") for c in coverage
    ):
        raise ValueError(
            "Campaign incomplete. Use --allow-partial to preview the available results"
        )
    if not records:
        raise ValueError("No valid measurements")
    return plan, records, coverage


def summarize(records):
    """Take the median of process medians and retain their range for plot shading."""
    groups = defaultdict(list)
    fields = (
        "workload",
        "operation",
        "steps",
        "n",
        "dtype",
        "position",
        "phase_kind",
        "implementation",
    )
    for row in records:
        groups[tuple(row[k] for k in fields)].append(row)
    output = []
    for key, rows in groups.items():
        if len({r["repeat"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate repeat for {key}")
        values = [r["median_s"] * 1000 for r in rows]
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "repeats": len(rows),
                "median_ms": float(np.median(values)),
                "min_ms": min(values),
                "max_ms": max(values),
                "compile_ms": float(np.median([1000 * r["compile_s"] for r in rows])),
                "cold_one_call_ms": float(
                    np.median([1000 * r["cold_one_call_s"] for r in rows])
                ),
                "temporary_mib": max(r["memory"]["temp_size_in_bytes"] for r in rows)
                / 2**20,
                "working_set_mib": max(r["compiled_working_set_bytes"] for r in rows)
                / 2**20,
            }
        )
    return output


def speedups(records):
    """Compute baseline/cholops ratios per process, then summarize across repeats."""
    paired = defaultdict(dict)
    fields = ("workload", "operation", "steps", "n", "dtype", "position", "phase_kind")
    for row in records:
        paired[(*[row[k] for k in fields], row["repeat"])][row["implementation"]] = row
    groups = defaultdict(list)
    for key, implementations in paired.items():
        if "tensor" not in implementations:
            continue
        for baseline in ("recompute", "naive"):
            if baseline in implementations:
                groups[(*key[:-1], baseline)].append(
                    implementations[baseline]["median_s"]
                    / implementations["tensor"]["median_s"]
                )
    return [
        {
            **dict(zip((*fields, "baseline"), key, strict=True)),
            "repeats": len(values),
            "median_speedup": float(np.median(values)),
            "min_speedup": min(values),
            "max_speedup": max(values),
        }
        for key, values in groups.items()
    ]


def axes_style(ax, sizes, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(sizes)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:,.6f}".rstrip("0").rstrip("."))
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_xlabel("Matrix size")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, which="major")
    ax.spines[["top", "right"]].set_visible(False)


def panel(ax, rows, sizes, metric="median_ms"):
    for name in NAMES:
        selected = sorted(
            (r for r in rows if r["implementation"] == name), key=lambda r: r["n"]
        )
        if not selected:
            continue
        # Missing sizes create gaps in the plotted line.
        by_n = {r["n"]: r for r in selected}
        x = sorted(set(sizes))
        y = [by_n[n][metric] if n in by_n else np.nan for n in x]
        ax.plot(x, y, "o-", label=NAMES[name], color=COLORS[name], markersize=4)
        if metric == "median_ms":
            ax.fill_between(
                x,
                [by_n[n]["min_ms"] if n in by_n else np.nan for n in x],
                [by_n[n]["max_ms"] if n in by_n else np.nan for n in x],
                color=COLORS[name],
                alpha=0.12,
            )
    axes_style(
        ax,
        sizes,
        "Time (ms)" if metric == "median_ms" else "Compiled working set (MiB)",
    )
    if not rows:
        ax.text(0.5, 0.5, "Not measured", transform=ax.transAxes, ha="center")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def render(plan, summary, coverage, output):
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary.csv", summary)
    # Heterogeneous case fields get a fixed union of CSV columns.
    fields = sorted({k for row in coverage for k in row})
    write_csv(
        output / "coverage.csv", [{k: row.get(k) for k in fields} for row in coverage]
    )
    for metric, filename in (
        ("median_ms", "replacement"),
        ("working_set_mib", "memory"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
        for column, dtype in enumerate(("float32", "float64")):
            for row, mode in enumerate(("forward", "full_vjp")):
                subset = [
                    r
                    for r in summary
                    if r["workload"] == "operations"
                    and r["operation"] == "replace"
                    and r["position"] == "first"
                    and r["dtype"] == dtype
                    and r["phase_kind"] == mode
                ]
                panel(axes[row, column], subset, plan["sizes"], metric)
                label = "forward" if mode == "forward" else "value + all-input gradient"
                axes[row, column].set_title(f"{dtype} · {label}")
        handles, labels = [], []
        for ax in axes.flat:
            h, axis_labels = ax.get_legend_handles_labels()
            for handle, label in zip(h, axis_labels, strict=True):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncols=3,
            frameon=False,
        )
        fig.suptitle("Single index-preserving replacement · first index")
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        for extension in ("png", "svg"):
            path = output / f"{filename}.{extension}"
            fig.savefig(path, dpi=180)
            if extension == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
        plt.close(fig)
    operations = sorted({r["operation"] for r in summary if r["operation"] in CORE})
    modes = [
        mode
        for mode in ("forward", "full_vjp", "jvp")
        if any(r["phase_kind"] == mode for r in summary)
    ]
    mode_labels = {
        "forward": "Forward",
        "full_vjp": "Value + gradient",
        "jvp": "Directional derivative",
    }
    # One row per operation keeps each mode aligned across the entire figure.
    for dtype in plan["dtypes"]:
        if not operations:
            continue
        with plt.rc_context({"font.size": 10}):
            fig, axes = plt.subplots(
                len(operations),
                len(modes),
                figsize=(4.6 * len(modes), 2.5 * len(operations) + 1.2),
                sharex=True,
                sharey="row",
                squeeze=False,
            )
            handles, labels = [], []
            for row, operation in enumerate(operations):
                for column, mode in enumerate(modes):
                    ax = axes[row, column]
                    subset = [
                        r
                        for r in summary
                        if r["workload"] == "operations"
                        and r["operation"] == operation
                        and r["dtype"] == dtype
                        and r["position"] == "first"
                        and r["phase_kind"] == mode
                    ]
                    panel(ax, subset, plan["sizes"])
                    ax.set_title(f"{operation} · {mode_labels[mode]}", loc="left")
                    ax.set_xlabel("Matrix size" if row == len(operations) - 1 else "")
                    ax.set_ylabel("Time (ms)" if column == 0 else "")
                    h, axis_labels = ax.get_legend_handles_labels()
                    for handle, label in zip(h, axis_labels, strict=True):
                        if label not in labels:
                            handles.append(handle)
                            labels.append(label)
            fig.suptitle(f"Operation timings · {dtype}", fontsize=15, y=0.985)
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.963),
                ncols=min(len(labels), 3),
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.6, w_pad=1.8)
            fig.savefig(output / f"operations-{dtype}.png", dpi=180)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    plan, records, coverage = read_campaign(args.root, args.allow_partial)
    render(plan, summarize(records), coverage, args.output)
    write_csv(args.output / "speedups.csv", speedups(records))


if __name__ == "__main__":
    main()
