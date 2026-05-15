"""Driver script — runs the scIB benchmark on both datasets and
emits a results CSV + a markdown summary table.

CPU pilot (default — fast, subsampled)::

    python benchmarks/run_benchmark.py

Full datasets (recommended on a GPU pod)::

    python benchmarks/run_benchmark.py --full

Outputs:

- ``benchmarks/results.csv``: one row per (dataset, method)
- ``benchmarks/results.md``: a tidy markdown table per dataset and a
  cross-dataset average, suitable for dropping into the docs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.scib_benchmark import (
    METHODS,
    results_to_dataframe,
    run_dataset,
)

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on full datasets (recommended on GPU). Default subsamples for CPU.",
    )
    parser.add_argument(
        "--cells-per-cohort",
        type=int,
        default=50,
        help="Stratified subsample size per (batch, label) cohort. Ignored with --full.",
    )
    parser.add_argument("--k", type=int, default=30, help="Latent dim k for all methods.")
    parser.add_argument(
        "--no-lisi",
        action="store_true",
        help="Skip iLISI/cLISI (off by default on Apple Silicon — scib's compiled "
        "binary is x86_64-only). Enable when running on Linux x86_64 (e.g., RunPod).",
    )
    parser.add_argument(
        "--no-hvg",
        action="store_true",
        help="Skip HVG=2000 batch-aware filter (scIB-canonical preprocessing). "
        "Off by default. Without HVG, methods operate on full ~20k genes per "
        "dataset and runtimes blow up ~10x.",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=2000,
        help="Number of HVGs to select if --no-hvg is NOT set. Default 2000 per scIB paper.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run (default: all). Choices: "
        + ", ".join(METHODS),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pancreas", "immune"],
        help="Datasets to benchmark (default: pancreas immune).",
    )
    args = parser.parse_args()

    cells_per_cohort = None if args.full else args.cells_per_cohort

    all_results = {}
    for dataset in args.datasets:
        all_results[dataset] = run_dataset(
            dataset,
            methods=args.methods,
            cells_per_cohort=cells_per_cohort,
            k=args.k,
            lisi=not args.no_lisi,
            hvg=not args.no_hvg,
            n_hvg=args.n_hvg,
        )

    df = results_to_dataframe(all_results)
    csv_path = HERE / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Write a markdown summary suitable for docs.
    def _fmt_time(s: float) -> str:
        if s < 0.5:
            return f"{s * 1000:.0f}ms"
        if s < 60:
            return f"{s:.1f}s"
        return f"{s / 60:.1f}m"

    def _fmt_mem(mb: float | None) -> str:
        if mb is None or mb <= 0:
            return "—"
        if mb < 1024:
            return f"{mb:.0f} MB"
        return f"{mb / 1024:.1f} GB"

    md_lines: list[str] = []
    for dataset, results in all_results.items():
        md_lines.append(f"### {dataset}\n")
        md_lines.append(
            "| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            t = r.timing
            fit = _fmt_time(t.fit_seconds)
            inf = _fmt_time(t.infer_seconds) if t.infer_seconds else "—"
            mt = _fmt_time(t.metric_seconds)
            rss = _fmt_mem(t.peak_rss_mb)
            gpu = _fmt_mem(t.gpu_peak_mb)
            if r.error:
                md_lines.append(
                    f"| {r.name} | {fit} | {inf} | {mt} | {rss} | {gpu} | "
                    f"`{r.error[:40]}` | — | — |"
                )
                continue
            md_lines.append(
                f"| {r.name} | {fit} | {inf} | {mt} | {rss} | {gpu} | "
                f"{r.metrics['_bio']:+.3f} | {r.metrics['_batch']:+.3f} | "
                f"**{r.metrics['_composite']:+.3f}** |"
            )
        md_lines.append("")

    # Cross-dataset composite average (per method)
    md_lines.append("### Cross-dataset composite average\n")
    md_lines.append("| method | mean composite ↑ | mean fit | mean GPU peak |")
    md_lines.append("|---|---:|---:|---:|")
    by_method: dict[str, dict[str, list[float]]] = {}
    for results in all_results.values():
        for r in results:
            if "_composite" not in r.metrics:
                continue
            d = by_method.setdefault(r.name, {"comp": [], "fit": [], "gpu": []})
            d["comp"].append(r.metrics["_composite"])
            d["fit"].append(r.timing.fit_seconds)
            if r.timing.gpu_peak_mb:
                d["gpu"].append(r.timing.gpu_peak_mb)
    for name, vals in sorted(by_method.items(), key=lambda kv: -float(np.mean(kv[1]["comp"]))):
        mean_comp = float(np.mean(vals["comp"]))
        mean_fit = _fmt_time(float(np.mean(vals["fit"])))
        mean_gpu = _fmt_mem(float(np.mean(vals["gpu"]))) if vals["gpu"] else "—"
        md_lines.append(f"| {name} | **{mean_comp:+.3f}** | {mean_fit} | {mean_gpu} |")

    md_path = HERE / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    json_path = HERE / "results.json"
    json_payload = {
        dataset: [
            {
                "method": r.name,
                "fit_seconds": r.timing.fit_seconds,
                "infer_seconds": r.timing.infer_seconds,
                "metric_seconds": r.timing.metric_seconds,
                "total_seconds": r.timing.total_seconds(),
                "peak_rss_mb": r.timing.peak_rss_mb,
                "gpu_peak_mb": r.timing.gpu_peak_mb,
                "error": r.error,
                **r.metrics,
            }
            for r in results
        ]
        for dataset, results in all_results.items()
    }
    json_path.write_text(json.dumps(json_payload, indent=2))
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
