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
    METHODS, composite_score, results_to_dataframe, run_dataset,
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
        )

    df = results_to_dataframe(all_results)
    csv_path = HERE / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Write a markdown summary suitable for docs.
    md_lines: list[str] = []
    for dataset, results in all_results.items():
        md_lines.append(f"### {dataset}\n")
        md_lines.append("| method | seconds | bio ↑ | batch ↑ | composite ↑ |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for r in results:
            if "_composite" not in r.metrics:
                md_lines.append(f"| {r.name} | {r.seconds:.1f} | — | — | — |")
                continue
            md_lines.append(
                f"| {r.name} | {r.seconds:.1f} | "
                f"{r.metrics['_bio']:+.3f} | {r.metrics['_batch']:+.3f} | "
                f"**{r.metrics['_composite']:+.3f}** |"
            )
        md_lines.append("")
    # Cross-dataset composite average (per method)
    md_lines.append("### Cross-dataset composite average\n")
    md_lines.append("| method | mean composite ↑ |")
    md_lines.append("|---|---:|")
    by_method = {}
    for results in all_results.values():
        for r in results:
            if "_composite" in r.metrics:
                by_method.setdefault(r.name, []).append(r.metrics["_composite"])
    for name, vals in sorted(by_method.items(), key=lambda kv: -float(np.mean(kv[1]))):
        md_lines.append(f"| {name} | **{float(np.mean(vals)):+.3f}** |")

    md_path = HERE / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    json_path = HERE / "results.json"
    json_payload = {
        dataset: [
            {"method": r.name, "seconds": r.seconds, **r.metrics}
            for r in results
        ]
        for dataset, results in all_results.items()
    }
    json_path.write_text(json.dumps(json_payload, indent=2))
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
