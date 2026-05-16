"""Aggregate parameter-sweep run directories into a single CSV.

Walks ``<runs-root>/<axis>-N/results.csv`` files (e.g. ``seed-0/``,
``seed-1/``, or ``k-10/``, ``k-30/``), injects the parameter value
as a column, and concatenates them. Downstream viz functions
auto-groupby on (dataset, method) so multi-seed → error bars; the
k-sweep plot uses the ``k`` column directly.

Usage::

    python -m benchmarks.scripts.aggregate_seeds \\
        --runs-root benchmarks/runs/full-seeds-v1/full-seeds \\
        --axis seed \\
        --out-csv benchmarks/runs/full-seeds-v1/results-multi-seed.csv

    python -m benchmarks.scripts.aggregate_seeds \\
        --runs-root benchmarks/runs/full-seeds-v1/k-sweep \\
        --axis k \\
        --out-csv benchmarks/runs/full-seeds-v1/results-k-sweep.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _value_from_dir(d: Path, axis: str) -> int | None:
    """Match ``<axis>-<int>`` or ``<axis>-<int>-<suffix>`` directory names.
    The optional trailing suffix (e.g. dataset name) is informational
    only — the results.csv itself carries the dataset column."""
    m = re.match(rf"^{re.escape(axis)}-(\d+)(?:-.+)?$", d.name)
    return int(m.group(1)) if m else None


def aggregate(runs_root: Path, axis: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        value = _value_from_dir(sub, axis)
        if value is None:
            continue
        csv = sub / "results.csv"
        if not csv.exists():
            print(f"  skip {sub.name}: no results.csv")
            continue
        df = pd.read_csv(csv)
        # Set the axis column (overwriting any orchestrator-injected
        # default — e.g. the orchestrator writes 'k' from --k, which
        # for the k-sweep matches the dir name).
        df[axis] = value
        rows.append(df)
        print(f"  {sub.name}: {len(df)} rows  ({axis}={value})")
    if not rows:
        raise FileNotFoundError(f"no {axis}-N/results.csv under {runs_root}")
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        required=True,
        type=Path,
        help="Directory containing <axis>-N subdirectories.",
    )
    parser.add_argument(
        "--axis", default="seed", help="Parameter axis encoded in subdir names (default: seed)"
    )
    parser.add_argument(
        "--out-csv", required=True, type=Path, help="Path to write the aggregated CSV."
    )
    args = parser.parse_args()

    df = aggregate(args.runs_root, args.axis)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(
        f"\nWrote {args.out_csv}  ({len(df)} rows, "
        f"{df[args.axis].nunique()} {args.axis} values, "
        f"{df['dataset'].nunique()} datasets, "
        f"{df['method'].nunique()} methods)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
