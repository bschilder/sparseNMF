"""scIB benchmark orchestrator — subprocess-isolated.

Each method runs in its own Python subprocess so framework state
(CUDA contexts, JAX devices, RNG, lightning trainer globals)
cannot leak between methods. The metrics step also runs in its own
subprocess, after all embeddings are on disk.

CLI examples::

    # Default: 5 methods × 2 datasets, subsampled, YosefLab metrics.
    python -m benchmarks.run_benchmark

    # Pin to a specific run directory + primary metrics impl.
    python -m benchmarks.run_benchmark \\
        --out-dir benchmarks/runs/2026-05-15 \\
        --metrics-impl scib_original

    # Just one method, full data.
    python -m benchmarks.run_benchmark --methods sparseNMF --full --datasets immune

Output layout::

    <out-dir>/
      <dataset>/
        <method>/
          X_emb.npz
          timing.json
          metrics_yosef.json     # or metrics_original.json
          error.txt              # if the embed step failed
      results.csv
      results.md
      results.json
      figures/*.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Each entry: (display name, module path under benchmarks.methods).
METHOD_MODULES: dict[str, str] = {
    "PCA": "benchmarks.methods.pca",
    "NMF": "benchmarks.methods.nmf",
    "sparseNMF": "benchmarks.methods.sparse_nmf",
    "sparseNMF+nonzero": "benchmarks.methods.sparse_nmf_nonzero",
    "sparseNMF_supervised": "benchmarks.methods.sparse_nmf_batch",
    "Harmony": "benchmarks.methods.harmony",
    "scVI": "benchmarks.methods.scvi",
}

DEFAULT_METHODS = ["PCA", "NMF", "sparseNMF", "sparseNMF_supervised", "Harmony", "scVI"]

METRICS_MODULES = {
    "scib_yosef": "benchmarks.metrics.scib_yosef",
    "scib_original": "benchmarks.metrics.scib_original",
}


def run_method(method: str, dataset: str, out_dir: Path, args) -> int:
    """Spawn a subprocess that embeds ``method`` on ``dataset``."""
    cmd = [
        sys.executable,
        "-m",
        METHOD_MODULES[method],
        "--dataset",
        dataset,
        "--out-dir",
        str(out_dir),
        "--method-name",
        method,
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
    ]
    if args.cells_per_cohort is not None:
        cmd += ["--cells-per-cohort", str(args.cells_per_cohort)]
    if args.no_hvg:
        cmd += ["--no-hvg"]
    cmd += ["--n-hvg", str(args.n_hvg)]
    print(f"\n>>> {method} / {dataset}: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=REPO)


def run_metrics(impl: str, dataset: str, methods: list[str], out_dir: Path, args) -> int:
    """Spawn a subprocess that scores all methods for ``dataset``."""
    cmd = [
        sys.executable,
        "-m",
        METRICS_MODULES[impl],
        "--dataset",
        dataset,
        "--out-dir",
        str(out_dir),
        "--methods",
        *methods,
        "--seed",
        str(args.seed),
    ]
    if args.cells_per_cohort is not None:
        cmd += ["--cells-per-cohort", str(args.cells_per_cohort)]
    if args.no_hvg:
        cmd += ["--no-hvg"]
    cmd += ["--n-hvg", str(args.n_hvg)]
    if args.no_lisi:
        cmd += ["--no-lisi"]
    print(f"\n>>> metrics({impl}) / {dataset}: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=REPO)


# ── Aggregation: read per-method artifacts → DataFrame ──────────────


def _load_method_row(
    out_dir: Path, dataset: str, method: str, impl: str, k: int | None = None
) -> dict | None:
    """Return one row dict (matching the legacy CSV schema) or None
    if the method has nothing to report (no timing, no metrics, no
    error file)."""
    method_dir = out_dir / dataset / method
    timing_path = method_dir / "timing.json"
    metrics_name = "metrics_original.json" if impl == "scib_original" else "metrics_yosef.json"
    metrics_path = method_dir / metrics_name
    error_path = method_dir / "error.txt"

    # Nothing was ever written for this method — don't emit a phantom row.
    if not (timing_path.exists() or metrics_path.exists() or error_path.exists()):
        return None

    # Initialize composite columns to NaN unconditionally so the
    # DataFrame schema stays stable even when some rows have no
    # metrics file (partial runs, --skip-metrics, scoring failures).
    base = {
        "dataset": dataset,
        "method": method,
        "k": k,
        "fit_seconds": 0.0,
        "infer_seconds": None,
        "metric_seconds": 0.0,
        "peak_rss_mb": 0.0,
        "gpu_peak_mb": None,
        "error": None,
        "_bio": float("nan"),
        "_batch": float("nan"),
        "_composite": float("nan"),
    }
    if timing_path.exists():
        t = json.loads(timing_path.read_text())
        base["fit_seconds"] = t.get("fit_seconds", 0.0)
        base["infer_seconds"] = t.get("infer_seconds")
        base["peak_rss_mb"] = t.get("peak_rss_mb", 0.0)
        base["gpu_peak_mb"] = t.get("gpu_peak_mb")

    if error_path.exists():
        base["error"] = error_path.read_text().strip()

    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        base["metric_seconds"] = m.pop("_metric_seconds", 0.0)
        bio = float(m.get("Bio conservation", float("nan")))
        bat = float(m.get("Batch correction", float("nan")))
        tot = float(m.get("Total", float("nan")))
        base.update(m)
        base["_bio"], base["_batch"], base["_composite"] = bio, bat, tot
    return base


def aggregate_results(
    out_dir: Path, datasets: list[str], methods: list[str], impl: str, k: int | None = None
):
    import pandas as pd

    rows = []
    for ds in datasets:
        for m in methods:
            row = _load_method_row(out_dir, ds, m, impl, k=k)
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    return df


def _fmt_time(s) -> str:
    if s is None or (isinstance(s, float) and (s != s or s == 0)):
        return "—"
    if s < 0.5:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    return f"{s / 60:.1f}m"


def _fmt_mem(mb) -> str:
    if mb is None or (isinstance(mb, float) and (mb != mb or mb <= 0)):
        return "—"
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb / 1024:.1f} GB"


def write_summary(df, out_dir: Path) -> None:
    md: list[str] = []
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        md.append(f"### {ds}\n")
        md.append(
            "| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |"
        )
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            fit = _fmt_time(r.get("fit_seconds"))
            inf = _fmt_time(r.get("infer_seconds"))
            mt = _fmt_time(r.get("metric_seconds"))
            rss = _fmt_mem(r.get("peak_rss_mb"))
            gpu = _fmt_mem(r.get("gpu_peak_mb"))
            if r.get("error"):
                md.append(
                    f"| {r['method']} | {fit} | {inf} | {mt} | {rss} | {gpu} | "
                    f"`{str(r['error'])[:40]}` | — | — |"
                )
                continue
            md.append(
                f"| {r['method']} | {fit} | {inf} | {mt} | {rss} | {gpu} | "
                f"{r.get('_bio', float('nan')):+.3f} | "
                f"{r.get('_batch', float('nan')):+.3f} | "
                f"**{r.get('_composite', float('nan')):+.3f}** |"
            )
        md.append("")

    # Cross-dataset means
    if "_composite" in df.columns:
        means = (
            df.dropna(subset=["_composite"])
            .groupby("method")
            .agg(
                comp=("_composite", "mean"),
                fit=("fit_seconds", "mean"),
                gpu=(
                    "gpu_peak_mb",
                    lambda x: x.dropna().mean() if x.dropna().size else float("nan"),
                ),
            )
            .sort_values("comp", ascending=False)
        )
        md.append("### Cross-dataset composite average\n")
        md.append("| method | mean composite ↑ | mean fit | mean GPU peak |")
        md.append("|---|---:|---:|---:|")
        for name, row in means.iterrows():
            md.append(
                f"| {name} | **{row['comp']:+.3f}** | {_fmt_time(row['fit'])} | "
                f"{_fmt_mem(row['gpu'])} |"
            )

    (out_dir / "results.md").write_text("\n".join(md) + "\n")

    # JSON payload
    payload: dict = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        rows = []
        for _, r in sub.iterrows():
            rows.append(
                {
                    k: (
                        None
                        if (isinstance(v, float) and v != v)
                        else v.item()
                        if hasattr(v, "item")
                        else v
                    )
                    for k, v in r.to_dict().items()
                }
            )
        payload[ds] = rows
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))


def render_figures(df, out_dir: Path) -> None:
    try:
        from benchmarks.viz import plot_all
    except ImportError as e:
        print(f"(matplotlib/pandas missing — skipping figures: {e})")
        return
    fig_dir = out_dir / "figures"
    try:
        paths = plot_all(df, fig_dir)
    except Exception as e:
        print(f"(figure rendering hit an error: {e!r} — results files are still written)")
        return
    for label, p in paths.items():
        print(f"Wrote {p}  ({label})")


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(HERE),
        type=Path,
        help="Run root. Defaults to benchmarks/ (overwriting the canonical CSV).",
    )
    parser.add_argument(
        "--full", action="store_true", help="Run on full datasets (recommended on GPU)."
    )
    parser.add_argument(
        "--cells-per-cohort",
        type=int,
        default=50,
        help="Stratified per-(batch,label) subsample size. Ignored with --full.",
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-hvg", action="store_true")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--no-lisi", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Subset of methods. Choices: {list(METHOD_MODULES)}",
    )
    parser.add_argument("--datasets", nargs="+", default=["pancreas", "immune"])
    parser.add_argument(
        "--metrics-impl",
        choices=list(METRICS_MODULES),
        default="scib_yosef",
        help="Which metrics implementation to use. scib_original is the canonical "
        "Theis-lab one (requires glibc 2.38+). scib_yosef is the JAX rewrite.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Run only the embed step; leave metrics for a later invocation.",
    )
    args = parser.parse_args()

    if args.full:
        args.cells_per_cohort = None

    methods = args.methods or DEFAULT_METHODS
    for m in methods:
        if m not in METHOD_MODULES:
            parser.error(f"unknown method '{m}'; choices: {list(METHOD_MODULES)}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_impl == "scib_original":
        print(
            "WARNING: --metrics-impl scib_original uses a hand-aggregated bio/batch "
            "composite that differs from scib_yosef by a few hundredths. Do not "
            "compare composite scores across impls without recalibrating. See "
            "benchmarks/metrics/scib_original.py docstring.",
            flush=True,
        )

    # Phase 1: embed (one subprocess per method × dataset).
    t0 = time.perf_counter()
    fail_count = 0
    for ds in args.datasets:
        print(f"\n=== {ds} :: embed phase ===", flush=True)
        for m in methods:
            rc = run_method(m, ds, out_dir, args)
            if rc != 0:
                fail_count += 1

    # Phase 2: score (one subprocess per dataset, all methods at once).
    if not args.skip_metrics:
        for ds in args.datasets:
            print(f"\n=== {ds} :: metrics phase ({args.metrics_impl}) ===", flush=True)
            rc = run_metrics(args.metrics_impl, ds, methods, out_dir, args)
            # Non-zero rc from metrics just means some methods skipped — don't fail the whole run.
            del rc

    # Phase 3: aggregate + figures.
    df = aggregate_results(out_dir, args.datasets, methods, args.metrics_impl, k=args.k)
    write_summary(df, out_dir)
    render_figures(df, out_dir)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed / 60:.1f}m  ({fail_count} method failures)", flush=True)
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
