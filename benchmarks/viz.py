"""Visualization helpers for the scIB-style benchmark.

Designed to operate on the long-form ``results.csv`` that
``run_benchmark.py`` writes — one row per (dataset, method) with
columns: method, dataset, fit_seconds, infer_seconds,
metric_seconds, total_seconds, peak_rss_mb, gpu_peak_mb, plus the
scib-metrics columns (Total, Bio conservation, Batch correction,
individual metrics).

Three figures land:

- ``method_comparison.png`` — grouped bar chart, one panel per
  metric (cell-type silhouette, batch ASW, …) with methods on the
  x-axis and a bar per dataset. The right level of detail for
  "which methods win on which axis."
- ``composite_summary.png`` — the headline bar chart of the
  composite Total score per (method, dataset). What goes in a
  README.
- ``score_vs_time.png`` — scatter of log fit time vs composite
  score, methods labeled. Visualizes the speed/accuracy frontier.

scib-metrics also ships a styled HTML / matplotlib results table
via ``Benchmarker.plot_results_table()``; we expose
``plot_scib_results_table`` as a thin wrapper that runs the
metrics once more, captures the figure, and saves it alongside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# scib-metrics aggregate column names. The benchmark CSV keeps
# whatever scib-metrics emits; these are stable across recent
# 0.5.x versions.
_AGGREGATE_COLS = ["Total", "Bio conservation", "Batch correction"]
_NON_METRIC_COLS = [
    "dataset", "method",
    "fit_seconds", "infer_seconds", "metric_seconds",
    "total_seconds", "peak_rss_mb", "gpu_peak_mb", "error",
    *_AGGREGATE_COLS,
]


def _palette(names: list[str]) -> dict[str, str]:
    """Stable per-method color (CB-friendly Wong palette, repeating)."""
    base = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
        "#D55E00", "#F0E442", "#000000", "#999999",
    ]
    return {name: base[i % len(base)] for i, name in enumerate(names)}


def _metric_cols(df: pd.DataFrame) -> list[str]:
    """Return the per-metric (non-aggregate, non-timing) columns."""
    return [c for c in df.columns if c not in _NON_METRIC_COLS]


def plot_composite_summary(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Grouped bar chart: composite Total per (method, dataset)."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = df.pivot(index="method", columns="dataset", values="Total").fillna(np.nan)
    methods = list(pivot.index)
    datasets = list(pivot.columns)
    palette = _palette(datasets)

    fig, ax = plt.subplots(figsize=(1.4 * max(len(methods), 4) + 1.5, 4.5))
    x = np.arange(len(methods))
    width = 0.8 / max(len(datasets), 1)
    for i, dset in enumerate(datasets):
        vals = pivot[dset].values
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=dset,
            color=palette[dset],
            edgecolor="black",
            linewidth=0.5,
        )
        for b, v in zip(bars, vals, strict=True):
            if np.isnan(v):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("scIB composite (Total)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(min(0.0, pivot.values[~np.isnan(pivot.values)].min() - 0.05),
                max(pivot.values[~np.isnan(pivot.values)].max() + 0.08, 1.0))
    ax.legend(title="dataset", frameon=False, loc="best", fontsize=8)
    ax.set_title(title or "scIB composite score per method × dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_method_comparison(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Per-metric grouped bars. One small panel per scIB metric;
    methods on the x-axis; bars colored by dataset.

    Skips the aggregate columns (Total/Bio/Batch) since those have
    their own figure."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = _metric_cols(df)
    if not metric_cols:
        raise ValueError("No per-metric columns found in results DataFrame")

    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    palette = _palette(datasets)

    ncols = min(3, len(metric_cols))
    nrows = (len(metric_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 2.8 * nrows),
        squeeze=False,
    )

    for idx, metric in enumerate(metric_cols):
        ax = axes[idx // ncols, idx % ncols]
        sub = df.pivot(index="method", columns="dataset", values=metric).reindex(methods)
        x = np.arange(len(methods))
        width = 0.8 / max(len(datasets), 1)
        for i, dset in enumerate(datasets):
            vals = sub[dset].values if dset in sub.columns else np.full(len(methods), np.nan)
            ax.bar(
                x + i * width - 0.4 + width / 2,
                vals,
                width,
                color=palette[dset],
                edgecolor="black",
                linewidth=0.3,
                label=dset if idx == 0 else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    # Hide unused subplot cells
    for j in range(len(metric_cols), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="dataset", loc="lower center",
                   ncol=len(datasets), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title or "scIB per-metric scores", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_score_vs_time(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Speed/accuracy frontier — log fit_seconds (x) vs composite (y),
    one marker per (method, dataset), connected within each dataset
    by a thin line for visual scan."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    method_color = _palette(methods)
    ds_marker = {d: m for d, m in zip(datasets, "ovsP^D*X", strict=False)}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in methods:
        sub = df[df["method"] == method]
        for _, row in sub.iterrows():
            if pd.isna(row.get("Total")) or pd.isna(row.get("fit_seconds")):
                continue
            ax.scatter(
                row["fit_seconds"],
                row["Total"],
                color=method_color[method],
                marker=ds_marker.get(row["dataset"], "o"),
                s=70,
                edgecolor="black",
                linewidth=0.6,
                label=f"{method} ({row['dataset']})",
            )
    # One label per method, near its centroid
    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["fit_seconds", "Total"])
        if sub.empty:
            continue
        x = float(np.median(sub["fit_seconds"]))
        y = float(np.median(sub["Total"]))
        ax.annotate(
            method, (x, y),
            xytext=(6, 3), textcoords="offset points",
            fontsize=8, fontweight="bold",
            color=method_color[method],
        )

    ax.set_xscale("log")
    ax.set_xlabel("fit time (s, log scale)")
    ax.set_ylabel("scIB composite (Total)")
    ax.set_title(title or "Speed vs accuracy")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    # Dataset marker legend
    handles = [
        plt.Line2D([], [], marker=ds_marker.get(d, "o"),
                   color="black", linestyle="", markersize=7,
                   markerfacecolor="white", markeredgewidth=0.8,
                   label=d)
        for d in datasets
    ]
    ax.legend(handles=handles, title="dataset", loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bio_vs_batch_tradeoff(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Bio-vs-batch trade-off scatter — the canonical scIB figure 2c.

    x = Batch correction (composite), y = Bio conservation (composite).
    One marker per (method, dataset); points for the same method are
    connected by a thin line. Methods in the top-right corner are the
    Pareto winners (good biology AND good batch mixing)."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    method_color = _palette(methods)
    ds_marker = {d: m for d, m in zip(datasets, "ovsP^D*X", strict=False)}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["Bio conservation", "Batch correction"])
        if sub.empty:
            continue
        # Connect this method's points across datasets with a thin line.
        ax.plot(
            sub["Batch correction"], sub["Bio conservation"],
            color=method_color[method], linewidth=0.7, alpha=0.5,
        )
        for _, row in sub.iterrows():
            ax.scatter(
                row["Batch correction"],
                row["Bio conservation"],
                color=method_color[method],
                marker=ds_marker.get(row["dataset"], "o"),
                s=80,
                edgecolor="black",
                linewidth=0.6,
            )
        # Annotate at the centroid.
        cx = float(sub["Batch correction"].mean())
        cy = float(sub["Bio conservation"].mean())
        ax.annotate(
            method, (cx, cy),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, fontweight="bold",
            color=method_color[method],
        )

    ax.set_xlabel("Batch correction (composite)")
    ax.set_ylabel("Bio conservation (composite)")
    ax.set_title(title or "Trade-off: bio conservation vs batch correction")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    # Add a faint "Pareto" gradient — diagonal lines of constant
    # 0.6·bio + 0.4·batch (the scIB composite formula).
    for total in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        # bio = (total - 0.4*batch) / 0.6
        x_vals = np.linspace(0, 1, 50)
        y_vals = (total - 0.4 * x_vals) / 0.6
        mask = (y_vals >= 0) & (y_vals <= 1)
        ax.plot(x_vals[mask], y_vals[mask],
                color="gray", linewidth=0.4, linestyle=":", alpha=0.5)
        # Label each iso-composite line at the top edge.
        if mask.any():
            yi = y_vals[mask][0]
            xi = x_vals[mask][0]
            ax.annotate(f"Total={total:.1f}", (xi, yi),
                        fontsize=6, color="gray",
                        xytext=(2, -2), textcoords="offset points")
    handles = [
        plt.Line2D([], [], marker=ds_marker.get(d, "o"),
                   color="black", linestyle="", markersize=7,
                   markerfacecolor="white", markeredgewidth=0.8,
                   label=d)
        for d in datasets
    ]
    ax.legend(handles=handles, title="dataset", loc="best",
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ranking_heatmap(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
    per_dataset: bool = False,
) -> Path:
    """scIB-paper-style ranking heatmap — rows = methods, columns =
    metrics, cells color-coded by *rank within the column* (best in
    green, worst in red). Annotations show the actual metric value.

    ``per_dataset=False``: average each metric across datasets first.
    ``per_dataset=True``: one heatmap per dataset stacked vertically."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = _metric_cols(df) + _AGGREGATE_COLS
    metric_cols = [c for c in metric_cols if c in df.columns]

    # scIB-paper colormap: red (low rank) → yellow → green (high rank).
    cmap = LinearSegmentedColormap.from_list(
        "scib_rank",
        ["#d62728", "#fcdd5d", "#2ca02c"],
        N=256,
    )

    def _render_one(table: pd.DataFrame, ax) -> None:
        # Rank within each column (1 = best, larger = worse → invert for
        # colormap so green = best).
        ranks = table.rank(axis=0, ascending=False, method="min")
        n = len(table.index)
        rank_norm = 1.0 - (ranks - 1) / max(n - 1, 1)
        im = ax.imshow(rank_norm.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        # Annotate raw scores
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.iat[i, j]
                txt = "—" if pd.isna(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_xticks(np.arange(table.shape[1]))
        ax.set_xticklabels(table.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(table.shape[0]))
        ax.set_yticklabels(table.index, fontsize=9)
        ax.tick_params(axis="both", which="both", length=0)
        return im

    if per_dataset:
        datasets = sorted(df["dataset"].unique())
        fig, axes = plt.subplots(
            len(datasets), 1,
            figsize=(0.8 * len(metric_cols) + 2, 3.5 * len(datasets)),
            squeeze=False,
        )
        for i, dset in enumerate(datasets):
            sub = df[df["dataset"] == dset].set_index("method")[metric_cols]
            sub = sub.reindex(sorted(sub.index))
            _render_one(sub, axes[i, 0])
            axes[i, 0].set_title(dset, fontsize=10)
        fig.suptitle(title or "scIB metrics — ranked per dataset", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        avg = (
            df.groupby("method")[metric_cols]
            .mean(numeric_only=True)
            .sort_values(by="Total" if "Total" in metric_cols else metric_cols[0],
                         ascending=False)
        )
        fig, ax = plt.subplots(
            figsize=(0.8 * len(metric_cols) + 2, 0.55 * len(avg) + 1.5)
        )
        _render_one(avg, ax)
        ax.set_title(title or "scIB metrics — averaged across datasets, ranked", fontsize=11)
        fig.tight_layout()

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all(df: pd.DataFrame, out_dir: Path | str) -> dict[str, Path]:
    """Render all five benchmark figures into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "composite": plot_composite_summary(df, out_dir / "composite_summary.png"),
        "per_metric": plot_method_comparison(df, out_dir / "method_comparison.png"),
        "speed_score": plot_score_vs_time(df, out_dir / "score_vs_time.png"),
        "tradeoff": plot_bio_vs_batch_tradeoff(df, out_dir / "bio_vs_batch_tradeoff.png"),
        "ranking": plot_ranking_heatmap(df, out_dir / "ranking_heatmap.png"),
    }
