"""Visualization helpers for the scIB-style benchmark.

Designed to operate on the long-form ``results.csv`` that
``run_benchmark.py`` writes — one row per (dataset, method) with
columns: method, dataset, fit_seconds, infer_seconds,
metric_seconds, total_seconds, peak_rss_mb, gpu_peak_mb, plus the
scib-metrics columns (Total, Bio conservation, Batch correction,
individual metrics).

Four figures land into the output directory:

- ``composite_summary.png`` — grouped bar chart of the composite
  Total score; one cluster per dataset, bars=methods.
- ``per_task_bars.png`` — per-metric breakdown; one subplot per
  dataset, x=metric, bars=methods.
- ``bio_vs_batch_tradeoff.png`` — square scatter of the
  bio-conservation vs batch-correction frontier, methods in
  legend, iso-composite diagonals overlaid.
- ``score_vs_time.png`` — log fit time vs composite score, methods
  in legend.

**Consistent method colors** across every plot — once a method is
assigned a color it keeps it everywhere, so the eye can track the
same method from chart to chart.
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
    "dataset",
    "method",
    "fit_seconds",
    "infer_seconds",
    "metric_seconds",
    "total_seconds",
    "peak_rss_mb",
    "gpu_peak_mb",
    "error",
    "_impl",  # provenance — string column, not a metric
    *_AGGREGATE_COLS,
]


# ColorBrewer "Set1" — designed for maximum hue separation in a
# qualitative palette. Wong is CB-safer but its first 5 entries
# include two blues (#0072B2 + #56B4E9) that landed on important
# methods after alphabetical sorting (Harmony got dark blue,
# sparseNMF got sky blue). Set1's first 5 are red/blue/green/
# purple/orange — distinct hues, no near-duplicates.
_SET1 = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#666666",  # grey
    "#000000",  # black
    "#ffff33",  # yellow
]


def method_palette(methods: list[str]) -> dict[str, str]:
    """Stable per-method colour. Use this everywhere so a given
    method has the same colour across plots."""
    methods = sorted(methods)
    return {name: _SET1[i % len(_SET1)] for i, name in enumerate(methods)}


def _metric_cols(df: pd.DataFrame) -> list[str]:
    """Return the per-metric (non-aggregate, non-timing) columns."""
    return [c for c in df.columns if c not in _NON_METRIC_COLS]


def plot_composite_summary(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Grouped bar chart: composite Total per (dataset, method).

    Datasets on x-axis (one cluster each), bars within each cluster
    are methods coloured by the shared palette.

    Multi-seed aware: if ``df`` has multiple rows per (dataset, method)
    — e.g. a ``seed`` column with values {0,1,2} — bars show mean
    and ``yerr`` shows the standard deviation across seeds. With a
    single row per cell the std is 0 / NaN and no error bar is drawn.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted(df["method"].unique())
    # Aggregate across seeds (or any other replicate axis). With one
    # row per cell, mean == that value and std is NaN.
    grouped = df.groupby(["dataset", "method"])["Total"].agg(["mean", "std"]).reset_index()
    pivot_mean = (
        grouped.pivot(index="dataset", columns="method", values="mean")
        .reindex(columns=methods)
    )
    pivot_std = (
        grouped.pivot(index="dataset", columns="method", values="std")
        .reindex(columns=methods)
    )
    datasets = list(pivot_mean.index)
    palette = method_palette(methods)
    n_seeds = (
        df.groupby(["dataset", "method"]).size().max()
        if len(df) > 0 else 1
    )

    fig, ax = plt.subplots(figsize=(1.0 * max(len(datasets), 3) + 2.4, 4.0))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        vals = pivot_mean[method].values
        stds = pivot_std[method].values
        # matplotlib's yerr is happy with NaN — single-rep cells just
        # don't draw error bars. Convert NaN std to 0 to avoid a
        # FutureWarning from matplotlib's error-bar handling.
        yerr = np.nan_to_num(stds, nan=0.0)
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            yerr=yerr,
            label=method,
            color=palette[method],
            edgecolor="black",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "black"},
        )
        for b, v, s in zip(bars, vals, stds, strict=True):
            if np.isnan(v):
                continue
            # Position the label above any error-bar cap.
            cap = s if not np.isnan(s) else 0.0
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + cap + 0.005,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("scIB composite (Total)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    finite = pivot_mean.values[~np.isnan(pivot_mean.values)]
    if finite.size:
        # Pad top to make room for error bars + value labels.
        top_pad = float(np.nanmax(np.nan_to_num(pivot_std.values, nan=0.0))) + 0.10
        ax.set_ylim(
            min(0.0, float(finite.min()) - 0.05),
            max(float(finite.max()) + top_pad, 1.0),
        )
    if n_seeds > 1:
        title = (title or "scIB composite score per dataset × method") + f"  (mean ± std, n={n_seeds} seeds)"
    else:
        title = title or "scIB composite score per dataset × method"
    ax.set_title(title)
    # Legend OUTSIDE the axes on the right, vertical (one column).
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        title="method",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_task_bars(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Per-dataset metric bars. One subplot per dataset; x-axis is the
    individual metrics (plus the aggregate columns at the right);
    bars are methods, coloured by the shared palette. Replaces the
    rank-heatmap from earlier versions — easier to read at a glance,
    and uses the same method colours as every other plot.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = _metric_cols(df) + [c for c in _AGGREGATE_COLS if c in df.columns]
    if not metric_cols:
        raise ValueError("No metric columns found in results DataFrame")

    datasets = sorted(df["dataset"].unique())
    methods = sorted(df["method"].unique())
    palette = method_palette(methods)

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(0.55 * len(metric_cols) + 3.0, 2.8 * len(datasets) + 0.4),
        squeeze=False,
        sharex=True,
    )
    for i, dset in enumerate(datasets):
        ax = axes[i, 0]
        sub = df[df["dataset"] == dset].set_index("method").reindex(methods)
        x = np.arange(len(metric_cols))
        width = 0.8 / max(len(methods), 1)
        for j, method in enumerate(methods):
            row = sub.loc[method]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            vals = np.array([row.get(c, np.nan) for c in metric_cols], dtype=float)
            ax.bar(
                x + j * width - 0.4 + width / 2,
                vals,
                width,
                color=palette[method],
                edgecolor="black",
                linewidth=0.3,
                label=method if i == 0 else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_cols, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(dset, fontsize=11)
        ax.set_ylim(0.0, 1.05)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        # Separator before the aggregate columns
        n_indiv = len(metric_cols) - sum(1 for c in _AGGREGATE_COLS if c in metric_cols)
        if 0 < n_indiv < len(metric_cols):
            ax.axvline(n_indiv - 0.5, color="black", linewidth=0.5, linestyle=":")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # Vertical legend OUTSIDE the axes on the right of the top
        # subplot. Anchored to the top-right axes so it doesn't get
        # cropped by tight_layout.
        axes[0, 0].legend(
            handles,
            labels,
            title="method",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=9,
            borderaxespad=0.0,
        )
    fig.suptitle(title or "scIB per-metric scores by dataset", fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_score_vs_time(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Speed/accuracy frontier — log fit_seconds (x) vs composite Total (y).
    One marker per (method, dataset). Methods coloured by shared palette;
    datasets distinguished by marker shape. Method labels in legend (not
    on the plot)."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    palette = method_palette(methods)
    ds_marker = {d: m for d, m in zip(datasets, "ovsP^D*X", strict=False)}

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["Total", "fit_seconds"])
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            ax.scatter(
                row["fit_seconds"],
                row["Total"],
                color=palette[method],
                marker=ds_marker.get(row["dataset"], "o"),
                s=85,
                edgecolor="black",
                linewidth=0.6,
            )
    ax.set_xscale("log")
    ax.set_xlabel("fit time (s, log scale)")
    ax.set_ylabel("scIB composite (Total)")
    ax.set_title(title or "Speed vs accuracy")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    # Force square plotting area regardless of data range / log scale.
    ax.set_box_aspect(1)

    # Both legends OUTSIDE the axes to the right, vertically stacked:
    # method colours on top, dataset markers below.
    method_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=palette[m],
                   markeredgecolor="black", markersize=8, label=m)
        for m in methods
    ]
    ds_handles = [
        plt.Line2D([], [], marker=ds_marker.get(d, "o"), color="black",
                   linestyle="", markersize=7, markerfacecolor="white",
                   markeredgewidth=0.8, label=d)
        for d in datasets
    ]
    method_legend = ax.legend(
        handles=method_handles, title="method",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=8, borderaxespad=0.0,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=ds_handles, title="dataset",
        loc="upper left", bbox_to_anchor=(1.02, 0.55),
        frameon=False, fontsize=8, borderaxespad=0.0,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bio_vs_batch_tradeoff(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Canonical scIB Fig 2c trade-off: x = batch correction,
    y = bio conservation, one marker per (method, dataset). Square
    plot area, equal aspect; methods coloured by shared palette and
    labelled via the legend (not on-plot text)."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    palette = method_palette(methods)
    ds_marker = {d: m for d, m in zip(datasets, "ovsP^D*X", strict=False)}

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["Bio conservation", "Batch correction"])
        if sub.empty:
            continue
        # Connect this method's points across datasets with a thin line.
        ax.plot(
            sub["Batch correction"],
            sub["Bio conservation"],
            color=palette[method],
            linewidth=0.7,
            alpha=0.4,
        )
        for _, row in sub.iterrows():
            ax.scatter(
                row["Batch correction"],
                row["Bio conservation"],
                color=palette[method],
                marker=ds_marker.get(row["dataset"], "o"),
                s=85,
                edgecolor="black",
                linewidth=0.6,
            )

    # Faint diagonal iso-composite lines: total = 0.4*batch + 0.6*bio.
    x_vals = np.linspace(0, 1, 100)
    for total in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        y_vals = (total - 0.4 * x_vals) / 0.6
        mask = (y_vals >= 0) & (y_vals <= 1)
        ax.plot(
            x_vals[mask], y_vals[mask],
            color="gray", linewidth=0.4, linestyle=":", alpha=0.5,
        )
        if mask.any():
            ax.annotate(
                f"Total={total:.1f}",
                (x_vals[mask][-1], y_vals[mask][-1]),
                fontsize=6, color="gray",
                xytext=(-30, 4), textcoords="offset points",
            )

    ax.set_xlabel("Batch correction (composite)")
    ax.set_ylabel("Bio conservation (composite)")
    ax.set_title(title or "Trade-off: bio conservation vs batch correction")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")  # SQUARE plot area

    # Legends: methods (color) + datasets (marker).
    method_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=palette[m],
                   markeredgecolor="black", markersize=8, label=m)
        for m in methods
    ]
    ds_handles = [
        plt.Line2D([], [], marker=ds_marker.get(d, "o"), color="black",
                   linestyle="", markersize=7, markerfacecolor="white",
                   markeredgewidth=0.8, label=d)
        for d in datasets
    ]
    # Both legends OUTSIDE the axes on the right, vertically stacked:
    # method colours on top, dataset markers below.
    method_legend = ax.legend(
        handles=method_handles, title="method",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=8, borderaxespad=0.0,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=ds_handles, title="dataset",
        loc="upper left", bbox_to_anchor=(1.02, 0.55),
        frameon=False, fontsize=8, borderaxespad=0.0,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all(df: pd.DataFrame, out_dir: Path | str) -> dict[str, Path]:
    """Render all four benchmark figures into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "composite": plot_composite_summary(df, out_dir / "composite_summary.png"),
        "per_task": plot_per_task_bars(df, out_dir / "per_task_bars.png"),
        "tradeoff": plot_bio_vs_batch_tradeoff(df, out_dir / "bio_vs_batch_tradeoff.png"),
        "speed_score": plot_score_vs_time(df, out_dir / "score_vs_time.png"),
    }
