"""Hyperparameter sweep for sparseNMF — train multiple configs and
record per-config quality + cost metrics.

Designed as a *lightweight* alternative to the scIB benchmark: instead
of running the full scib-metrics suite (LISI, NMI, ARI, ...), this
module records pretty quick downstream metrics directly:

- **silhouette_label** — Bio signal: how well the embedding clusters
  by cell type. Higher = better.
- **silhouette_batch** — Batch mixing: how much the embedding
  separates batches. Lower (more negative) = better.
- **train_seconds** — wall-clock fit time.
- **n_iter** — iterations until early stop or max_iter.
- **W_sparsity** — fraction of entries in the W matrix below a
  numerical threshold. Higher = more interpretable factor loadings.

The sweep takes a list of config dicts (each is kwargs for either
``train_sparse_nmf`` or ``train_sparse_nmf_batch_aware``); the
caller picks which to use via ``mode``. Single-pass sweep returns
a long-form DataFrame the caller can pivot/plot however they like.

Example::

    from sparse_nmf import sweep_hyperparameters
    from sparse_nmf.data import load_pancreas

    adata = load_pancreas()
    configs = [
        {"n_components": k, "normalize_inputs": norm}
        for k in [10, 20, 30, 50, 100]
        for norm in [True, False]
    ]
    result = sweep_hyperparameters(
        adata.X, configs,
        labels=adata.obs["celltype"].values,
        batch=adata.obs["tech"].values,
    )
    result.df.to_csv("sweep.csv", index=False)
    result.plot("sweep_figures/")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix

# ── Quality metrics (cheap; no scIB stack) ───────────────────────────


def _safe_silhouette(W: np.ndarray, labels, max_cells: int = 5000) -> float:
    """sklearn silhouette is O(n²) — subsample for large datasets."""
    from sklearn.metrics import silhouette_score

    n = len(W)
    if n == 0 or labels is None:
        return float("nan")
    # Need at least 2 distinct labels; sklearn errors otherwise.
    if len(set(labels)) < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    if n > max_cells:
        idx = rng.choice(n, size=max_cells, replace=False)
        W = W[idx]
        labels = np.asarray(labels)[idx]
    try:
        return float(silhouette_score(W, labels))
    except Exception:
        return float("nan")


def _sparsity(W: np.ndarray, threshold: float = 1e-3) -> float:
    """Fraction of W entries with magnitude below ``threshold``."""
    return float(np.mean(np.abs(W) < threshold))


# ── Sweep core ──────────────────────────────────────────────────────


@dataclass
class SweepResult:
    """One row per (config, dataset_name) in ``.df``. The DataFrame
    is the canonical artifact — plot helpers read from it."""

    df: object  # pandas.DataFrame (typed `object` so this module
    #   doesn't need a hard pandas import at decl time)

    def plot(self, out_dir: Path | str) -> dict[str, Path]:
        """Write multi-panel figures to ``out_dir``. Returns
        {label: path}."""
        return _plot_sweep(self.df, Path(out_dir))


def sweep_hyperparameters(
    X_sparse: spmatrix,
    configs: list[dict],
    *,
    labels=None,
    batch=None,
    mode: str = "standard",
    dataset_name: str = "dataset",
    device: str = "cuda",
    verbose: bool = True,
):
    """Run sparseNMF for each config, record per-config metrics.

    Parameters
    ----------
    X_sparse
        Sparse cell × gene matrix.
    configs
        List of kwargs dicts. Each must include ``n_components``.
        Other recognized keys depend on ``mode``:
        - ``"standard"``: passes to ``train_sparse_nmf``.
          Useful keys: ``normalize_inputs``, ``max_iter``,
          ``patience``, ``nonzero_mse_weight``, ``random_state``.
        - ``"batch_aware"``: passes to ``train_sparse_nmf_batch_aware``.
          Useful keys: ``alignment_weight``, ``sparsity_weight``,
          plus the standard ones above. Requires ``batch`` to be set.
    labels
        Cell-type labels for silhouette-bio metric (skipped if None).
    batch
        Batch labels. Required for ``mode="batch_aware"``; also used
        for silhouette-batch metric when present.
    mode
        ``"standard"`` (default) or ``"batch_aware"``.
    dataset_name
        Tag included in every result row. Lets the caller call
        ``sweep_hyperparameters`` for multiple datasets and concat
        the resulting DataFrames.
    device
        ``"cuda"`` or ``"cpu"``. Auto-falls-back to cpu when
        ``torch.cuda.is_available()`` is False.
    verbose
        Print one-line progress per config.

    Returns
    -------
    SweepResult — wrapper around a long-form DataFrame.
    """
    import pandas as pd
    import torch

    from sparse_nmf import (
        train_sparse_nmf,
        train_sparse_nmf_batch_aware,
    )

    if not torch.cuda.is_available() and device.startswith("cuda"):
        if verbose:
            print("  cuda not available — falling back to cpu", flush=True)
        device = "cpu"

    rows: list[dict] = []
    for i, cfg in enumerate(configs):
        k = cfg.get("n_components")
        if k is None:
            raise ValueError(f"config {i} missing 'n_components': {cfg}")
        cfg = dict(cfg)
        cfg.setdefault("random_state", 0)
        cfg.setdefault("verbose", False)

        log_summary = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "verbose")
        if verbose:
            print(f"  [{i + 1}/{len(configs)}] {dataset_name}: {log_summary}", flush=True)

        t0 = time.perf_counter()
        if mode == "standard":
            W, model = train_sparse_nmf(X_sparse=X_sparse, device=device, **cfg)
            n_iter = getattr(model, "n_iter_", None)
        elif mode == "batch_aware":
            if batch is None:
                raise ValueError("mode='batch_aware' requires batch=...")
            res = train_sparse_nmf_batch_aware(
                X_sparse=X_sparse,
                batch=np.asarray(batch),
                device=device,
                **cfg,
            )
            W = res.W
            n_iter = res.n_iter
        else:
            raise ValueError(f"unknown mode={mode!r}; expected 'standard' or 'batch_aware'")
        train_seconds = time.perf_counter() - t0

        row = {
            "dataset": dataset_name,
            "mode": mode,
            "k": int(cfg.get("n_components", 0)),
            "normalize_inputs": cfg.get("normalize_inputs", True),
            "nonzero_mse_weight": cfg.get("nonzero_mse_weight", 0.0),
            "alignment_weight": cfg.get("alignment_weight", float("nan")),
            "sparsity_weight": cfg.get("sparsity_weight", float("nan")),
            "max_iter": cfg.get("max_iter", 500),
            "random_state": cfg.get("random_state", 0),
            "train_seconds": train_seconds,
            "n_iter": n_iter,
            "W_sparsity": _sparsity(W),
            "silhouette_label": _safe_silhouette(W, labels),
            "silhouette_batch": _safe_silhouette(W, batch),
        }
        rows.append(row)
        if verbose:
            sl = row["silhouette_label"]
            sb = row["silhouette_batch"]
            print(
                f"      → sil_label={sl:+.3f}  sil_batch={sb:+.3f}  "
                f"t={train_seconds:.1f}s  sparsity={row['W_sparsity']:.2f}",
                flush=True,
            )
    return SweepResult(pd.DataFrame(rows))


# ── Plotting ────────────────────────────────────────────────────────


def _plot_sweep(df, out_dir: Path) -> dict[str, Path]:
    """Render 4-panel summary figure + per-axis line plots."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    datasets = sorted(df["dataset"].unique())
    palette = {
        d: c
        for d, c in zip(
            datasets,
            [
                "#e41a1c",
                "#377eb8",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
            ],
            strict=False,
        )
    }

    # ── Panel A: k vs silhouette_label ──────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))

    for d in datasets:
        sub = df[df["dataset"] == d].dropna(subset=["k", "silhouette_label"]).sort_values("k")
        grouped = sub.groupby("k")["silhouette_label"].agg(["mean", "std"]).reset_index()
        axes[0].plot(grouped["k"], grouped["mean"], "-o", color=palette[d], label=d, markersize=5)
        axes[0].errorbar(
            grouped["k"],
            grouped["mean"],
            yerr=np.nan_to_num(grouped["std"].values, nan=0.0),
            fmt="none",
            ecolor=palette[d],
            elinewidth=0.8,
            alpha=0.6,
            capsize=0,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("k (latent dim, log)")
    axes[0].set_ylabel("silhouette (label) ↑")
    axes[0].set_title("Bio signal vs k")
    axes[0].grid(True, which="both", linewidth=0.3, alpha=0.5)

    # ── Panel B: k vs silhouette_batch ──────────────────────────────
    for d in datasets:
        sub = df[df["dataset"] == d].dropna(subset=["k", "silhouette_batch"]).sort_values("k")
        grouped = sub.groupby("k")["silhouette_batch"].agg(["mean", "std"]).reset_index()
        axes[1].plot(grouped["k"], grouped["mean"], "-o", color=palette[d], label=d, markersize=5)
        axes[1].errorbar(
            grouped["k"],
            grouped["mean"],
            yerr=np.nan_to_num(grouped["std"].values, nan=0.0),
            fmt="none",
            ecolor=palette[d],
            elinewidth=0.8,
            alpha=0.6,
            capsize=0,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("k (latent dim, log)")
    axes[1].set_ylabel("silhouette (batch) ↓")
    axes[1].set_title("Batch mixing vs k (lower=better)")
    axes[1].axhline(0, color="gray", linewidth=0.4, linestyle="--")
    axes[1].grid(True, which="both", linewidth=0.3, alpha=0.5)

    # ── Panel C: k vs train_time ────────────────────────────────────
    for d in datasets:
        sub = df[df["dataset"] == d].dropna(subset=["k", "train_seconds"]).sort_values("k")
        grouped = sub.groupby("k")["train_seconds"].agg(["mean", "std"]).reset_index()
        axes[2].plot(grouped["k"], grouped["mean"], "-o", color=palette[d], label=d, markersize=5)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("k (latent dim, log)")
    axes[2].set_ylabel("train time (s, log)")
    axes[2].set_title("Cost vs k")
    axes[2].grid(True, which="both", linewidth=0.3, alpha=0.5)

    # ── Panel D: k vs W_sparsity ────────────────────────────────────
    for d in datasets:
        sub = df[df["dataset"] == d].dropna(subset=["k", "W_sparsity"]).sort_values("k")
        grouped = sub.groupby("k")["W_sparsity"].agg(["mean", "std"]).reset_index()
        axes[3].plot(grouped["k"], grouped["mean"], "-o", color=palette[d], label=d, markersize=5)
    axes[3].set_xscale("log")
    axes[3].set_xlabel("k (latent dim, log)")
    axes[3].set_ylabel("W sparsity (frac < 1e-3)")
    axes[3].set_title("Embedding sparsity vs k")
    axes[3].grid(True, which="both", linewidth=0.3, alpha=0.5)

    # Single legend at the right of the whole row.
    axes[0].legend(
        title="dataset",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=len(datasets),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("sparseNMF hyperparameter sweep", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_k.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["sweep_k"] = out_dir / "sweep_k.png"

    # ── Panel E: bio vs batch trade-off scatter ─────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for d in datasets:
        sub = df[df["dataset"] == d].dropna(subset=["silhouette_label", "silhouette_batch"])
        ax.scatter(
            sub["silhouette_batch"],
            sub["silhouette_label"],
            c=[palette[d]] * len(sub),
            s=60 + sub["k"].values,  # marker size ~ k
            edgecolor="black",
            linewidth=0.6,
            alpha=0.7,
            label=d,
        )
    ax.set_xlabel("silhouette (batch) ←  lower is better")
    ax.set_ylabel("silhouette (label) →  higher is better")
    ax.set_title("Bio-vs-batch trade-off (marker size ∝ k)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(
        title="dataset", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["sweep_tradeoff"] = out_dir / "sweep_tradeoff.png"

    # ── Panel F: normalize_inputs and nonzero_mse comparison ────────
    # Faceted bars: at fixed k=30, compare {normalize_inputs} × {nonzero_mse_weight>0}.
    facet = df[df["k"] == 30].copy() if 30 in df["k"].values else df.copy()
    facet["loss_mode"] = facet["nonzero_mse_weight"].apply(
        lambda x: "nonzero-only" if x and x > 0 else "all (incl. zeros)"
    )
    facet["norm_mode"] = facet["normalize_inputs"].apply(lambda x: "L2-row-norm" if x else "raw")
    if not facet.empty:
        agg = (
            facet.groupby(["dataset", "loss_mode", "norm_mode"])["silhouette_label"]
            .agg(["mean", "std"])
            .reset_index()
        )
        groups = sorted(
            {(lm, nm) for lm in agg.loss_mode.unique() for nm in agg.norm_mode.unique()}
        )
        n_groups = len(groups)
        x = np.arange(len(datasets))
        width = 0.8 / max(n_groups, 1)
        fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(datasets) + 2.0), 4.0))
        for i, (lm, nm) in enumerate(groups):
            vals = []
            stds = []
            for d in datasets:
                row = agg[(agg.dataset == d) & (agg.loss_mode == lm) & (agg.norm_mode == nm)]
                vals.append(row["mean"].values[0] if not row.empty else np.nan)
                stds.append(row["std"].values[0] if not row.empty else np.nan)
            ax.bar(
                x + i * width - 0.4 + width / 2,
                vals,
                width,
                yerr=np.nan_to_num(stds, nan=0.0),
                label=f"{lm} / {nm}",
                edgecolor="black",
                linewidth=0.5,
                error_kw={"elinewidth": 0.7, "capsize": 0},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel("silhouette (label) ↑")
        ax.set_title("Loss × normalization (at k=30)")
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "sweep_loss_norm.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["sweep_loss_norm"] = out_dir / "sweep_loss_norm.png"

    return paths
