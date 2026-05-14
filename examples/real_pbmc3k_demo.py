"""Demo on real scRNA-seq data: PBMC 3k.

The synthetic demo in ``sparsity_confound_demo.py`` constructed a
worst-case sparsity confound to show what PCA / NMF look like when
the per-row magnitude axis dominates variance. This demo runs the
same comparison on **real single-cell data** — the 10x Genomics
PBMC 3k dataset bundled with scanpy:

* 2,638 peripheral-blood mononuclear cells × 13,714 genes (raw
  counts, post-QC). The demo stratified-subsamples to 100 cells
  per cell type (~750 cells total) so the whole script runs in
  about a minute on CPU. Set ``cells_per_type`` higher in
  :func:`load_pbmc3k` to scale up.
* 8 published Louvain cell types: B cells, CD14+ Monocytes,
  CD4 T cells, CD8 T cells, Dendritic cells, FCGR3A+ Monocytes,
  Megakaryocytes, NK cells.
* Natural per-cell library-depth variation: ~16× between the
  shallowest (~556 UMI) and deepest (~8,875 UMI) cells.

Three 2-D embeddings are produced, all at the same latent dim ``k``
followed by the same UMAP projection — only the factorization
differs:

* **PCA(k)** on the raw matrix.
* **NMF(k)** (sklearn) on the raw matrix.
* **sparseNMF(k=auto)** — ``train_sparse_nmf(X)`` with the
  library defaults (``normalize_inputs=True``, ``patience=10``,
  auto-sized ``n_components``).

The 2×3 facet:

* Row 1 — colored by **published cell type** (categorical, 8 colors).
* Row 2 — colored by **log10(UMI counts / cell)** — the library
  depth, the technical axis we want the factorization NOT to align
  with.

A note on what the figure shows. Unlike the synthetic demo, which
constructs a worst-case sparsity confound, PBMC is a clean
single-protocol dataset where the 8 cell types are biologically
distinct enough that linear methods can recover them. So this demo
is more honest than dramatic: PCA does well on its own (cell-type
silhouette ≈ +0.44 at the subsample size used), vanilla NMF without
preprocessing struggles disproportionately on a small dataset (~+0.16,
dragging the within-cluster structure into a few super-clusters),
and ``sparseNMF`` — with ``normalize_inputs=True`` as the default —
matches PCA's cell-type separation (≈ +0.44) while preserving the
parts-based interpretable factorization NMF gives you. All three
methods successfully avoid latching onto the library-depth axis
(depth-quartile silhouette is near zero or slightly negative — i.e.
quartiles are well-mixed across the embedding).

Run from repo root::

    python examples/real_pbmc3k_demo.py

Writes ``docs/_static/real_pbmc3k_demo.png``. First run downloads
the dataset (~24 MB cached to ~/.cache/scanpy).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse

from sparse_nmf import train_sparse_nmf


def load_pbmc3k(
    cells_per_type: int = 100, seed: int = 0
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Return raw counts (CSR), cell-type labels, and per-cell UMI
    totals from the scanpy-bundled processed PBMC 3k.

    Uses ``raw.X`` so the input is genuine sparse counts rather than
    the log-normalized HVG matrix used for the published UMAP. A
    stratified subsample (``cells_per_type`` per cell type, or all
    cells if a type has fewer than this) keeps the demo CPU-runnable
    end-to-end in under a minute — pass a larger value or skip the
    subsample for a heavier benchmark.
    """
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "This demo requires scanpy. Install with: pip install 'sparse-nmf[viz]'"
        ) from e

    proc = sc.datasets.pbmc3k_processed()
    raw = proc.raw
    X = raw.X
    X = X if issparse(X) else csr_matrix(X)
    X = X.tocsr().astype(np.float32)
    labels = np.asarray(proc.obs["louvain"].astype(str).values)
    n_counts = np.asarray(proc.obs["n_counts"].values, dtype=np.float32)

    rng = np.random.default_rng(seed)
    keep_idx = []
    for ct in np.unique(labels):
        in_ct = np.where(labels == ct)[0]
        take = min(cells_per_type, in_ct.size)
        keep_idx.append(rng.choice(in_ct, size=take, replace=False))
    keep = np.sort(np.concatenate(keep_idx))
    return X[keep], labels[keep], n_counts[keep]


def fit_pca(X: csr_matrix, seed: int, k: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=k, random_state=seed).fit_transform(X.toarray())


def fit_nmf(X: csr_matrix, seed: int, k: int) -> np.ndarray:
    from sklearn.decomposition import NMF

    return NMF(n_components=k, init="nndsvd", max_iter=500, random_state=seed).fit_transform(X)


def fit_sparse_nmf(X: csr_matrix, seed: int, k: int) -> np.ndarray:
    # Pass n_components=k explicitly to match the baselines. Everything
    # else uses the library defaults — normalize_inputs=True is the
    # whole point of the comparison.
    W, _model = train_sparse_nmf(
        X_sparse=X,
        n_components=k,
        device="cpu",
        random_state=seed,
        verbose=False,
    )
    return W


def umap_project(X_high: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "This demo requires umap-learn. Install with: pip install 'sparse-nmf[viz]'"
        ) from e
    return umap.UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(X_high)


def make_figure(
    embeddings: dict[str, np.ndarray],
    metrics: dict[str, tuple[float, float]],
    labels: np.ndarray,
    log_depth: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    methods = list(embeddings)
    n = len(methods)
    fig, axes = plt.subplots(
        2,
        n + 1,
        figsize=(3.7 * n + 0.7, 7.4),
        gridspec_kw={"width_ratios": [1.0] * n + [0.06]},
    )

    # Stable colour assignment for the 8 cell types (cb-friendly-ish).
    unique = sorted(np.unique(labels))
    palette = [
        "#e6194B",
        "#3cb44b",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#42d4f4",
        "#f032e6",
        "#9A6324",
    ]
    group_cmap = ListedColormap(palette[: len(unique)])
    label_to_int = {lab: i for i, lab in enumerate(unique)}
    label_idx = np.asarray([label_to_int[lab] for lab in labels])

    perm = np.random.default_rng(0).permutation(len(labels))
    labels_p = label_idx[perm]
    depth_p = log_depth[perm]

    sc_depth = None
    for col, name in enumerate(methods):
        z = embeddings[name][perm]
        sg, sb = metrics[name]

        # Row 0: by cell type.
        ax = axes[0, col]
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=labels_p,
            cmap=group_cmap,
            vmin=0,
            vmax=len(unique) - 1,
            s=8,
            alpha=0.85,
            linewidth=0,
        )
        ax.set_title(
            f"{name}\nsilhouette: cell-type={sg:+.2f}  depth-quartile={sb:+.2f}",
            fontsize=10,
        )
        if col == 0:
            ax.set_ylabel("colored by\npublished cell type", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 1: by log10(UMI total per cell).
        ax = axes[1, col]
        sc_depth = ax.scatter(
            z[:, 0],
            z[:, 1],
            c=depth_p,
            cmap="viridis",
            s=8,
            alpha=0.85,
            linewidth=0,
        )
        if col == 0:
            ax.set_ylabel("colored by\nlog10(UMI / cell)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, -1].axis("off")
    cbar = fig.colorbar(sc_depth, cax=axes[1, -1])
    cbar.set_label("log10(UMI / cell)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=group_cmap(i),
            markersize=7,
            linestyle="",
            label=unique[i],
        )
        for i in range(len(unique))
    ]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="best", frameon=False)

    fig.suptitle(
        "Real PBMC scRNA-seq (10x Genomics 3k): same k, same UMAP step.\n"
        "Only the factorization differs.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    seed = 0
    print("Loading PBMC 3k (scanpy-bundled)...")
    X, labels, n_counts = load_pbmc3k()
    log_depth = np.log10(n_counts + 1)
    print(f"  X shape={X.shape}  nnz={X.nnz:,}  density={X.nnz / (X.shape[0] * X.shape[1]):.2%}")
    print(
        f"  {len(np.unique(labels))} cell types  "
        f"depth range: {n_counts.min():.0f} – {n_counts.max():.0f} UMI/cell"
    )

    k = int(np.clip(min(X.shape) // 8, 32, 1024))
    print(f"  shared latent dim k={k} (auto-sized from input shape)")

    # Depth quartile is the "batch-like" proxy: cells in the same UMI
    # quartile shouldn't end up colocated in the embedding if biology
    # is what's being captured.
    quartiles = np.digitize(n_counts, np.quantile(n_counts, [0.25, 0.5, 0.75]))

    embeddings: dict[str, np.ndarray] = {}
    metrics: dict[str, tuple[float, float]] = {}
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        silhouette_score = None  # type: ignore[assignment]

    for name, fn in (("PCA", fit_pca), ("NMF", fit_nmf), ("sparseNMF", fit_sparse_nmf)):
        t0 = time.time()
        high = fn(X, seed, k)
        z = umap_project(high, seed)
        embeddings[name] = z
        if silhouette_score is not None:
            sg = float(silhouette_score(z, labels))
            sb = float(silhouette_score(z, quartiles))
        else:
            sg, sb = float("nan"), float("nan")
        metrics[name] = (sg, sb)
        print(
            f"  {name:>10s}: {time.time() - t0:5.1f}s  "
            f"silhouette(cell-type)={sg:+.3f}  silhouette(depth-quartile)={sb:+.3f}"
        )

    out = Path(__file__).resolve().parents[1] / "docs" / "_static" / "real_pbmc3k_demo.png"
    make_figure(embeddings, metrics, labels, log_depth, out)
    print(f"  wrote {out.relative_to(out.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
