"""Demo: why sparseNMF beats raw PCA / NMF when batches differ in sparsity.

Constructs a synthetic gene-by-cell matrix in which **biological signal**
(three groups, each enriched on its own gene-program block) is identical
across two batches, but the **sparsity signature** (per-cell non-zero
gene count) differs sharply between batches. This is the canonical
"library-depth confound" in single-cell data: cells captured by a
shallow protocol have ~5× fewer detected genes than cells from a deep
protocol, yet the underlying biology is the same.

Three 2-D embeddings are then compared:

* **PCA(2)** on the raw matrix.
* **NMF(2)** (sklearn) on the raw matrix.
* **sparseNMF** with ``normalize_inputs=True`` — each row of ``X`` is
  L2-normalized *before* the multiplicative updates, so the dominant
  per-row magnitude (library depth) is quotiented out at the input
  stage. NMF then factorizes the *direction* of expression, which is
  what's actually shared across batches of the same biology.

The resulting figure is a 2×3 facet of scatter plots: rows are
*color-by-biological-group* (top) and *color-by-non-zero-gene-count*
(bottom). The story:

* PCA & NMF top row → groups smear across batches (poor mixing).
* PCA & NMF bottom row → strong nnz gradient ⇒ they're tracking
  library depth, not biology.
* sparseNMF top row → groups cluster cleanly, mixed across batches.
* sparseNMF bottom row → no nnz gradient ⇒ the sparsity confound is
  gone.

Run from repo root::

    python examples/sparsity_confound_demo.py

Writes ``docs/_static/sparsity_confound_demo.png``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from sparse_nmf import train_sparse_nmf


def make_sparsity_confound_data(
    n_per_cohort: int = 100,
    n_groups: int = 3,
    n_features: int = 900,
    p_keep_low: float = 0.05,
    p_keep_high: float = 0.50,
    seed: int = 0,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Synthesize the gene × cell matrix with shared biology / split sparsity.

    Each of ``n_groups`` biological groups gets a disjoint block of genes
    it is enriched on. Cells in a group share the same loading template,
    so the *signal* is identical across batches. Then each cell is
    assigned to one of two batches that differ only in their Bernoulli
    keep-rate over genes (value-independent dropout). Magnitude per cell
    is otherwise constant.

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (n_cells, n_features)
    group_labels : np.ndarray of int, shape (n_cells,)
    batch_labels : np.ndarray of int, shape (n_cells,)  -- 0 (low) or 1 (high)
    """
    rng = np.random.default_rng(seed)

    # Disjoint gene-program blocks, one per group. Inside the block each
    # gene gets a positive loading; outside it loads weakly.
    block = n_features // n_groups
    H = np.full((n_groups, n_features), 0.05, dtype=np.float32)
    for k in range(n_groups):
        H[k, k * block : (k + 1) * block] = rng.uniform(1.0, 2.0, block).astype(np.float32)

    # 2 batches per group -> 6 cohorts total. Cells inside a (group,
    # batch) cohort share group identity but get the batch's keep-rate.
    rows, cols, vals, groups, batches = [], [], [], [], []
    row_idx = 0
    for k in range(n_groups):
        # Loading template: cell loads heavily on its group's component,
        # weakly on the others. Identical for both batches of the group.
        for batch in (0, 1):
            p_keep = p_keep_low if batch == 0 else p_keep_high
            for _ in range(n_per_cohort):
                w = rng.gamma(shape=2.0, scale=1.0, size=n_groups).astype(np.float32) * 0.2
                w[k] += rng.gamma(shape=5.0, scale=1.0)  # dominant loading
                mean = w @ H  # shape (n_features,)
                counts = rng.poisson(mean).astype(np.float32)
                mask = rng.random(n_features) < p_keep
                counts *= mask
                nz = np.flatnonzero(counts)
                rows.extend([row_idx] * nz.size)
                cols.extend(nz.tolist())
                vals.extend(counts[nz].tolist())
                groups.append(k)
                batches.append(batch)
                row_idx += 1

    n_cells = row_idx
    X = csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_features),
    )
    return X, np.asarray(groups), np.asarray(batches)


def _two_d(emb: np.ndarray) -> np.ndarray:
    """Coerce an (n, k>=2) embedding to (n, 2) by selecting the top-2
    columns by variance — needed because sklearn NMF with ``n_components=2``
    can be axis-aligned but we want consistent orientation."""
    if emb.shape[1] == 2:
        return emb
    order = np.argsort(emb.var(axis=0))[::-1][:2]
    return emb[:, order]


def fit_pca(X: csr_matrix, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(X.toarray())


def fit_nmf(X: csr_matrix, seed: int) -> np.ndarray:
    from sklearn.decomposition import NMF

    return NMF(n_components=2, init="nndsvd", max_iter=500, random_state=seed).fit_transform(X)


def fit_sparse_nmf(X: csr_matrix, seed: int) -> np.ndarray:
    # ``normalize_inputs=True`` is the load-bearing knob: each cell's
    # expression vector is L2-normalized before the multiplicative
    # updates, so factorization happens in *direction* space — library
    # depth (the per-row magnitude axis) is gone before NMF even starts.
    W, _model = train_sparse_nmf(
        X_sparse=X,
        n_components=3,
        max_iter=300,
        device="cpu",
        normalize_inputs=True,
        random_state=seed,
        verbose=False,
    )
    return W


def make_figure(
    embeddings: dict[str, np.ndarray],
    metrics: dict[str, tuple[float, float]],
    groups: np.ndarray,
    batches: np.ndarray,
    nnz: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    methods = list(embeddings)
    n = len(methods)
    # Add an extra slim column for a shared colorbar so the rightmost
    # scatter doesn't get squashed.
    fig, axes = plt.subplots(
        2,
        n + 1,
        figsize=(3.7 * n + 0.6, 7.2),
        gridspec_kw={"width_ratios": [1.0] * n + [0.06]},
    )

    group_cmap = ListedColormap(["#e6194B", "#3cb44b", "#4363d8"])
    batch_markers = {0: "o", 1: "^"}
    sc_nnz = None

    for col, name in enumerate(methods):
        z = embeddings[name]
        sg, sb = metrics[name]

        # Row 0: colored by biological group, marker = batch.
        ax = axes[0, col]
        for b, m in batch_markers.items():
            mask = batches == b
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                c=groups[mask],
                cmap=group_cmap,
                vmin=0,
                vmax=2,
                marker=m,
                s=18,
                alpha=0.85,
                linewidth=0,
            )
        ax.set_title(f"{name}\nsilhouette: group={sg:+.2f}  batch={sb:+.2f}", fontsize=11)
        if col == 0:
            ax.set_ylabel("colored by\nbiological group", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 1: colored by per-cell nnz (sparsity signature).
        ax = axes[1, col]
        sc_nnz = ax.scatter(z[:, 0], z[:, 1], c=nnz, cmap="viridis", s=18, alpha=0.85, linewidth=0)
        if col == 0:
            ax.set_ylabel("colored by\nnon-zero gene count", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide the spare top-right axis; use the bottom one for the shared
    # nnz colorbar.
    axes[0, -1].axis("off")
    cbar = fig.colorbar(sc_nnz, cax=axes[1, -1])
    cbar.set_label("nnz / cell", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=group_cmap(i),
            markersize=8,
            linestyle="",
            label=f"group {i + 1}",
        )
        for i in range(3)
    ] + [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="#888",
            markersize=8,
            linestyle="",
            label="batch A (low nnz)",
        ),
        plt.Line2D(
            [],
            [],
            marker="^",
            color="w",
            markerfacecolor="#888",
            markersize=8,
            linestyle="",
            label="batch B (high nnz)",
        ),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8, loc="best", frameon=False)

    fig.suptitle(
        "Biology is identical across batches — only sparsity differs.\n"
        "PCA / NMF lock onto the nnz axis; sparseNMF recovers groups.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    seed = 0
    print("Generating synthetic data (3 groups × 2 sparsity-batches)...")
    X, groups, batches = make_sparsity_confound_data(seed=seed)
    nnz = np.asarray((X != 0).sum(axis=1)).ravel()
    print(
        f"  X shape={X.shape}  nnz/cell: batch_low mean={nnz[batches == 0].mean():.0f}, "
        f"batch_high mean={nnz[batches == 1].mean():.0f}"
    )

    embeddings: dict[str, np.ndarray] = {}
    metrics: dict[str, tuple[float, float]] = {}
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:  # silhouette is best-effort; figure works without it
        silhouette_score = None  # type: ignore[assignment]

    for name, fn in (("PCA", fit_pca), ("NMF", fit_nmf), ("sparseNMF", fit_sparse_nmf)):
        t0 = time.time()
        z = _two_d(fn(X, seed))
        embeddings[name] = z
        if silhouette_score is not None:
            sg = float(silhouette_score(z, groups))
            sb = float(silhouette_score(z, batches))
        else:
            sg, sb = float("nan"), float("nan")
        metrics[name] = (sg, sb)
        print(
            f"  {name:>10s}: {time.time() - t0:4.1f}s  "
            f"silhouette(group)={sg:+.3f}  silhouette(batch)={sb:+.3f}"
        )

    out = Path(__file__).resolve().parents[1] / "docs" / "_static" / "sparsity_confound_demo.png"
    make_figure(embeddings, metrics, groups, batches, nnz, out)
    print(f"  wrote {out.relative_to(out.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
