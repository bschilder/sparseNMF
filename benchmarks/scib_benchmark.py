"""scIB-style integration benchmark module.

Self-contained helpers for benchmarking sparseNMF against the
standard scRNA-seq integration methods on the scIB benchmark
datasets (Luecken et al. 2022, Nature Methods, DOI
10.1038/s41592-021-01336-8).

Methods covered:

- **PCA** (no batch correction; baseline)
- **NMF** (sklearn; no batch correction)
- **sparseNMF** (this package; defaults: normalize_inputs=True)
- **sparseNMF + nonzero_mse_weight=1.0** (MSE only on non-zero entries)
- **Harmony** (Korsunsky 2019, harmonypy Python port)
- **scVI** (Lopez 2018, scvi-tools)

Each method produces an embedding in ``adata.obsm["X_emb"]``; we then
compute the scIB metric suite via the ``scib`` package and aggregate
into a results table.

The full immune dataset (~33k cells) is the compute bottleneck:
scVI on CPU takes ~30 min; on a single GPU ~2 min. Other methods
complete in seconds-to-minutes on CPU.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, issparse


# ── Datasets ─────────────────────────────────────────────────────────

SCIB_DATASETS = {
    "pancreas": {
        "url": "https://ndownloader.figshare.com/files/24539828",
        "filename": "human_pancreas_norm_complexBatch.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "counts_layer": "counts",
    },
    "immune": {
        "url": "https://ndownloader.figshare.com/files/25717328",
        "filename": "Immune_ALL_human.h5ad",
        "batch_key": "batch",
        "label_key": "final_annotation",
        "counts_layer": "counts",
    },
}


def _cache_dir() -> Path:
    p = Path.home() / ".cache" / "sparse-nmf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url} → {dest}")
    last = -10

    def _hook(block: int, size: int, total: int) -> None:
        nonlocal last
        if total <= 0:
            return
        pct = min(int(100 * block * size / total), 100)
        if pct - last >= 10:
            print(f"  ... {pct}% ({block * size / 1e6:.0f} / {total / 1e6:.0f} MB)")
            last = pct

    urllib.request.urlretrieve(url, dest, _hook)


def load_scib_dataset(name: str, *, cells_per_cohort: int | None = None, seed: int = 0):
    """Return (adata, batch_key, label_key) for a named scIB dataset.

    If ``cells_per_cohort`` is set, stratified-subsample by
    (batch × label) so the benchmark is CPU-runnable. Pass ``None``
    to use the full dataset (recommended on GPU).
    """
    import anndata as ad

    spec = SCIB_DATASETS[name]
    cache = _cache_dir() / spec["filename"]
    if not cache.exists():
        _download_with_progress(spec["url"], cache)
    adata = ad.read_h5ad(cache)

    if cells_per_cohort is not None:
        rng = np.random.default_rng(seed)
        batch = adata.obs[spec["batch_key"]].astype(str).values
        label = adata.obs[spec["label_key"]].astype(str).values
        keep: list[np.ndarray] = []
        for b in np.unique(batch):
            for ll in np.unique(label):
                idx = np.where((batch == b) & (label == ll))[0]
                if idx.size == 0:
                    continue
                keep.append(rng.choice(idx, size=min(cells_per_cohort, idx.size), replace=False))
        adata = adata[np.sort(np.concatenate(keep))].copy()
    return adata, spec["batch_key"], spec["label_key"], spec["counts_layer"]


def _counts(adata, layer: str) -> csr_matrix:
    """Return raw counts as a CSR matrix from ``adata.layers[layer]``."""
    X = adata.layers[layer]
    if not issparse(X):
        X = csr_matrix(X)
    return X.tocsr().astype(np.float32)


# ── Methods (each returns an (n_cells, k) embedding) ────────────────


def embed_pca(adata, batch_key, label_key, counts_layer, k, seed):
    from sklearn.decomposition import PCA

    X = _counts(adata, counts_layer).toarray()
    return PCA(n_components=k, random_state=seed).fit_transform(X)


def embed_nmf(adata, batch_key, label_key, counts_layer, k, seed):
    from sklearn.decomposition import NMF

    X = _counts(adata, counts_layer)
    return NMF(n_components=k, init="nndsvd", max_iter=500, random_state=seed).fit_transform(X)


def embed_sparse_nmf(adata, batch_key, label_key, counts_layer, k, seed, **kwargs):
    from sparse_nmf import train_sparse_nmf

    X = _counts(adata, counts_layer)
    W, _ = train_sparse_nmf(
        X_sparse=X, n_components=k, device="cpu", random_state=seed, verbose=False, **kwargs
    )
    return W


def embed_sparse_nmf_nonzero(adata, batch_key, label_key, counts_layer, k, seed):
    return embed_sparse_nmf(
        adata, batch_key, label_key, counts_layer, k, seed,
        mse_weight=0.0, nonzero_mse_weight=1.0,
    )


def embed_harmony(adata, batch_key, label_key, counts_layer, k, seed):
    """PCA(k) then Harmony correction in PC space."""
    import harmonypy as hm
    from sklearn.decomposition import PCA

    X = _counts(adata, counts_layer).toarray()
    pca = PCA(n_components=k, random_state=seed).fit_transform(X)
    ho = hm.run_harmony(pca, adata.obs, batch_key, max_iter_harmony=20)
    return np.asarray(ho.Z_corr).T


def embed_scvi(adata, batch_key, label_key, counts_layer, k, seed):
    """scVI VAE — uses raw counts directly. ``k`` is mapped to
    ``n_latent``. On GPU set ``accelerator='gpu'`` via env var (see
    runner)."""
    import scvi

    a = adata.copy()
    a.X = a.layers[counts_layer]  # scVI wants counts in .X by convention
    scvi.model.SCVI.setup_anndata(a, batch_key=batch_key)
    model = scvi.model.SCVI(a, n_latent=k)
    model.train(max_epochs=100, accelerator="auto", devices=1, plan_kwargs={"lr": 1e-3})
    return model.get_latent_representation()


METHODS: dict[str, Callable] = {
    "PCA": embed_pca,
    "NMF": embed_nmf,
    "sparseNMF": embed_sparse_nmf,
    "sparseNMF+nonzero": embed_sparse_nmf_nonzero,
    "Harmony": embed_harmony,
    "scVI": embed_scvi,
}


# ── Metrics ──────────────────────────────────────────────────────────


@dataclass
class MethodResult:
    name: str
    seconds: float
    metrics: dict[str, float]


def evaluate(
    adata,
    embedding: np.ndarray,
    batch_key: str,
    label_key: str,
    *,
    lisi: bool = True,
) -> dict[str, float]:
    """Compute the scIB metric suite for one embedding.

    Wraps ``scib.metrics.metrics`` with sensible defaults. The
    R-requiring kBET / PCR / HVG / cell-cycle / trajectory metrics
    are off — they need ancillary inputs (counts reference, cycle
    genes) and we want the runner method-agnostic.

    ``lisi`` toggles the iLISI/cLISI graph LISI metrics. scib ships
    these as a pre-compiled C binary that's x86_64-only — disable on
    Apple Silicon (arm64); re-enable on Linux x86_64 (e.g., a RunPod
    pod). Without LISI the remaining bio metrics are NMI/ARI/ASW/
    isolated-label F1/ASW; the only batch metric is graph
    connectivity.
    """
    import scanpy as sc
    import scib

    a = adata.copy()
    a.obsm["X_emb"] = embedding
    sc.pp.neighbors(a, use_rep="X_emb")
    scib.metrics.cluster_optimal_resolution(a, cluster_key="cluster", label_key=label_key)

    res = scib.metrics.metrics(
        adata, a,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_emb",
        cluster_key="cluster",
        nmi_=True, ari_=True, silhouette_=True,
        isolated_labels_f1_=True, isolated_labels_asw_=True,
        graph_conn_=True,
        clisi_=lisi, ilisi_=lisi,
        kBET_=False, pcr_=False, hvg_score_=False, cell_cycle_=False,
        trajectory_=False,
    )
    return {k: float(v) for k, v in res.iloc[:, 0].dropna().items()}


def composite_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Return (bio, batch, composite=0.4*batch + 0.6*bio) per scIB."""
    bio_keys = {"NMI_cluster/label", "ARI_cluster/label", "ASW_label",
                "isolated_label_F1", "isolated_label_silhouette", "cLISI"}
    batch_keys = {"graph_conn", "iLISI"}
    bio_vals = [metrics[k] for k in bio_keys if k in metrics]
    batch_vals = [metrics[k] for k in batch_keys if k in metrics]
    bio = float(np.mean(bio_vals)) if bio_vals else float("nan")
    batch = float(np.mean(batch_vals)) if batch_vals else float("nan")
    return bio, batch, 0.6 * bio + 0.4 * batch


# ── Runner ───────────────────────────────────────────────────────────


def run_dataset(
    name: str,
    *,
    methods: list[str] | None = None,
    cells_per_cohort: int | None = 50,
    k: int = 30,
    seed: int = 0,
    lisi: bool = True,
) -> list[MethodResult]:
    """Run all methods on a dataset, returning per-method results."""
    methods = methods or list(METHODS)
    print(f"\n=== {name} ===")
    adata, batch_key, label_key, counts_layer = load_scib_dataset(
        name, cells_per_cohort=cells_per_cohort, seed=seed,
    )
    print(
        f"  {adata.shape}  batches={adata.obs[batch_key].nunique()}  "
        f"labels={adata.obs[label_key].nunique()}"
    )

    results: list[MethodResult] = []
    for method_name in methods:
        fn = METHODS[method_name]
        print(f"  {method_name:>18s}: fitting...", flush=True)
        t0 = time.time()
        try:
            emb = fn(adata, batch_key, label_key, counts_layer, k, seed)
            metrics = evaluate(adata, emb, batch_key, label_key, lisi=lisi)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            results.append(MethodResult(method_name, time.time() - t0, {"error": -1.0}))
            continue
        bio, batch, composite = composite_score(metrics)
        metrics["_bio"] = bio
        metrics["_batch"] = batch
        metrics["_composite"] = composite
        results.append(MethodResult(method_name, elapsed, metrics))
        print(
            f"    {method_name:>18s}: {elapsed:6.1f}s  "
            f"bio={bio:+.3f}  batch={batch:+.3f}  composite={composite:+.3f}"
        )
    return results


def results_to_dataframe(results_by_dataset: dict[str, list[MethodResult]]):
    """Tidy DataFrame: one row per (dataset, method)."""
    import pandas as pd

    rows = []
    for dataset, results in results_by_dataset.items():
        for r in results:
            row = {"dataset": dataset, "method": r.name, "seconds": r.seconds, **r.metrics}
            rows.append(row)
    return pd.DataFrame(rows)
