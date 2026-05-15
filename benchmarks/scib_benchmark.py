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

import resource
import sys
import time
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse

# ── Timing + memory instrumentation ────────────────────────────────


@dataclass
class MethodTiming:
    """Per-method resource usage. All durations in seconds."""

    fit_seconds: float
    infer_seconds: float | None  # None when fit and infer can't be separated
    metric_seconds: float = 0.0
    peak_rss_mb: float = 0.0  # delta RSS over the method's call
    gpu_peak_mb: float | None = None  # peak CUDA allocator usage, if applicable

    def total_seconds(self) -> float:
        return self.fit_seconds + (self.infer_seconds or 0.0) + self.metric_seconds


@contextmanager
def _track_memory():
    """Yield a dict that gets populated with peak RSS / GPU memory deltas
    over the contextmanager's lifetime. RSS is in MB; GPU is in MB.

    ``ru_maxrss`` is the *high-water mark since process start*, so we
    snapshot before and after and report the delta — what the method
    itself added on top of pre-existing memory."""
    out: dict[str, float | None] = {"peak_rss_mb": 0.0, "gpu_peak_mb": None}
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gpu_before = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_before = 0
    except ImportError:
        pass
    try:
        yield out
    finally:
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss unit is platform-specific: bytes on macOS, kibibytes
        # on Linux. Normalize to MiB.
        if sys.platform == "darwin":
            out["peak_rss_mb"] = max(0.0, (rss_after - rss_before) / (1024.0 * 1024.0))
        else:
            out["peak_rss_mb"] = max(0.0, (rss_after - rss_before) / 1024.0)
        try:
            import torch

            if gpu_before is not None and torch.cuda.is_available():
                out["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        except ImportError:
            pass


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


def load_scib_dataset(
    name: str,
    *,
    cells_per_cohort: int | None = None,
    seed: int = 0,
    hvg: bool = True,
    n_hvg: int = 2000,
):
    """Return (adata, batch_key, label_key, counts_layer) for a named
    scIB dataset.

    If ``cells_per_cohort`` is set, stratified-subsample by
    (batch × label) so the benchmark is CPU-runnable. Pass ``None``
    to use the full dataset (recommended on GPU).

    If ``hvg=True`` (default), runs ``scib.preprocessing.hvg_batch`` to
    select the top ``n_hvg`` highly variable genes per batch (Cell
    Ranger flavor, n_bins=20). This is the scIB-paper-canonical
    preprocessing — every method then operates on the same 2000-gene
    matrix instead of the full ~20k genes. Massively faster *and*
    matches what the published scIB benchmark actually used.
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

    if hvg:
        # Per scIB methods: hvg_batch picks top-n_hvg per batch by
        # cell_ranger flavor, then ranks first by # batches each gene is
        # HV in, then by mean dispersion across batches. Run on log1p
        # (.X is scran-log1p in scIB-published files).
        import scib.preprocessing as scib_pp

        adata = scib_pp.hvg_batch(
            adata,
            batch_key=spec["batch_key"],
            target_genes=n_hvg,
            flavor="cell_ranger",
            n_bins=20,
            adataOut=True,
        )

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
    with _track_memory() as mem:
        pca = PCA(n_components=k, random_state=seed)
        t0 = time.perf_counter()
        pca.fit(X)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb = pca.transform(X)
        inf_s = time.perf_counter() - t1
    return emb, MethodTiming(fit_s, inf_s, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def embed_nmf(adata, batch_key, label_key, counts_layer, k, seed):
    from sklearn.decomposition import NMF

    X = _counts(adata, counts_layer)
    with _track_memory() as mem:
        nmf = NMF(n_components=k, init="nndsvd", max_iter=500, random_state=seed)
        t0 = time.perf_counter()
        nmf.fit(X)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb = nmf.transform(X)
        inf_s = time.perf_counter() - t1
    return emb, MethodTiming(fit_s, inf_s, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def embed_sparse_nmf(adata, batch_key, label_key, counts_layer, k, seed, **kwargs):
    from sparse_nmf import train_sparse_nmf

    # sparseNMF is GPU-native — pick cuda when available so the
    # benchmark exercises the package as it's actually intended to be
    # used in production. Falls back to CPU silently if no CUDA.
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    X = _counts(adata, counts_layer)
    with _track_memory() as mem:
        # train_sparse_nmf returns W (the embedding) directly — fit and
        # initial transform are fused. No separable infer step for this
        # configuration.
        t0 = time.perf_counter()
        W, _ = train_sparse_nmf(
            X_sparse=X, n_components=k, device=device, random_state=seed,
            verbose=False, **kwargs,
        )
        fit_s = time.perf_counter() - t0
    return W, MethodTiming(fit_s, None, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def embed_sparse_nmf_nonzero(adata, batch_key, label_key, counts_layer, k, seed):
    # The gradient-descent path (triggered by nonzero_mse_weight > 0)
    # has a CUDA memory issue at this k on a 16 GB A4000 *and* doesn't
    # keep the workload on the GPU as cleanly as the MU path. Forcing
    # device='cpu' here makes it finish reliably; it's slower per
    # iteration but doesn't OOM and produces complete results.
    from sparse_nmf import train_sparse_nmf

    X = _counts(adata, counts_layer)
    with _track_memory() as mem:
        t0 = time.perf_counter()
        W, _ = train_sparse_nmf(
            X_sparse=X, n_components=k, device="cpu", random_state=seed,
            verbose=False,
            mse_weight=0.0, nonzero_mse_weight=1.0,
        )
        fit_s = time.perf_counter() - t0
    return W, MethodTiming(fit_s, None, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def embed_harmony(adata, batch_key, label_key, counts_layer, k, seed):
    """PCA(k) then Harmony correction in PC space. We split the timing
    so the PCA stage isn't lumped into Harmony's runtime — Harmony's
    cost is the iterative correction in PC space, not the PCA itself."""
    import harmonypy as hm
    from sklearn.decomposition import PCA

    X = _counts(adata, counts_layer).toarray()
    with _track_memory() as mem:
        t0 = time.perf_counter()
        pca = PCA(n_components=k, random_state=seed).fit_transform(X)
        pca_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        ho = hm.run_harmony(pca, adata.obs, batch_key, max_iter_harmony=20)
        harm_s = time.perf_counter() - t1
    # harmonypy 2.0 returns Z_corr shaped (n_cells, k) directly; the
    # transpose in earlier versions is no longer needed. (Pre-2.0 the
    # shape was (k, n_cells) and we did .T to fix it; the API changed
    # between versions.) Pinning to >=2.0 in the install side.
    emb = np.asarray(ho.Z_corr)
    if emb.shape[0] != adata.n_obs:
        # Defensive: if some future harmonypy reverts the shape, fix it.
        emb = emb.T
    return emb, MethodTiming(pca_s + harm_s, None, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def embed_scvi(adata, batch_key, label_key, counts_layer, k, seed):
    """scVI VAE — uses raw counts directly. ``k`` maps to
    ``n_latent``. We measure training time and inference time
    (``get_latent_representation``) separately — these are typically
    *very* different (train: minutes; infer: seconds) and the split
    matters for downstream cost forecasting."""
    import scvi

    a = adata.copy()
    a.X = a.layers[counts_layer]  # scVI wants counts in .X by convention
    with _track_memory() as mem:
        scvi.model.SCVI.setup_anndata(a, batch_key=batch_key)
        model = scvi.model.SCVI(a, n_latent=k)
        t0 = time.perf_counter()
        model.train(max_epochs=100, accelerator="auto", devices=1, plan_kwargs={"lr": 1e-3})
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb = model.get_latent_representation()
        inf_s = time.perf_counter() - t1
    return emb, MethodTiming(fit_s, inf_s, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


# Methods active in the default benchmark. ``sparseNMF+nonzero`` is
# defined above but kept out of the default set because the gradient-
# descent path it triggers is much slower than the MU path (10-100x)
# and not yet GPU-efficient at k=30 — it OOMs on a 16 GB GPU and is
# CPU-bound elsewhere. Opt-in via ``--methods sparseNMF+nonzero`` if
# you want to benchmark it specifically.
METHODS: dict[str, Callable] = {
    "PCA": embed_pca,
    "NMF": embed_nmf,
    "sparseNMF": embed_sparse_nmf,
    "Harmony": embed_harmony,
    "scVI": embed_scvi,
}

# Available-but-not-in-default; addressable via --methods.
EXTRA_METHODS: dict[str, Callable] = {
    "sparseNMF+nonzero": embed_sparse_nmf_nonzero,
}
METHODS_ALL: dict[str, Callable] = {**METHODS, **EXTRA_METHODS}


# ── Metrics ──────────────────────────────────────────────────────────


@dataclass
class MethodResult:
    name: str
    timing: MethodTiming
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def total_seconds(self) -> float:
        return self.timing.total_seconds()


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
    hvg: bool = True,
    n_hvg: int = 2000,
) -> list[MethodResult]:
    """Run all methods on a dataset, returning per-method results."""
    methods = methods or list(METHODS)
    print(f"\n=== {name} ===")
    adata, batch_key, label_key, counts_layer = load_scib_dataset(
        name, cells_per_cohort=cells_per_cohort, seed=seed,
        hvg=hvg, n_hvg=n_hvg,
    )
    print(
        f"  {adata.shape}  batches={adata.obs[batch_key].nunique()}  "
        f"labels={adata.obs[label_key].nunique()}"
    )

    results: list[MethodResult] = []
    for method_name in methods:
        fn = METHODS_ALL[method_name]
        print(f"  {method_name:>18s}: fitting...", flush=True)
        try:
            emb, timing = fn(adata, batch_key, label_key, counts_layer, k, seed)
            t0 = time.perf_counter()
            metrics = evaluate(adata, emb, batch_key, label_key, lisi=lisi)
            timing.metric_seconds = time.perf_counter() - t0
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {err}")
            results.append(MethodResult(
                method_name,
                MethodTiming(0.0, None),
                error=err,
            ))
            continue
        bio, batch, composite = composite_score(metrics)
        metrics["_bio"] = bio
        metrics["_batch"] = batch
        metrics["_composite"] = composite
        results.append(MethodResult(method_name, timing, metrics))
        infer_str = f"infer={timing.infer_seconds:5.1f}s" if timing.infer_seconds else "infer=  N/A"
        gpu_str = f"gpu={timing.gpu_peak_mb:6.0f}MB" if timing.gpu_peak_mb else "gpu=     —"
        print(
            f"    {method_name:>18s}: "
            f"fit={timing.fit_seconds:6.1f}s  {infer_str}  "
            f"metrics={timing.metric_seconds:5.1f}s  "
            f"rss={timing.peak_rss_mb:5.0f}MB  {gpu_str}  "
            f"bio={bio:+.3f}  batch={batch:+.3f}  composite={composite:+.3f}"
        )
    return results


def results_to_dataframe(results_by_dataset: dict[str, list[MethodResult]]):
    """Tidy DataFrame: one row per (dataset, method)."""
    import pandas as pd

    rows = []
    for dataset, results in results_by_dataset.items():
        for r in results:
            row = {
                "dataset": dataset,
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
            rows.append(row)
    return pd.DataFrame(rows)
