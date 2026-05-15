"""Shared I/O + dataset helpers for the subprocess-isolated
benchmark.

Each method runs in its own subprocess. The subprocess produces
exactly two artifacts on disk, in a per-(dataset, method)
directory:

- ``X_emb.npz`` — the (n_cells, k) embedding, plus the cell-order
  fingerprint so downstream metrics can sanity-check alignment with
  the adata they re-load.
- ``timing.json`` — fit/infer/metric seconds, peak RSS, peak GPU
  memory, plus the dataset metadata the metrics step needs
  (batch_key, label_key) so it doesn't have to re-derive it.

If a method fails it writes ``error.txt`` instead of ``X_emb.npz``.
The metrics step skips entries without an embedding.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


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


def cache_dir() -> Path:
    p = Path.home() / ".cache" / "sparse-nmf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url} → {dest}", flush=True)
    last = -10

    def _hook(block: int, size: int, total: int) -> None:
        nonlocal last
        if total <= 0:
            return
        pct = min(int(100 * block * size / total), 100)
        if pct - last >= 10:
            print(f"  ... {pct}% ({block * size / 1e6:.0f} / {total / 1e6:.0f} MB)", flush=True)
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
    scIB dataset. Lives in its own module so each subprocess can call
    it without pulling in the rest of the legacy benchmark module.
    """
    import anndata as ad

    spec = SCIB_DATASETS[name]
    cache = cache_dir() / spec["filename"]
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
        import scanpy as sc

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            batch_key=spec["batch_key"],
            flavor="cell_ranger",
            n_bins=20,
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata, spec["batch_key"], spec["label_key"], spec["counts_layer"]


def adata_fingerprint(adata) -> str:
    """Stable hash of the cell-order + dataset shape. The metrics step
    re-loads the adata and computes this; if the fingerprint doesn't
    match what the embedding was produced from, we refuse to score —
    catches silent miscompare bugs where a subprocess used different
    HVG / subsample parameters than the metrics pass."""
    # adata.obs_names is a pandas Index of Python strings; converting
    # via np.asarray gives an object-dtype array whose tobytes() returns
    # pointer values (different every process). Hash the actual string
    # content via a unicode dtype + a separator-joined fallback for safety.
    obs_names = np.asarray(adata.obs_names, dtype=str)
    h = hashlib.sha256()
    h.update(str(adata.shape).encode())
    h.update(obs_names.tobytes())
    return h.hexdigest()[:16]


# ── Per-method input routing ────────────────────────────────────────


def counts(adata, layer: str):
    """Return raw counts as a CSR matrix from ``adata.layers[layer]``."""
    from scipy.sparse import csr_matrix, issparse

    X = adata.layers[layer]
    if not issparse(X):
        X = csr_matrix(X)
    return X.tocsr().astype(np.float32)


def lognorm_X(adata) -> np.ndarray:
    """Return adata.X (log1p-scran-norm in scIB files) as a dense
    float32 array. Non-negative."""
    from scipy.sparse import issparse

    X = adata.X
    if issparse(X):
        X = X.toarray()
    return X.astype(np.float32)


def scaled_X(adata, batch_key: str) -> np.ndarray:
    """Zero-center + unit-variance scaling of the log1p .X.
    Used for PCA / Harmony. May contain negatives — not for NMF."""
    import scanpy as sc
    from scipy.sparse import issparse

    a = adata.copy()
    sc.pp.scale(a, max_value=10)
    X = a.X
    if issparse(X):
        X = X.toarray()
    return X.astype(np.float32)


# ── Timing + memory instrumentation ────────────────────────────────


@dataclass
class MethodTiming:
    """Per-method resource usage. All durations in seconds."""

    fit_seconds: float
    infer_seconds: float | None = None
    metric_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    gpu_peak_mb: float | None = None

    def total_seconds(self) -> float:
        return self.fit_seconds + (self.infer_seconds or 0.0) + self.metric_seconds


@contextmanager
def track_memory() -> Iterator[dict]:
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


# ── Embedding / timing I/O ──────────────────────────────────────────


def method_out_dir(out_root: Path | str, dataset: str, method: str) -> Path:
    p = Path(out_root) / dataset / method
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_embedding(out_root: Path | str, dataset: str, method: str, emb: np.ndarray, fingerprint: str) -> Path:
    """Write embedding as .npz (allow_pickle=False on load) — never pickle."""
    d = method_out_dir(out_root, dataset, method)
    path = d / "X_emb.npz"
    np.savez_compressed(path, X_emb=np.asarray(emb), fingerprint=np.array(fingerprint))
    return path


def load_embedding(out_root: Path | str, dataset: str, method: str) -> tuple[np.ndarray, str]:
    path = Path(out_root) / dataset / method / "X_emb.npz"
    z = np.load(path, allow_pickle=False)
    return z["X_emb"], str(z["fingerprint"])


def save_timing(
    out_root: Path | str,
    dataset: str,
    method: str,
    timing: MethodTiming,
    *,
    batch_key: str,
    label_key: str,
    fingerprint: str,
) -> Path:
    d = method_out_dir(out_root, dataset, method)
    path = d / "timing.json"
    payload = {
        **asdict(timing),
        "dataset": dataset,
        "method": method,
        "batch_key": batch_key,
        "label_key": label_key,
        "fingerprint": fingerprint,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_timing(out_root: Path | str, dataset: str, method: str) -> dict | None:
    path = Path(out_root) / dataset / method / "timing.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_error(out_root: Path | str, dataset: str, method: str, msg: str) -> Path:
    d = method_out_dir(out_root, dataset, method)
    path = d / "error.txt"
    path.write_text(msg)
    return path


def load_error(out_root: Path | str, dataset: str, method: str) -> str | None:
    path = Path(out_root) / dataset / method / "error.txt"
    if not path.exists():
        return None
    return path.read_text()


# ── Common __main__ scaffolding for method subprocesses ─────────────


def add_common_method_args(parser):
    """Register the args every per-method subprocess needs."""
    parser.add_argument("--dataset", required=True, choices=list(SCIB_DATASETS))
    parser.add_argument("--out-dir", required=True,
                        help="Run root directory; method writes to <out-dir>/<dataset>/<method>/")
    parser.add_argument("--method-name", required=True,
                        help="Method name as it appears in the results table")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells-per-cohort", type=int, default=None,
                        help="Stratified per-(batch,label) subsample size; omit for full dataset.")
    parser.add_argument("--no-hvg", action="store_true")
    parser.add_argument("--n-hvg", type=int, default=2000)


def run_method_subprocess(args, embed_fn) -> int:
    """Standard wiring: load dataset → embed → save artifacts.

    ``embed_fn`` has signature
    ``(adata, batch_key, label_key, counts_layer, k, seed) -> (emb, MethodTiming)``.
    """
    method = args.method_name
    print(f"  {method}: loading {args.dataset}...", flush=True)
    adata, batch_key, label_key, counts_layer = load_scib_dataset(
        args.dataset,
        cells_per_cohort=args.cells_per_cohort,
        seed=args.seed,
        hvg=not args.no_hvg,
        n_hvg=args.n_hvg,
    )
    fp = adata_fingerprint(adata)
    print(f"  {method}: {adata.shape}  fp={fp}  batches={adata.obs[batch_key].nunique()}  "
          f"labels={adata.obs[label_key].nunique()}", flush=True)

    print(f"  {method}: fitting...", flush=True)
    try:
        t0 = time.perf_counter()
        emb, timing = embed_fn(adata, batch_key, label_key, counts_layer, args.k, args.seed)
        wall = time.perf_counter() - t0
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"    FAILED: {msg[:200]}", flush=True)
        save_error(args.out_dir, args.dataset, method, msg)
        return 1

    save_embedding(args.out_dir, args.dataset, method, emb, fingerprint=fp)
    save_timing(args.out_dir, args.dataset, method, timing,
                batch_key=batch_key, label_key=label_key, fingerprint=fp)
    infer = f"{timing.infer_seconds:.2f}s" if timing.infer_seconds is not None else "—"
    gpu = f"{timing.gpu_peak_mb:.0f}MB" if timing.gpu_peak_mb is not None else "—"
    print(
        f"    {method}: fit={timing.fit_seconds:.1f}s  infer={infer}  "
        f"wall={wall:.1f}s  rss={timing.peak_rss_mb:.0f}MB  gpu={gpu}",
        flush=True,
    )
    return 0
