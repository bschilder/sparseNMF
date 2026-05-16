"""scib-metrics (YosefLab) scoring subprocess — fallback impl.

Loads all (dataset, method) embeddings from disk and writes per-method
score rows as JSON into the same directory tree. JAX is pinned to CPU
*locally* (this module imports scib-metrics which imports jax), but that
no longer matters because no other GPU-using code runs in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# JAX claims the GPU at import. Earlier we pinned this to CPU because
# JAX + PyTorch in the same process caused CUDA-context wedging
# (scVI/sparseNMF couldn't get the device after scib-metrics ran).
# With subprocess isolation (each method in its own process) that's
# no longer a concern — this metrics subprocess has no PyTorch and
# can safely use GPU JAX, which is 10-50× faster for LISI/silhouette
# than the CPU path on 30k+ cell datasets.
#
# Set JAX_PLATFORMS=cpu in the calling environment to override if
# you're on a CPU-only host or want to force CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np

from benchmarks.io import (
    SCIB_DATASETS,
    adata_fingerprint,
    load_embedding,
    load_error,
    load_scib_dataset,
    load_timing,
    method_out_dir,
)


def evaluate(adata, embedding: np.ndarray, batch_key: str, label_key: str, *, lisi: bool = True) -> dict:
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    a = adata.copy()
    a.obsm["X_emb"] = embedding
    bio = BioConservation(
        nmi_ari_cluster_labels_leiden=True,
        silhouette_label=True,
        isolated_labels=True,
        clisi_knn=lisi,
        nmi_ari_cluster_labels_kmeans=False,
    )
    batch = BatchCorrection(
        graph_connectivity=True,
        bras=True,
        ilisi_knn=lisi,
        kbet_per_label=False,
        pcr_comparison=False,
    )
    bm = Benchmarker(
        a,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=["X_emb"],
        bio_conservation_metrics=bio,
        batch_correction_metrics=batch,
        n_jobs=-1,
    )
    bm.benchmark()
    df = bm.get_results(min_max_scale=False, clean_names=False)
    row = df.loc["X_emb"] if "X_emb" in df.index else df.iloc[0]
    out: dict[str, float] = {}
    for k, v in row.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(SCIB_DATASETS))
    parser.add_argument("--out-dir", required=True,
                        help="Run root; reads <out-dir>/<dataset>/<method>/X_emb.npz, "
                        "writes <out-dir>/<dataset>/<method>/metrics_yosef.json")
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Method names to score (must match the dir name under <out-dir>/<dataset>/).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells-per-cohort", type=int, default=None)
    parser.add_argument("--no-hvg", action="store_true")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--no-lisi", action="store_true")
    args = parser.parse_args()

    print(f"  metrics(yosef): loading {args.dataset}...", flush=True)
    adata, batch_key, label_key, _ = load_scib_dataset(
        args.dataset,
        cells_per_cohort=args.cells_per_cohort,
        seed=args.seed,
        hvg=not args.no_hvg,
        n_hvg=args.n_hvg,
    )
    expected_fp = adata_fingerprint(adata)
    print(f"  metrics(yosef): {adata.shape}  fp={expected_fp}", flush=True)

    failures = 0
    for method in args.methods:
        if load_error(args.out_dir, args.dataset, method) is not None:
            print(f"    {method}: SKIP (embedding step failed)", flush=True)
            failures += 1
            continue
        try:
            emb, fp = load_embedding(args.out_dir, args.dataset, method)
        except FileNotFoundError:
            print(f"    {method}: SKIP (no embedding written)", flush=True)
            failures += 1
            continue
        if fp != expected_fp:
            msg = f"fingerprint mismatch: embedding={fp} vs current adata={expected_fp}"
            print(f"    {method}: SKIP ({msg})", flush=True)
            failures += 1
            continue
        print(f"    {method}: scoring (n={emb.shape[0]}, k={emb.shape[1]})...", flush=True)
        t0 = time.perf_counter()
        try:
            m = evaluate(adata, emb, batch_key, label_key, lisi=not args.no_lisi)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"      FAILED: {msg[:200]}", flush=True)
            (method_out_dir(args.out_dir, args.dataset, method) / "metrics_yosef.error.txt").write_text(msg)
            failures += 1
            continue
        elapsed = time.perf_counter() - t0
        m["_metric_seconds"] = elapsed
        m["_impl"] = "scib_metrics_yosef"
        (method_out_dir(args.out_dir, args.dataset, method) / "metrics_yosef.json").write_text(
            json.dumps(m, indent=2)
        )
        bio = float(m.get("Bio conservation", float("nan")))
        bat = float(m.get("Batch correction", float("nan")))
        tot = float(m.get("Total", float("nan")))
        print(f"      {method}: bio={bio:+.3f}  batch={bat:+.3f}  composite={tot:+.3f}  "
              f"({elapsed:.1f}s)", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
