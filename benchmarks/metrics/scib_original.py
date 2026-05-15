"""Original Theis-lab ``scib`` scoring subprocess — primary impl.

Mirrors the YosefLab fallback's interface so the orchestrator can
pick either via ``--metrics-impl {scib_original, scib_yosef}``.

Requires glibc 2.38+ for scib's *shipped* precompiled LISI binary,
OR a one-shot rebuild of that binary against the host glibc. On
Ubuntu 22.04 (glibc 2.35) the shipped ``knn_graph.o`` fails to
load; the source ships alongside it and rebuilds in ~1 second:

    bash benchmarks/scripts/rebuild_scib_lisi.sh

This rebuild is idempotent and only needs to run once per
environment after ``pip install scib`` — it overwrites the .o
in scib's package directory. Without LISI on, scib_original
still scores graph_connectivity + silhouette_batch + bio metrics
correctly (validated on the L4 pod cr9kutuaxc5l88, 2026-05-15).

⚠ UNVALIDATED COMPOSITE AGGREGATION ⚠
The ``scib`` package returns per-metric scores as a flat DataFrame
without computing the bio/batch/Total category means. We hand-
aggregate via ``np.mean`` over the legacy column names below. This
is **not** identical to scib-metrics' aggregation:

  - scib-metrics weights bio sub-categories (NMI/ARI/silhouette/
    isolated_labels/cLISI) at 0.25 each within "Bio conservation";
    we use a flat mean over 6 keys (treating isolated_label_F1 and
    isolated_label_silhouette as separate, while scib-metrics
    averages them into one isolated_labels bucket).
  - Composite weights match Luecken 2022 (0.6 bio + 0.4 batch),
    but the bio and batch components are off the scib-metrics
    numbers by a few hundredths.

**Do not directly compare composites across ``--metrics-impl``
flags without recalibrating.** Use ``--metrics-impl scib_yosef``
for the canonical, validated aggregator. ``scib_original`` is
useful for cross-checking individual metric values but its
composites are best treated as a sanity-check signal, not a
score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.io import (
    SCIB_DATASETS,
    adata_fingerprint,
    load_embedding,
    load_error,
    load_scib_dataset,
    method_out_dir,
)


# Composite weights matching Luecken 2022: 0.6*bio + 0.4*batch.
_BIO_W, _BATCH_W = 0.6, 0.4


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def evaluate(adata, embedding: np.ndarray, batch_key: str, label_key: str, *, lisi: bool = True) -> dict:
    """Call ``scib.metrics.metrics`` with the standard scIB-paper config.

    scib's API uses ``adata`` (uncorrected) + ``adata_int`` (post-
    integration, with embedding in ``.obsm["X_emb"]``). For non-graph
    methods (everything except BBKNN/scanorama-graph) the embedding
    flag set is the same.
    """
    import scib

    a = adata.copy()
    a_int = adata.copy()
    a_int.obsm["X_emb"] = embedding

    # scIB's metric flags. Embedding-based methods → embed=True; pcr
    # needs the uncorrected adata which we pass as `adata`.
    # scib 1.1.x kwargs (verified via inspect.signature on the pod).
    # silhouette_=True turns on BOTH ASW_label and ASW_label/batch
    # (there is no separate silhouette_batch_ flag). lisi_graph_ is
    # the umbrella for LISI computation; clisi_/ilisi_ select which.
    results = scib.metrics.metrics(
        adata=a,
        adata_int=a_int,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_emb",
        cluster_key="cluster",
        # Bio
        nmi_=True,
        ari_=True,
        silhouette_=True,  # both ASW_label and ASW_label/batch
        isolated_labels_asw_=True,
        isolated_labels_f1_=True,
        clisi_=lisi,
        # Batch
        graph_conn_=True,
        pcr_=False,  # needs HVGs handled separately; skip for now
        kBET_=False,  # legacy R dep
        lisi_graph_=lisi,  # umbrella — required when clisi_/ilisi_ are on
        ilisi_=lisi,
        # Higher-cost extras we're not running
        cell_cycle_=False,
        hvg_score_=False,
        trajectory_=False,
        type_="embed",
        verbose=False,
    )
    # ``results`` is a single-column DataFrame indexed by metric name;
    # flatten to a plain dict.
    out: dict[str, float] = {}
    for name, val in results.iloc[:, 0].items():
        out[str(name)] = _safe_float(val)

    # Compute aggregates the same way scib-metrics does so the
    # downstream CSV schema matches.
    bio_keys = [
        "NMI_cluster/label", "ARI_cluster/label", "ASW_label",
        "isolated_label_F1", "isolated_label_silhouette", "cLISI",
    ]
    batch_keys = ["ASW_label/batch", "graph_conn", "iLISI"]

    def _mean(keys):
        vals = [out[k] for k in keys if k in out and np.isfinite(out.get(k, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    bio = _mean(bio_keys)
    batch = _mean(batch_keys)
    total = _BIO_W * bio + _BATCH_W * batch if (np.isfinite(bio) and np.isfinite(batch)) else float("nan")
    out["Bio conservation"] = bio
    out["Batch correction"] = batch
    out["Total"] = total
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(SCIB_DATASETS))
    parser.add_argument("--out-dir", required=True,
                        help="Run root; reads <out-dir>/<dataset>/<method>/X_emb.npz, "
                        "writes <out-dir>/<dataset>/<method>/metrics_original.json")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells-per-cohort", type=int, default=None)
    parser.add_argument("--no-hvg", action="store_true")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--no-lisi", action="store_true")
    args = parser.parse_args()

    print(f"  metrics(original): loading {args.dataset}...", flush=True)
    adata, batch_key, label_key, _ = load_scib_dataset(
        args.dataset,
        cells_per_cohort=args.cells_per_cohort,
        seed=args.seed,
        hvg=not args.no_hvg,
        n_hvg=args.n_hvg,
    )
    expected_fp = adata_fingerprint(adata)
    print(f"  metrics(original): {adata.shape}  fp={expected_fp}", flush=True)

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
            (method_out_dir(args.out_dir, args.dataset, method) / "metrics_original.error.txt").write_text(msg)
            failures += 1
            continue
        elapsed = time.perf_counter() - t0
        m["_metric_seconds"] = elapsed
        m["_impl"] = "scib_original"
        (method_out_dir(args.out_dir, args.dataset, method) / "metrics_original.json").write_text(
            json.dumps(m, indent=2)
        )
        print(f"      {method}: bio={m.get('Bio conservation', float('nan')):+.3f}  "
              f"batch={m.get('Batch correction', float('nan')):+.3f}  "
              f"composite={m.get('Total', float('nan')):+.3f}  ({elapsed:.1f}s)", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
