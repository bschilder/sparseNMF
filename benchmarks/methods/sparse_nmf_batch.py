"""Batch-aware sparseNMF embedding subprocess.

Uses ``train_sparse_nmf_batch_aware`` — an iNMF-style factorization
where per-batch additive corrections V[b] absorb batch-specific
gene-expression patterns, leaving the cell embeddings W in a
batch-invariant latent space.

Same interface as the regular sparseNMF method module so the
orchestrator can spawn it identically.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.io import (
    MethodTiming,
    add_common_method_args,
    counts,
    run_method_subprocess,
    track_memory,
)


def embed(adata, batch_key, label_key, counts_layer, k, seed):
    from sparse_nmf import train_sparse_nmf_batch_aware

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    X = counts(adata, counts_layer)
    batch = np.asarray(adata.obs[batch_key].astype(str).values)

    with track_memory() as mem:
        t0 = time.perf_counter()
        result = train_sparse_nmf_batch_aware(
            X_sparse=X,
            batch=batch,
            n_components=k,
            device=device,
            random_state=seed,
            verbose=False,
            # alignment_weight=0.5 from hyperparam sweep
            # (docs/_static/hyperparam_sweep/results.csv at k=30):
            # avg silhouette_label across pancreas+sim1
            #   α_v=0.5: 0.270 (best)
            #   α_v=2.0: 0.263
            #   α_v=8.0: 0.257
            # Lighter batch correction preserves more bio signal
            # without measurably worsening batch mixing.
            sparsity_weight=0.01,
            alignment_weight=0.5,
        )
        fit_s = time.perf_counter() - t0

    return result.W, MethodTiming(
        fit_s,
        None,
        peak_rss_mb=mem["peak_rss_mb"],
        gpu_peak_mb=mem["gpu_peak_mb"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_method_args(parser)
    args = parser.parse_args()
    return run_method_subprocess(args, embed)


if __name__ == "__main__":
    raise SystemExit(main())
