"""sparseNMF with nonzero_mse_weight=1.0 embedding subprocess.

Forced to CPU — the gradient-descent path at k=30 OOMs on a 16 GB
GPU and is CPU-bound elsewhere. Run as opt-in via
``--methods sparseNMF+nonzero`` in the orchestrator.
"""

from __future__ import annotations

import argparse
import time

from benchmarks.io import (
    MethodTiming,
    add_common_method_args,
    counts,
    run_method_subprocess,
    track_memory,
)


def embed(adata, batch_key, label_key, counts_layer, k, seed):
    from sparse_nmf import train_sparse_nmf

    X = counts(adata, counts_layer)
    with track_memory() as mem:
        t0 = time.perf_counter()
        W, _ = train_sparse_nmf(
            X_sparse=X, n_components=k, device="cpu", random_state=seed,
            verbose=False,
            mse_weight=0.0, nonzero_mse_weight=1.0,
        )
        fit_s = time.perf_counter() - t0
    return W, MethodTiming(fit_s, None, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_method_args(parser)
    args = parser.parse_args()
    return run_method_subprocess(args, embed)


if __name__ == "__main__":
    raise SystemExit(main())
