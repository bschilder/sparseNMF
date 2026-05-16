"""sparseNMF embedding subprocess.

Single CUDA context per process — no JAX, no scib-metrics in this
module. The wedged-GPU probe + cpu fallback we had in the original
``embed_sparse_nmf`` is no longer needed: nothing else holds the
context in this subprocess.
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


def embed(adata, batch_key, label_key, counts_layer, k, seed, **kwargs):
    from sparse_nmf import train_sparse_nmf

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    X = counts(adata, counts_layer)
    with track_memory() as mem:
        t0 = time.perf_counter()
        W, _ = train_sparse_nmf(
            X_sparse=X,
            n_components=k,
            device=device,
            random_state=seed,
            verbose=False,
            **kwargs,
        )
        fit_s = time.perf_counter() - t0
    return W, MethodTiming(
        fit_s, None, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_method_args(parser)
    args = parser.parse_args()
    return run_method_subprocess(args, embed)


if __name__ == "__main__":
    raise SystemExit(main())
