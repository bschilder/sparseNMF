"""PCA embedding subprocess.

scIB recipe: PCA on per-batch scaled log1p .X.

Run as:
    python -m benchmarks.methods.pca --dataset pancreas --out-dir runs/foo \
        --method-name PCA --k 30
"""

from __future__ import annotations

import argparse
import time

from benchmarks.io import (
    MethodTiming,
    add_common_method_args,
    run_method_subprocess,
    scaled_X,
    track_memory,
)


def embed(adata, batch_key, label_key, counts_layer, k, seed):
    from sklearn.decomposition import PCA

    X = scaled_X(adata, batch_key)
    with track_memory() as mem:
        pca = PCA(n_components=k, random_state=seed)
        t0 = time.perf_counter()
        pca.fit(X)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb_ = pca.transform(X)
        inf_s = time.perf_counter() - t1
    return emb_, MethodTiming(fit_s, inf_s, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_method_args(parser)
    args = parser.parse_args()
    return run_method_subprocess(args, embed)


if __name__ == "__main__":
    raise SystemExit(main())
