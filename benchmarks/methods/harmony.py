"""Harmony embedding subprocess.

scIB recipe: PCA on scaled log1p .X, then Harmony correction in PC
space. We bill the PCA stage to Harmony's fit time since that's what
the published benchmarks do (it's part of running Harmony in practice).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.io import (
    MethodTiming,
    add_common_method_args,
    run_method_subprocess,
    scaled_X,
    track_memory,
)


def embed(adata, batch_key, label_key, counts_layer, k, seed):
    import harmonypy as hm
    from sklearn.decomposition import PCA

    X = scaled_X(adata, batch_key)
    with track_memory() as mem:
        t0 = time.perf_counter()
        pca = PCA(n_components=k, random_state=seed).fit_transform(X)
        pca_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        ho = hm.run_harmony(pca, adata.obs, batch_key, max_iter_harmony=20)
        harm_s = time.perf_counter() - t1
    emb_ = np.asarray(ho.Z_corr)
    if emb_.shape[0] != adata.n_obs:
        emb_ = emb_.T
    return emb_, MethodTiming(pca_s + harm_s, None, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_method_args(parser)
    args = parser.parse_args()
    return run_method_subprocess(args, embed)


if __name__ == "__main__":
    raise SystemExit(main())
