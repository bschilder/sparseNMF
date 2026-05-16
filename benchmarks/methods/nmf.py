"""NMF embedding subprocess.

scIB recipe: NMF on unscaled log1p-norm .X (non-negative).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.io import (
    MethodTiming,
    add_common_method_args,
    lognorm_X,
    run_method_subprocess,
    track_memory,
)


def embed(adata, batch_key, label_key, counts_layer, k, seed):
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import NMF

    X_dense = np.clip(lognorm_X(adata), 0.0, None)
    X = csr_matrix(X_dense)
    with track_memory() as mem:
        nmf = NMF(n_components=k, init="nndsvd", max_iter=500, random_state=seed)
        t0 = time.perf_counter()
        nmf.fit(X)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb_ = nmf.transform(X)
        inf_s = time.perf_counter() - t1
    return emb_, MethodTiming(
        fit_s, inf_s, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_method_args(parser)
    args = parser.parse_args()
    return run_method_subprocess(args, embed)


if __name__ == "__main__":
    raise SystemExit(main())
