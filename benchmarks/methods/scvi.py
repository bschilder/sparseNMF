"""scVI embedding subprocess.

Uses raw counts; ``k`` maps to ``n_latent``. Fresh subprocess means
no JAX has touched the GPU yet — accelerator detection just works.
"""

from __future__ import annotations

import argparse
import time

from benchmarks.io import (
    MethodTiming,
    add_common_method_args,
    run_method_subprocess,
    track_memory,
)


def embed(adata, batch_key, label_key, counts_layer, k, seed):
    import scvi

    accelerator = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            accelerator = "gpu"
    except ImportError:
        pass

    a = adata.copy()
    a.X = a.layers[counts_layer]
    with track_memory() as mem:
        scvi.model.SCVI.setup_anndata(a, batch_key=batch_key)
        model = scvi.model.SCVI(a, n_latent=k)
        t0 = time.perf_counter()
        model.train(
            max_epochs=100,
            accelerator=accelerator,
            devices=1,
            plan_kwargs={"lr": 1e-3},
        )
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb_ = model.get_latent_representation()
        inf_s = time.perf_counter() - t1
    return emb_, MethodTiming(fit_s, inf_s, peak_rss_mb=mem["peak_rss_mb"], gpu_peak_mb=mem["gpu_peak_mb"])


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_method_args(parser)
    args = parser.parse_args()
    return run_method_subprocess(args, embed)


if __name__ == "__main__":
    raise SystemExit(main())
