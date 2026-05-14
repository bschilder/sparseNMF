"""Determinism helpers for the demo scripts and notebooks.

Reproducible UMAP / PCA / NMF / sparseNMF requires more than just
``random_state=seed`` on each method. Two extra sources of run-to-run
variation matter in practice:

1. **Multi-threaded BLAS reductions** (OpenBLAS, MKL) sum
   floating-point matrices in parallel. Parallel summation isn't
   associative, so the bit-exact result depends on thread scheduling.
   Tiny rounding differences in :math:`W` / :math:`H` propagate
   through to visibly different UMAP layouts.

2. **PyTorch / cuBLAS non-determinism** on GPU sparse matmul.

Module-load behavior — the env vars below are set **at import time**,
which means this module must be imported *before* numpy / torch.
Practically: put ``from _determinism import set_global_seed`` at the
top of your script, immediately after ``from __future__ import
annotations``. The env-var ``setdefault`` calls only fire when the
variable isn't already set, so callers can still override from the
shell if they want multi-threaded speed at the cost of reproducibility.
"""

from __future__ import annotations

import os

# ── Single-threaded BLAS / NumExpr / Numba ──────────────────────────
# Set before numpy / scipy / torch import so the thread pools come up
# with a count of 1. Caller can override by setting the env var
# explicitly before invoking the script.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate
):
    os.environ.setdefault(_v, "1")

# Hash randomization (set/dict iteration order)
os.environ.setdefault("PYTHONHASHSEED", "0")

# cuBLAS deterministic workspace — required for
# torch.use_deterministic_algorithms(True) when GPU is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_global_seed(seed: int = 0) -> None:
    """Seed every RNG the demos touch.

    Idempotent; safe to call multiple times. Skips libraries that
    aren't installed (e.g. ``scvi`` is only relevant for the
    benchmark, not the demos).
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # warn_only=True so a non-deterministic op falls back to a
        # warning rather than an error. Some ops (e.g. scatter_add on
        # GPU) don't have a deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import scvi  # type: ignore[import-not-found]

        scvi.settings.seed = seed
    except ImportError:
        pass
