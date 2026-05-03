"""sparse_nmf — GPU-accelerated sparse non-negative matrix factorization.

Public API: re-exports the most-used names from :mod:`sparse_nmf._core` so
downstream callers can ``from sparse_nmf import SparseNMF`` without
reaching into the internal layout.

Usage
-----
Standalone NMF::

    from sparse_nmf import SparseNMF
    from scipy.sparse import csr_matrix

    X_sparse = csr_matrix(...)
    nmf = SparseNMF(n_components=256, max_iter=500, device="cuda")
    X_reduced = nmf.fit_transform(X_sparse)

Joint NMF + autoencoder::

    from sparse_nmf import train_joint_model

    z, model = train_joint_model(
        X_sparse,
        n_samples=X_sparse.shape[0],
        n_features=X_sparse.shape[1],
        nmf_components=256,
        latent_dim=2,
        device="cuda",
        n_epochs=100,
    )
"""

from __future__ import annotations

# ── AoU shim ────────────────────────────────────────────────────────
# Vendored ``_core.py`` (synced verbatim from bschilder/AoU) does
# ``from AoU.utils import l2_normalize`` inside two branches of
# ``train_sparse_nmf`` (lines ~2094 and ~2178 — the
# ``normalize_outputs=True`` paths). The standalone sparseNMF package
# doesn't ship the AoU module, so without this shim those branches
# raise ModuleNotFoundError at runtime.
#
# Rather than patching the verbatim file (which we keep clean to
# minimize churn on upstream re-syncs), register a minimal
# ``AoU.utils`` namespace in ``sys.modules`` so the import resolves
# transparently. Done at package import time so ``from sparse_nmf
# import ...`` is enough to enable the path.
import sys as _sys
import types as _types

import numpy as _np


def _l2_normalize(x):
    """L2-normalize each row of ``x`` to unit length.

    Drop-in for ``AoU.utils.l2_normalize`` — that function takes the
    NMF embeddings ndarray (shape ``n_samples × n_components``) and
    returns it with each row scaled so its L2 norm is 1.0. Zero rows
    pass through unchanged (avoid div-by-zero).

    Exposed at the package level (``sparse_nmf._l2_normalize``) so
    tests can exercise it directly without poking at ``sys.modules``.
    """
    arr = _np.asarray(x, dtype=_np.float32)
    norms = _np.linalg.norm(arr, axis=1, keepdims=True)
    norms = _np.where(norms < 1e-12, 1.0, norms)
    return arr / norms


_aou_root = _types.ModuleType("AoU")
_aou_utils = _types.ModuleType("AoU.utils")
_aou_utils.l2_normalize = _l2_normalize
_aou_root.utils = _aou_utils
_sys.modules.setdefault("AoU", _aou_root)
_sys.modules.setdefault("AoU.utils", _aou_utils)
# NB: keep ``_np``, ``_sys``, ``_types`` in module scope — they're
# closed over by ``_l2_normalize`` at call time. Pre-leading-
# underscore makes them implicitly private to the package.

# E402 (module-level import not at top of file) is intentional here:
# the AoU shim above MUST register ``AoU.utils`` in sys.modules
# BEFORE we trigger the import of ``_core`` (which imports
# ``AoU.utils.l2_normalize`` at module level inside its
# ``train_sparse_nmf``). Reordering would re-introduce the
# ModuleNotFoundError this file was added to fix.
from sparse_nmf._core import (  # noqa: E402
    SparseNMF,
    SparseNMF_Autoencoder,
    compute_attention_correlation,
    compute_joint_loss,
    extract_and_aggregate_attention,
    extract_attention_weights,
    plot_nmf_factor_distributions,
    sparse_nmf,
    trace_attention_to_genes,
    train_joint_model,
    train_sparse_nmf,
)

__all__ = [
    "SparseNMF",
    "SparseNMF_Autoencoder",
    "compute_attention_correlation",
    "compute_joint_loss",
    "extract_and_aggregate_attention",
    "extract_attention_weights",
    "plot_nmf_factor_distributions",
    "sparse_nmf",
    "trace_attention_to_genes",
    "train_joint_model",
    "train_sparse_nmf",
]

# Single source of truth for the package version. The release workflow
# reads ``__version__`` to tag wheels and Docker images consistently.
__version__ = "0.1.0"
