"""Standalone NMF on the bundled synthetic data.

Run from repo root::

    python examples/basic_usage.py

Prints reconstruction error per iteration and a small recovery
sanity check. ``matplotlib`` is optional — if installed, also writes
a recon-error plot to ``examples/basic_usage.png``.
"""

from __future__ import annotations

import time

import numpy as np

from sparse_nmf import SparseNMF
from sparse_nmf.data import generate_synthetic_sparse


def main() -> int:
    print("Generating a 2,000 × 5,000 sparse matrix with planted rank=16 structure...")
    X = generate_synthetic_sparse(
        n_samples=2_000,
        n_features=5_000,
        n_components=16,
        density=0.05,
        seed=42,
    )
    print(f"  shape={X.shape}  nnz={X.nnz:,}  density={X.nnz / (X.shape[0]*X.shape[1]):.3%}")

    print("Fitting SparseNMF (n_components=16, max_iter=200)...")
    t0 = time.time()
    nmf = SparseNMF(n_components=16, max_iter=200, device="cpu", verbose=True)
    W = nmf.fit_transform(X)
    H = nmf.components_
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  W.shape={W.shape}  H.shape={H.shape}")

    # Quick recovery sanity check.
    X_dense = X.toarray()
    recon = W @ H
    rel_err = np.linalg.norm(X_dense - recon) / np.linalg.norm(X_dense)
    print(f"  relative reconstruction error: {rel_err:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
