"""One-shot script to materialize ``src/sparse_nmf/data/synthetic_sparse.npz``
that's bundled with the wheel.

Run from repo root::

    python data/generate_sample_data.py

Re-run when the synthetic generator's defaults change so the bundled
copy stays in sync. The bundled file is what
:func:`sparse_nmf.data.load_synthetic_sparse` returns when no
explicit generation parameters are passed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sparse_nmf.data import generate_synthetic_sparse

OUT = Path(__file__).resolve().parents[1] / "src" / "sparse_nmf" / "data" / "synthetic_sparse.npz"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    X = generate_synthetic_sparse(
        n_samples=500,
        n_features=1_000,
        n_components=8,
        density=0.05,
        noise=0.1,
        seed=0,
    )
    np.savez(
        OUT,
        data=X.data,
        indices=X.indices,
        indptr=X.indptr,
        shape=np.array(X.shape, dtype=np.int64),
    )
    nnz = X.nnz
    print(f"wrote {OUT}  shape={X.shape}  nnz={nnz:,}  size={OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
