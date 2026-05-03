"""Sample data loaders for the sparse_nmf package.

A few small synthetic datasets shipped with the package so users can
run the examples / tests without an external download. Each dataset is
constructed to have a known low-rank-plus-noise structure, which makes
it useful both as a doctest fixture and as a sanity check that NMF is
recovering something meaningful.

Public helpers:

- :func:`generate_synthetic_sparse` — programmatic factory for a
  ``(n_samples, n_features)`` CSR matrix with controllable rank and
  density. Deterministic via ``seed``.
- :func:`load_synthetic_sparse` — loads the bundled
  ``data/synthetic_sparse.npz`` if present, falls back to generating
  on the fly.
"""

from __future__ import annotations

from importlib import resources

import numpy as np


def generate_synthetic_sparse(
    n_samples: int = 500,
    n_features: int = 1_000,
    n_components: int = 8,
    density: float = 0.05,
    noise: float = 0.1,
    seed: int = 0,
):
    """Build a ``(n_samples, n_features)`` CSR matrix with rank-``n_components``
    structure plus sparse noise.

    The matrix is constructed as ``W @ H + noise``, then thresholded to
    keep only the top ``density`` fraction of entries — emulating the
    sparsity pattern of, e.g., gene-association count data.

    Parameters
    ----------
    n_samples, n_features
        Output shape.
    n_components
        Rank of the underlying low-rank structure. NMF with
        ``n_components`` should recover (close to) this.
    density
        Fraction of non-zero entries in the output.
    noise
        Standard deviation of additive Gaussian noise on the dense
        product before thresholding. Larger ``noise`` makes recovery
        harder.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_samples, n_features)``, dtype ``float32``,
        non-negative.
    """
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(seed)
    W = rng.gamma(shape=2.0, scale=1.0, size=(n_samples, n_components)).astype(np.float32)
    H = rng.gamma(shape=2.0, scale=1.0, size=(n_components, n_features)).astype(np.float32)
    dense = W @ H
    dense += noise * rng.standard_normal(dense.shape).astype(np.float32) * dense.std()
    dense = np.clip(dense, 0.0, None)

    n_keep = int(density * n_samples * n_features)
    threshold = np.partition(dense.ravel(), -n_keep)[-n_keep]
    dense[dense < threshold] = 0.0
    return csr_matrix(dense)


def load_synthetic_sparse():
    """Load the bundled ``synthetic_sparse.npz``; generate if missing.

    Returns the same shape as :func:`generate_synthetic_sparse`'s
    defaults so callers can switch between the two without changing
    downstream code.
    """
    try:
        with resources.files("sparse_nmf.data").joinpath("synthetic_sparse.npz").open("rb") as f:
            data = np.load(f)
            from scipy.sparse import csr_matrix

            return csr_matrix(
                (data["data"], data["indices"], data["indptr"]),
                shape=tuple(data["shape"]),
            )
    except (FileNotFoundError, ModuleNotFoundError):
        return generate_synthetic_sparse()
