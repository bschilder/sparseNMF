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

from sparse_nmf._core import (
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
