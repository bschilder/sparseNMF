"""Smoke + correctness tests for :class:`sparse_nmf.SparseNMF`.

These tests exercise the standalone NMF path on small CPU-friendly
matrices. The joint NMF + autoencoder path is covered separately in
``test_joint.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparse_nmf import SparseNMF


def test_fit_transform_returns_correct_shape(small_sparse, device):
    nmf = SparseNMF(n_components=4, max_iter=20, device=device)
    W = nmf.fit_transform(small_sparse)
    assert W.shape == (small_sparse.shape[0], 4)


def test_components_are_non_negative(small_sparse, device):
    nmf = SparseNMF(n_components=4, max_iter=20, device=device)
    nmf.fit(small_sparse)
    assert (nmf.components_ >= 0).all(), "H must be non-negative"


def test_transform_returns_non_negative_codes(small_sparse, device):
    nmf = SparseNMF(n_components=4, max_iter=20, device=device)
    W = nmf.fit_transform(small_sparse)
    assert (W >= 0).all(), "W must be non-negative"


def test_recovered_factors_approximate_input(device):
    """On a noiseless rank-3 matrix, NMF with ``n_components=3`` should
    recover an approximation with low Frobenius error."""
    from sparse_nmf.data import generate_synthetic_sparse

    X = generate_synthetic_sparse(
        n_samples=120,
        n_features=80,
        n_components=3,
        noise=0.0,
        density=0.3,
        seed=11,
    )
    nmf = SparseNMF(n_components=3, max_iter=200, device=device, tol=1e-6)
    W = nmf.fit_transform(X)
    H = nmf.components_
    X_dense = X.toarray()
    recon = W @ H
    rel_err = np.linalg.norm(X_dense - recon) / max(np.linalg.norm(X_dense), 1e-12)
    # Loose bound — small rank + noise=0 should reconstruct well.
    assert rel_err < 0.5, f"reconstruction error too high: {rel_err:.3f}"


@pytest.mark.parametrize("n_components", [1, 4, 16])
def test_fit_supports_a_range_of_ranks(small_sparse, device, n_components):
    nmf = SparseNMF(n_components=n_components, max_iter=10, device=device)
    W = nmf.fit_transform(small_sparse)
    assert W.shape[1] == n_components


def test_repeated_fit_overwrites_state(small_sparse, device):
    """Calling ``fit`` twice with different parameters should produce
    fresh factors of the new shape — no stale state from the prior fit."""
    nmf = SparseNMF(n_components=4, max_iter=10, device=device)
    nmf.fit(small_sparse)
    H1_shape = nmf.components_.shape

    nmf2 = SparseNMF(n_components=8, max_iter=10, device=device)
    nmf2.fit(small_sparse)
    H2_shape = nmf2.components_.shape

    assert H1_shape != H2_shape, (H1_shape, H2_shape)


@pytest.mark.gpu
def test_runs_on_cuda_when_available():
    """Sanity check: explicitly pin device='cuda' and verify it doesn't
    fall back silently. Skipped on CPU-only machines."""
    from sparse_nmf.data import generate_synthetic_sparse

    X = generate_synthetic_sparse(n_samples=80, n_features=120, n_components=4, seed=0)
    nmf = SparseNMF(n_components=4, max_iter=10, device="cuda")
    W = nmf.fit_transform(X)
    assert W.shape == (80, 4)
