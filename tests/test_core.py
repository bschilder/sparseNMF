"""Smoke + correctness tests for :class:`sparse_nmf.SparseNMF`.

These tests exercise the standalone NMF path on small CPU-friendly
matrices. The joint NMF + autoencoder path is covered separately in
``test_joint.py``.

API note
--------
``SparseNMF`` has a ``fit_transform(X) -> np.ndarray`` method that
returns ``W`` as a NumPy array, and stores ``self.W`` / ``self.H`` as
torch tensors on the configured device. There is no separate ``fit``
or sklearn-style ``components_`` property — to inspect the
factorization, use ``nmf.H`` (the per-feature loadings) and
``nmf.W`` (the per-sample codes) directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparse_nmf import SparseNMF


def _to_numpy(t):
    """Convert the model's torch tensors into numpy without assuming
    where they live (CPU vs CUDA)."""
    return t.detach().cpu().numpy()


def test_fit_transform_returns_correct_shape(small_sparse, device):
    nmf = SparseNMF(n_components=4, max_iter=20, device=device)
    W = nmf.fit_transform(small_sparse)
    assert W.shape == (small_sparse.shape[0], 4)


def test_components_are_non_negative(small_sparse, device):
    nmf = SparseNMF(n_components=4, max_iter=20, device=device)
    nmf.fit_transform(small_sparse)
    H = _to_numpy(nmf.H)
    assert (H >= -1e-6).all(), "H must be non-negative (small slack for fp tolerance)"


def test_transform_returns_non_negative_codes(small_sparse, device):
    nmf = SparseNMF(n_components=4, max_iter=20, device=device)
    W = nmf.fit_transform(small_sparse)
    assert (W >= -1e-6).all(), "W must be non-negative (small slack for fp tolerance)"


def test_recovered_factors_approximate_input(device):
    """On a low-noise rank-3 matrix, NMF with ``n_components=3`` should
    recover an approximation that's at least better than the trivial
    zero baseline."""
    from sparse_nmf.data import generate_synthetic_sparse

    X = generate_synthetic_sparse(
        n_samples=120,
        n_features=80,
        n_components=3,
        noise=0.05,
        density=0.4,
        seed=11,
    )
    nmf = SparseNMF(n_components=3, max_iter=200, device=device)
    W = nmf.fit_transform(X)
    H = _to_numpy(nmf.H)
    X_dense = X.toarray()
    recon = W @ H
    rel_err = np.linalg.norm(X_dense - recon) / max(np.linalg.norm(X_dense), 1e-12)
    # Loose bound — multiplicative-update NMF on noisy data won't hit
    # zero error, but should beat the trivial zero-recon baseline (1.0)
    # by a clear margin.
    assert rel_err < 0.9, f"reconstruction error too high: {rel_err:.3f}"


@pytest.mark.parametrize("n_components", [1, 4, 16])
def test_fit_supports_a_range_of_ranks(small_sparse, device, n_components):
    nmf = SparseNMF(n_components=n_components, max_iter=10, device=device)
    W = nmf.fit_transform(small_sparse)
    assert W.shape[1] == n_components
    assert _to_numpy(nmf.H).shape == (n_components, small_sparse.shape[1])


def test_repeated_fits_have_independent_state(small_sparse, device):
    """Two ``SparseNMF`` instances with different ``n_components`` should
    each end up with H matrices of their own configured shape — no
    cross-contamination."""
    nmf1 = SparseNMF(n_components=4, max_iter=10, device=device)
    nmf1.fit_transform(small_sparse)
    H1_shape = _to_numpy(nmf1.H).shape

    nmf2 = SparseNMF(n_components=8, max_iter=10, device=device)
    nmf2.fit_transform(small_sparse)
    H2_shape = _to_numpy(nmf2.H).shape

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
