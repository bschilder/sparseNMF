"""Sample-data generator + loader correctness."""

from __future__ import annotations

import numpy as np
import pytest

from sparse_nmf.data import generate_synthetic_sparse, load_synthetic_sparse


def test_generated_matrix_is_sparse_and_non_negative():
    X = generate_synthetic_sparse(n_samples=100, n_features=200, n_components=4, seed=1)
    assert X.shape == (100, 200)
    assert X.nnz > 0
    # density should land within ~30% of the requested 5%
    actual_density = X.nnz / (X.shape[0] * X.shape[1])
    assert 0.03 < actual_density < 0.08, actual_density
    assert (X.data >= 0).all(), "NMF requires non-negative input"


def test_generated_matrix_is_deterministic_under_seed():
    a = generate_synthetic_sparse(seed=42).toarray()
    b = generate_synthetic_sparse(seed=42).toarray()
    np.testing.assert_array_equal(a, b)


def test_generated_matrices_differ_with_different_seeds():
    a = generate_synthetic_sparse(seed=1).toarray()
    b = generate_synthetic_sparse(seed=2).toarray()
    assert not np.array_equal(a, b)


def test_load_synthetic_sparse_returns_csr():
    """Should always return a usable matrix even when the bundled npz
    is missing — the loader falls back to on-the-fly generation."""
    X = load_synthetic_sparse()
    assert X.shape[0] > 0 and X.shape[1] > 0
    assert X.nnz > 0


@pytest.mark.parametrize("rank", [2, 4, 16])
def test_rank_parameter_controls_underlying_structure(rank):
    """SVD of the dense form should have ~``rank`` significant
    singular values when noise is moderate."""
    X = generate_synthetic_sparse(
        n_samples=120,
        n_features=180,
        n_components=rank,
        noise=0.05,
        density=0.4,  # higher density → cleaner SVD signal
        seed=0,
    ).toarray().astype(np.float32)
    s = np.linalg.svd(X, compute_uv=False)
    # The top ``rank`` singular values should each be larger than the
    # average of the rest by a clear margin. Loose check — just
    # confirms rank parameter has the intended effect.
    top_mean = s[:rank].mean()
    tail_mean = s[rank:].mean() if rank < len(s) else 0.0
    assert top_mean > 2 * tail_mean, (rank, top_mean, tail_mean)
