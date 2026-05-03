"""Tests for the attention-helper functions.

These pure-ish helpers wrap a few common downstream operations:
``trace_attention_to_genes`` propagates per-component attention back
to gene features via the NMF H matrix, ``compute_attention_correlation``
correlates attention scores against the original input, and
``extract_attention_weights`` pulls per-sample attention out of a
trained autoencoder.

The math in the first two is simple enough that we can test against
hand-rolled reference computations rather than chasing numerical
stability.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sparse_nmf import (
    compute_attention_correlation,
    extract_attention_weights,
    trace_attention_to_genes,
)


# ── trace_attention_to_genes ────────────────────────────────────────


def test_trace_returns_correct_shape():
    """The output gene attention should be
    (n_samples, n_genes) when input is (n_samples, n_components) @
    (n_components, n_genes)."""
    n_samples, n_components, n_genes = 5, 3, 8
    A = np.random.RandomState(0).rand(n_samples, n_components).astype(np.float32)
    H = np.random.RandomState(1).rand(n_components, n_genes).astype(np.float32)
    out = trace_attention_to_genes(A, H, normalize=False)
    assert out.shape == (n_samples, n_genes)


def test_trace_unnormalized_equals_matmul():
    """With normalize=False, output must match A @ H exactly — this
    is the documented contract."""
    rng = np.random.RandomState(42)
    A = rng.rand(4, 3).astype(np.float32)
    H = rng.rand(3, 7).astype(np.float32)
    expected = A @ H
    out = trace_attention_to_genes(A, H, normalize=False)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_trace_normalized_rows_sum_to_one_or_less():
    """``normalize=True`` divides by max(row_sum, 1.0). Rows that
    have row_sum ≥ 1 should be exactly normalized to sum to 1.0.
    Rows with row_sum < 1 keep their original values (since
    dividing by 1.0 is a no-op)."""
    A = np.array([[10.0, 20.0, 30.0], [0.1, 0.1, 0.1]], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    out = trace_attention_to_genes(A, H, normalize=True)
    # First row had sum 60 > 1, must end up summing to 1.0.
    np.testing.assert_allclose(out[0].sum(), 1.0, atol=1e-6)
    # Second row had sum 0.3 < 1, divided by 1.0 → unchanged.
    np.testing.assert_allclose(out[1].sum(), 0.3, atol=1e-6)


def test_trace_accepts_torch_H():
    """``nmf_H`` is allowed to be a torch.Tensor (e.g. straight off
    a fitted SparseNMF.H attribute) — the helper must convert it
    transparently. Tests both CPU and the .detach().cpu() path."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    H_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    H_torch = torch.tensor(H_np)
    out_np = trace_attention_to_genes(A, H_np, normalize=False)
    out_torch = trace_attention_to_genes(A, H_torch, normalize=False)
    np.testing.assert_array_equal(out_np, out_torch)


# ── compute_attention_correlation ───────────────────────────────────


def test_correlation_returns_dataframe_with_expected_columns():
    """The returned DataFrame must have the columns the docstring
    advertises — downstream code keys off them."""
    rng = np.random.RandomState(0)
    n, m = 30, 10
    X = rng.rand(n, m)
    A = rng.rand(n, m)
    df = compute_attention_correlation(A, X, stratify_by_unique_values=False, verbose=False)
    expected = {"stratum", "subset", "n_samples", "pearson", "spearman", "spearman_p"}
    assert expected <= set(df.columns), f"missing columns: {expected - set(df.columns)}"


def test_correlation_perfect_match_yields_correlation_one():
    """Setting attention = X exactly should produce Pearson and
    Spearman ≈ 1.0 — sanity check that we're correlating the right
    things in the right direction."""
    rng = np.random.RandomState(1)
    X = rng.rand(40, 6)
    A = X.copy()
    df = compute_attention_correlation(
        A, X, stratify_by_unique_values=False, verbose=False
    )
    # 'all' subset row — perfect correlation
    all_row = df[df["subset"] == "all"].iloc[0]
    assert all_row["pearson"] > 0.99
    assert all_row["spearman"] > 0.99


def test_correlation_shape_mismatch_raises():
    """Mismatched shapes should fail loudly with ValueError so users
    don't silently get garbage correlations."""
    rng = np.random.RandomState(2)
    A = rng.rand(10, 5)
    X = rng.rand(10, 6)  # wrong feature count
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_attention_correlation(A, X, verbose=False)


def test_correlation_obs_mask_subsets_X():
    """``obs_mask`` selects a subset of X rows. The attention matrix
    is expected to already be in subset-order, so its row count
    determines what's correlated."""
    rng = np.random.RandomState(3)
    X = rng.rand(50, 5)
    mask = np.zeros(50, dtype=bool)
    mask[:10] = True
    A = X[mask].copy()  # (10, 5) — matches subset
    df = compute_attention_correlation(
        A, X, obs_mask=mask, stratify_by_unique_values=False, verbose=False
    )
    # Should run without ValueError (which would fire if the mask
    # weren't applied).
    assert len(df) > 0


def test_correlation_with_sparse_input_works():
    """Sparse X must be auto-densified to compute the correlation —
    drops bypassed via ``hasattr(X, 'toarray')``."""
    from scipy.sparse import csr_matrix

    rng = np.random.RandomState(4)
    X_dense = rng.rand(20, 4)
    A = X_dense.copy()
    X_sparse = csr_matrix(X_dense)
    df = compute_attention_correlation(
        A, X_sparse, stratify_by_unique_values=False, verbose=False
    )
    all_row = df[df["subset"] == "all"].iloc[0]
    assert all_row["pearson"] > 0.99


def test_correlation_stratification_yields_multiple_strata_rows():
    """With ``stratify_by_unique_values=True``, the result should
    include rows for ``2_unique`` / ``3_unique`` / ``4+_unique`` strata
    when the input contains samples with each cardinality.

    Build a synthetic X where row 0 has 2 unique values, row 1 has
    3, and rows 2-9 have many unique values."""
    n, m = 10, 10
    X = np.zeros((n, m), dtype=np.float32)
    X[0] = np.array([0, 1] * 5, dtype=np.float32)  # 2 unique
    X[1] = np.tile(np.array([0, 1, 2], dtype=np.float32), 4)[:m]  # 3 unique
    X[2:] = np.arange(m, dtype=np.float32)  # 10 unique each (4+_unique stratum)
    # Permute X[2:] rows so they aren't all identical (else corr is undefined).
    rng = np.random.RandomState(5)
    for i in range(2, n):
        X[i] = rng.permutation(X[i])
    A = X.copy()
    df = compute_attention_correlation(A, X, stratify_by_unique_values=True, verbose=False)
    strata = set(df["stratum"].unique())
    assert {"2_unique", "3_unique", "4+_unique"} <= strata


# ── extract_attention_weights ───────────────────────────────────────


def test_extract_attention_raises_when_no_attention_enabled(small_sparse, device):
    """A model trained without ``use_feature_attention`` or
    ``use_transformer`` must reject this call rather than silently
    returning garbage."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=4,
        latent_dim=2,
        hidden_dims=(8,),
        use_feature_attention=False,
        device=device,
        random_state=0,
    )
    X_nmf = np.random.RandomState(0).rand(n, 4).astype(np.float32)
    with pytest.raises(ValueError, match="attention enabled"):
        extract_attention_weights(model, X_nmf, batch_size=64, verbose=False)


def test_extract_attention_feature_mode_returns_correct_shape(small_sparse, device):
    """With ``use_feature_attention=True`` the result is
    (n_samples, nmf_components) and values are sigmoid-bounded in
    [0, 1]."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    nmf_comp = 4
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=nmf_comp,
        latent_dim=2,
        hidden_dims=(8,),
        use_feature_attention=True,
        device=device,
        random_state=0,
    )
    # Synthetic NMF embeddings — any positive values work for shape test.
    X_nmf = np.random.RandomState(0).rand(n, nmf_comp).astype(np.float32)
    A = extract_attention_weights(model, X_nmf, batch_size=64, verbose=False)
    assert A.shape == (n, nmf_comp)
    assert A.dtype == np.float32
    # Sigmoid output ∈ (0, 1) — strict inside since we never hit ±∞ logit.
    assert (A > 0).all() and (A < 1).all()


def test_extract_attention_accepts_torch_input(small_sparse, device):
    """X_nmf can be either np.ndarray or torch.Tensor — both code
    paths should produce identical output (modulo float)."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    nmf_comp = 4
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=nmf_comp,
        latent_dim=2,
        hidden_dims=(8,),
        use_feature_attention=True,
        device=device,
        random_state=0,
    )
    X_nmf_np = np.random.RandomState(0).rand(n, nmf_comp).astype(np.float32)
    X_nmf_torch = torch.from_numpy(X_nmf_np)
    A_np = extract_attention_weights(model, X_nmf_np, batch_size=64, verbose=False)
    A_torch = extract_attention_weights(model, X_nmf_torch, batch_size=64, verbose=False)
    np.testing.assert_allclose(A_np, A_torch, atol=1e-6)
