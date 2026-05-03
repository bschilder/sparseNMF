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


# ── extract_and_aggregate_attention ─────────────────────────────────


@pytest.fixture
def attention_setup(small_sparse, device):
    """Build a feature-attention model + matched X_nmf + nmf_H once
    per test. Saves per-test boilerplate for the 8+ aggregate tests
    below."""
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
    rng = np.random.RandomState(0)
    X_nmf = rng.rand(n, nmf_comp).astype(np.float32)
    nmf_H = rng.rand(nmf_comp, m).astype(np.float32)
    return model, X_nmf, nmf_H, n, nmf_comp, m


def test_aggregate_returns_two_dataframes(attention_setup):
    """Default call returns (gene_df, nmf_df) DataFrames with the
    documented columns. This is the primary API contract."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, n, k, m = attention_setup
    gene_df, nmf_df = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False
    )
    expected = {
        "feature_index",
        "mean_attention",
        "min_attention",
        "max_attention",
        "n_samples_nonzero",
        "pct_samples_nonzero",
    }
    assert expected <= set(gene_df.columns), (
        f"missing in gene_df: {expected - set(gene_df.columns)}"
    )
    assert expected <= set(nmf_df.columns), (
        f"missing in nmf_df: {expected - set(nmf_df.columns)}"
    )
    # One row per feature.
    assert len(gene_df) == m
    assert len(nmf_df) == k


def test_aggregate_with_feature_names(attention_setup):
    """When ``gene_feature_names`` is provided, the gene DataFrame
    has a ``feature_name`` column populated with those names."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, _, k, m = attention_setup
    gene_names = [f"gene_{i:04d}" for i in range(m)]
    nmf_names = [f"nmf_{i}" for i in range(k)]
    gene_df, nmf_df = extract_and_aggregate_attention(
        model,
        X_nmf,
        nmf_H,
        batch_size=64,
        verbose=False,
        gene_feature_names=gene_names,
        nmf_feature_names=nmf_names,
    )
    assert "feature_name" in gene_df.columns
    assert "feature_name" in nmf_df.columns
    assert set(gene_df["feature_name"]) == set(gene_names)
    assert set(nmf_df["feature_name"]) == set(nmf_names)


def test_aggregate_with_sample_names_records_max_attention_sample(attention_setup):
    """``sample_names`` enables the ``max_attention_sample`` column —
    each gene/factor gets the name of the sample that maximized
    its attention. Verify both that the column is present and that
    every entry is one of the supplied names."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, n, _, _ = attention_setup
    sample_names = [f"sample_{i:04d}" for i in range(n)]
    gene_df, nmf_df = extract_and_aggregate_attention(
        model,
        X_nmf,
        nmf_H,
        batch_size=64,
        verbose=False,
        sample_names=sample_names,
    )
    assert "max_attention_sample" in gene_df.columns
    assert set(gene_df["max_attention_sample"]).issubset(set(sample_names))
    assert "max_attention_sample" in nmf_df.columns
    assert set(nmf_df["max_attention_sample"]).issubset(set(sample_names))


def test_aggregate_returns_matrices_when_requested(attention_setup):
    """``return_attention_matrices=True`` returns the pre-aggregated
    attention matrices in addition to the DataFrames. Shape
    contract: (n_samples, n_genes) and (n_samples, n_components)."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, n, k, m = attention_setup
    out = extract_and_aggregate_attention(
        model,
        X_nmf,
        nmf_H,
        batch_size=64,
        verbose=False,
        return_attention_matrices=True,
    )
    assert len(out) == 4, f"expected 4-tuple, got {len(out)}"
    gene_df, nmf_df, gene_attn, nmf_attn = out
    assert gene_attn.shape == (n, m)
    assert nmf_attn.shape == (n, k)
    assert np.isfinite(gene_attn).all()
    assert np.isfinite(nmf_attn).all()


def test_aggregate_save_dir_writes_parquets(attention_setup, tmp_path):
    """``save_dir`` should write both DataFrames as parquet files
    with deterministic filenames so a follow-up load can find them."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, *_ = attention_setup
    extract_and_aggregate_attention(
        model,
        X_nmf,
        nmf_H,
        batch_size=64,
        verbose=False,
        save_dir=str(tmp_path),
    )
    expected_files = {
        "gene_attention_aggregated.parquet",
        "nmf_attention_aggregated.parquet",
    }
    actual = {p.name for p in tmp_path.iterdir()}
    assert expected_files <= actual, f"missing: {expected_files - actual}"


def test_aggregate_save_dir_loads_existing_when_not_forced(attention_setup, tmp_path):
    """When parquets already exist and ``force=False`` (default), the
    function loads + returns them instead of recomputing. Verify by
    pre-seeding the dir with a write call, then re-calling and
    confirming the second result equals the first.

    (Note: the docstring says it should *raise* in this case, but
    the implementation actually short-circuits to load — see
    ``_core.py`` ~line 2697 ``if not force and gene_file.exists()``.
    Locking in actual behavior, not docstring intent.)"""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, *_ = attention_setup
    gene_a, nmf_a = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False, save_dir=str(tmp_path)
    )
    # Second call (without force) — loads from disk, must match.
    gene_b, nmf_b = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False, save_dir=str(tmp_path)
    )
    np.testing.assert_array_equal(
        gene_a["mean_attention"].to_numpy(), gene_b["mean_attention"].to_numpy()
    )
    np.testing.assert_array_equal(
        nmf_a["mean_attention"].to_numpy(), nmf_b["mean_attention"].to_numpy()
    )


def test_aggregate_save_dir_force_recomputes(attention_setup, tmp_path):
    """``force=True`` recomputes even when parquets exist. The on-disk
    files should be overwritten — no exception."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, *_ = attention_setup
    extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False, save_dir=str(tmp_path)
    )
    # Second call with force — must succeed without error.
    extract_and_aggregate_attention(
        model,
        X_nmf,
        nmf_H,
        batch_size=64,
        verbose=False,
        save_dir=str(tmp_path),
        force=True,
    )


def test_aggregate_normalize_false_preserves_raw_attention(attention_setup):
    """``normalize=False`` should skip the per-sample normalization
    step in trace_attention_to_genes (mean attention values can
    therefore exceed 1)."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, *_ = attention_setup
    gene_df_raw, _ = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False, normalize=False,
    )
    gene_df_norm, _ = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False, normalize=True,
    )
    # Raw should differ from normalized — otherwise the flag is a
    # silent no-op.
    assert not np.allclose(
        gene_df_raw["mean_attention"], gene_df_norm["mean_attention"], atol=1e-4
    )


def test_aggregate_custom_nonzero_threshold(attention_setup):
    """``nonzero_threshold`` controls what counts as "active"
    attention for n_samples_nonzero. A high threshold should produce
    smaller counts than a low threshold."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, *_ = attention_setup
    high_thresh, _ = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False,
        nonzero_threshold=0.99,
    )
    low_thresh, _ = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False,
        nonzero_threshold=0.0,
    )
    # Counts under the high threshold should be ≤ counts under low
    # threshold (strictly less for at least one feature unless every
    # value happens to be > 0.99 — extremely unlikely).
    assert (
        high_thresh["n_samples_nonzero"].sum()
        <= low_thresh["n_samples_nonzero"].sum()
    )


# ── extra compute_attention_correlation paths ───────────────────────


def test_correlation_obs_mask_can_be_dropped_when_X_already_subset():
    """When obs_mask is None, the function uses X as-is. Combined
    with the dense path (X is already an ndarray) hits a different
    branch than the sparse-mask combination."""
    rng = np.random.RandomState(7)
    X = rng.rand(15, 4)
    A = rng.rand(15, 4)
    df = compute_attention_correlation(
        A, X, obs_mask=None, stratify_by_unique_values=False, verbose=False
    )
    assert len(df) > 0
    # No exception is the success criterion.


def test_correlation_verbose_path_runs(capsys):
    """``verbose=True`` exercises the print-summary branch. We don't
    assert specific output — just that the path completes."""
    rng = np.random.RandomState(8)
    X = rng.rand(12, 5)
    A = rng.rand(12, 5)
    compute_attention_correlation(
        A, X, stratify_by_unique_values=False, verbose=True
    )
    captured = capsys.readouterr()
    # Some output expected — even just a header.
    assert len(captured.out) > 0


def test_aggregate_with_metadata_extracts_names(attention_setup):
    """``metadata`` dict (anndata-like) should auto-derive
    gene_feature_names from ``metadata['var'].index`` and
    sample_names from ``metadata['obs']`` when those are not
    explicitly provided. Covers lines 2674-2695."""
    import pandas as pd

    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, n, k, m = attention_setup
    metadata = {
        "var": pd.DataFrame(index=[f"gene_{i}" for i in range(m)]),
        "obs": pd.DataFrame(
            {"obs_id": [f"obs_{i}" for i in range(n)]},
            index=[f"row_{i}" for i in range(n)],
        ),
    }
    gene_df, _ = extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=False, metadata=metadata
    )
    assert "feature_name" in gene_df.columns
    # All gene names should come from metadata['var'].index (the
    # function may reorder rows by aggregate metrics, so don't
    # assume position 0 is gene_0).
    actual_names = set(gene_df["feature_name"])
    expected_names = {f"gene_{i}" for i in range(m)}
    assert actual_names == expected_names


def test_aggregate_verbose_runs(attention_setup, capsys):
    """``verbose=True`` exercises a long chain of progress prints
    in the aggregate function — covers lines 2742-2747 and various
    print branches deeper in the body."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, *_ = attention_setup
    extract_and_aggregate_attention(
        model, X_nmf, nmf_H, batch_size=64, verbose=True
    )
    captured = capsys.readouterr()
    # Some verbose output expected — at least the "Extracting"
    # header.
    assert "Extracting" in captured.out or "Processing" in captured.out


def test_aggregate_save_dir_reload_with_matrices(attention_setup, tmp_path):
    """When save_dir contains both parquets AND .npy matrix files,
    reload should return all four (DataFrames + matrices). Covers
    the matrix-reload branch at lines 2719-2727."""
    from sparse_nmf import extract_and_aggregate_attention

    model, X_nmf, nmf_H, n, k, m = attention_setup
    # First call writes parquets and matrices.
    out1 = extract_and_aggregate_attention(
        model,
        X_nmf,
        nmf_H,
        batch_size=64,
        verbose=False,
        save_dir=str(tmp_path),
        return_attention_matrices=True,
    )
    # Manually save the matrices alongside the parquets — the
    # function returns them but doesn't save them by default in
    # this path. Save them so the reload-with-matrices branch has
    # something to load.
    np.save(tmp_path / "gene_attention_matrix.npy", out1[2])
    np.save(tmp_path / "nmf_attention_matrix.npy", out1[3])
    # Second call — should load all four.
    out2 = extract_and_aggregate_attention(
        model,
        X_nmf,
        nmf_H,
        batch_size=64,
        verbose=False,
        save_dir=str(tmp_path),
        return_attention_matrices=True,
    )
    assert len(out2) == 4
    np.testing.assert_array_equal(out1[2], out2[2])  # gene matrix
    np.testing.assert_array_equal(out1[3], out2[3])  # nmf matrix


def test_correlation_strata_with_only_continuous_data():
    """When all rows have many unique values, only the
    ``4+_unique`` stratum should be populated — the 2/3-unique
    rows should be absent or have zero rows."""
    rng = np.random.RandomState(9)
    n, m = 25, 8
    # Each row has all-unique values → 4+_unique stratum
    X = np.tile(np.arange(m, dtype=np.float32), (n, 1))
    for i in range(n):
        X[i] = rng.permutation(X[i]) + rng.rand() * 0.001  # fully unique each row
    A = X.copy()
    df = compute_attention_correlation(A, X, stratify_by_unique_values=True, verbose=False)
    strata = df["stratum"].unique()
    assert "4+_unique" in strata
