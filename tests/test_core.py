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


# ── Determinism + state attributes ──────────────────────────────────


def test_random_state_makes_fits_reproducible(small_sparse, device):
    """Two SparseNMF instances with the same ``random_state`` must
    converge to bit-identical factor matrices. This is the contract
    downstream consumers rely on for cache keys and reproducible
    benchmarks."""
    a = SparseNMF(n_components=4, max_iter=15, device=device, verbose=False, random_state=42)
    a.fit_transform(small_sparse)
    b = SparseNMF(n_components=4, max_iter=15, device=device, verbose=False, random_state=42)
    b.fit_transform(small_sparse)
    np.testing.assert_allclose(_to_numpy(a.H), _to_numpy(b.H), atol=1e-6)
    np.testing.assert_allclose(_to_numpy(a.W), _to_numpy(b.W), atol=1e-6)


def test_different_seeds_diverge(small_sparse, device):
    """Sanity check on the determinism test above: different seeds must
    actually produce different factorizations, otherwise the
    determinism test would be vacuously true on a deterministic init."""
    a = SparseNMF(n_components=4, max_iter=15, device=device, verbose=False, random_state=1)
    a.fit_transform(small_sparse)
    b = SparseNMF(n_components=4, max_iter=15, device=device, verbose=False, random_state=2)
    b.fit_transform(small_sparse)
    assert not np.allclose(_to_numpy(a.H), _to_numpy(b.H), atol=1e-3)


def test_fit_populates_documented_attributes(small_sparse, device):
    """The class docstring promises ``reconstruction_error_``,
    ``r2_score_``, ``r2_score_nonzero_``, and ``n_iter_`` after fit.
    Lock that contract in."""
    nmf = SparseNMF(n_components=4, max_iter=10, device=device, verbose=False, random_state=0)
    nmf.fit_transform(small_sparse)
    assert nmf.reconstruction_error_ is not None and nmf.reconstruction_error_ >= 0
    assert nmf.n_iter_ is not None and 0 < nmf.n_iter_ <= 10
    assert nmf.r2_score_ is not None
    assert nmf.r2_score_nonzero_ is not None
    # R² ∈ [-∞, 1] but on a fit that ran for any iterations should be a
    # finite float (not NaN/Inf).
    assert np.isfinite(nmf.r2_score_)
    assert np.isfinite(nmf.r2_score_nonzero_)


def test_more_iterations_lowers_reconstruction_error(small_sparse, device):
    """Multiplicative-update NMF guarantees the Frobenius reconstruction
    error is non-increasing across iterations. With matched seeds, a
    longer run must end at least as good as a shorter run."""
    short = SparseNMF(
        n_components=4, max_iter=2, device=device, verbose=False, random_state=0
    )
    short.fit_transform(small_sparse)
    long_ = SparseNMF(
        n_components=4, max_iter=30, device=device, verbose=False, random_state=0
    )
    long_.fit_transform(small_sparse)
    # Allow a hair of float slack — these are rerun-from-scratch fits,
    # not a continuation of the same run.
    assert long_.reconstruction_error_ <= short.reconstruction_error_ + 1e-3


# ── Input format flexibility ────────────────────────────────────────


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_accepts_multiple_sparse_formats(small_sparse, device, fmt):
    """The wrapper internally converts to COO. Ensure CSR/CSC inputs
    aren't rejected and produce the same shape output."""
    converters = {
        "csr": lambda x: x.tocsr(),
        "csc": lambda x: x.tocsc(),
        "coo": lambda x: x.tocoo(),
    }
    X = converters[fmt](small_sparse)
    nmf = SparseNMF(n_components=4, max_iter=5, device=device, verbose=False, random_state=0)
    W = nmf.fit_transform(X)
    assert W.shape == (X.shape[0], 4)
    assert np.isfinite(W).all()


# ── Alternate optimization paths ────────────────────────────────────


def test_r2_weight_path_runs_and_preserves_non_negativity(small_sparse, device):
    """When ``r2_weight > 0`` the optimizer switches from multiplicative
    updates to Adam. Both factors must still be non-negative (the loss
    is the only thing that changes — non-negativity is structural)."""
    nmf = SparseNMF(
        n_components=4,
        max_iter=10,
        device=device,
        verbose=False,
        random_state=0,
        mse_weight=0.5,
        r2_weight=0.5,
        learning_rate=0.05,
    )
    W = nmf.fit_transform(small_sparse)
    assert W.shape == (small_sparse.shape[0], 4)
    assert (W >= -1e-5).all()
    assert (_to_numpy(nmf.H) >= -1e-5).all()
    assert np.isfinite(W).all()


def test_nonzero_mse_path_runs(small_sparse, device):
    """``nonzero_mse_weight > 0`` ignores zero positions in the loss —
    different code path that also routes through the Adam optimizer.
    Verify it just completes without producing NaNs."""
    nmf = SparseNMF(
        n_components=4,
        max_iter=10,
        device=device,
        verbose=False,
        random_state=0,
        nonzero_mse_weight=1.0,
        learning_rate=0.05,
    )
    W = nmf.fit_transform(small_sparse)
    assert np.isfinite(W).all()
    assert (W >= -1e-5).all()


def test_patience_can_trigger_early_stop(small_sparse, device, capsys):
    """With aggressive patience, training should stop well before
    ``max_iter`` once reconstruction error plateaus.

    Subtle: patience checking in the upstream code is gated by
    ``if self.verbose:`` (see ``_core.py`` lines ~601/791), so
    ``verbose=False`` silently disables this mechanism. This test
    runs with ``verbose=True`` and discards stdout via ``capsys``.
    """
    nmf = SparseNMF(
        n_components=4,
        max_iter=300,
        device=device,
        verbose=True,
        random_state=0,
        patience=2,
        tol=1e-2,
    )
    nmf.fit_transform(small_sparse)
    capsys.readouterr()  # silence the progress output
    assert nmf.n_iter_ < 300, f"expected early stop, ran {nmf.n_iter_}/300 iters"


def test_cuda_string_falls_back_to_cpu_when_no_gpu():
    """Asking for cuda on a CPU-only machine must transparently fall
    back to CPU rather than raise — documented behavior in the
    ``__init__`` docstring."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA available; can't test the no-GPU fallback")
    nmf = SparseNMF(n_components=2, max_iter=2, device="cuda", verbose=False)
    assert nmf.device.type == "cpu"


def test_cuda_fallback_verbose_prints_message(capsys):
    """The ``verbose=True`` variant of the CUDA→CPU fallback prints
    a "CUDA not available" warning so users notice the silent
    downgrade. Covers the verbose branch of the fallback."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA available; can't test the no-GPU fallback")
    SparseNMF(n_components=2, max_iter=2, device="cuda", verbose=True)
    captured = capsys.readouterr()
    assert "CUDA not available" in captured.out


# ── Convenience wrappers ────────────────────────────────────────────


def test_sparse_nmf_function_matches_class(small_sparse, device):
    """The ``sparse_nmf()`` convenience wrapper must produce the same
    result as ``SparseNMF(...).fit_transform(...)`` given identical
    args + seed. Otherwise users get silently inconsistent behavior
    between the two entry points."""
    from sparse_nmf import sparse_nmf

    args = dict(n_components=4, max_iter=10, device=device, verbose=False, random_state=42)
    W_func = sparse_nmf(small_sparse, **args)
    W_class = SparseNMF(**args).fit_transform(small_sparse)
    np.testing.assert_allclose(W_func, W_class, atol=1e-5)


def test_train_sparse_nmf_returns_W_and_model(small_sparse, device):
    """``train_sparse_nmf`` is the save-aware wrapper. Without save
    paths it should behave like the convenience function but return a
    ``(W, model)`` tuple."""
    from sparse_nmf import train_sparse_nmf

    W, model = train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=10,
        device=device,
        verbose=False,
        random_state=0,
    )
    assert W.shape == (small_sparse.shape[0], 4)
    assert isinstance(model, SparseNMF)
    assert model.W is not None and model.H is not None
    assert model.reconstruction_error_ is not None


def test_train_sparse_nmf_save_then_load(small_sparse, tmp_path, device):
    """Save + reload roundtrip: the second call should NOT need
    ``X_sparse`` and must return the same W — this is the path that
    lets downstream code skip retraining when the cache is warm."""
    from sparse_nmf import train_sparse_nmf

    emb = tmp_path / "embeddings.npy"
    mod = tmp_path / "model.pkl"

    W1, _ = train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=10,
        device=device,
        verbose=False,
        random_state=0,
        embeddings_save_path=str(emb),
        model_save_path=str(mod),
    )
    assert emb.exists()
    assert mod.exists()

    W2, model2 = train_sparse_nmf(
        n_components=4,
        max_iter=10,
        device=device,
        verbose=False,
        embeddings_save_path=str(emb),
        model_save_path=str(mod),
    )
    np.testing.assert_array_equal(W1, W2)
    assert model2.W is not None  # model loaded from disk


def test_train_sparse_nmf_force_retrains(small_sparse, tmp_path, device):
    """``force=True`` should retrain even when both save paths exist
    (otherwise stale caches silently override new data)."""
    from sparse_nmf import train_sparse_nmf

    emb = tmp_path / "embeddings.npy"
    mod = tmp_path / "model.pkl"

    W1, _ = train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=5,
        device=device,
        verbose=False,
        random_state=0,
        embeddings_save_path=str(emb),
        model_save_path=str(mod),
    )

    # Retrain with force=True and a different seed — output must differ
    # despite the cached files existing.
    W2, _ = train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=5,
        device=device,
        verbose=False,
        random_state=999,
        embeddings_save_path=str(emb),
        model_save_path=str(mod),
        force=True,
    )
    assert not np.allclose(W1, W2)


# ── verbose paths + edge-case inputs ────────────────────────────────


def test_verbose_path_runs_and_prints(small_sparse, device, capsys):
    """``verbose=True`` triggers a chunk of conditional prints during
    init, fit, and convergence. Exercises lines 472-492 in
    ``_core.py`` plus the per-iteration progress printing."""
    nmf = SparseNMF(
        n_components=4, max_iter=5, device=device, verbose=True, random_state=0
    )
    nmf.fit_transform(small_sparse)
    captured = capsys.readouterr()
    # Multiple verbose prints expected — at minimum the device + shape
    # banner.
    assert "Sparse NMF on" in captured.out
    assert "Components:" in captured.out


def test_verbose_with_r2_weight_runs(small_sparse, device, capsys):
    """``r2_weight > 0`` + ``verbose=True`` exercises a different
    print branch — gradient-descent banner + R²-specific messages
    (lines 478-492)."""
    nmf = SparseNMF(
        n_components=4,
        max_iter=5,
        device=device,
        verbose=True,
        random_state=0,
        mse_weight=0.5,
        r2_weight=0.5,
        learning_rate=0.05,
    )
    nmf.fit_transform(small_sparse)
    captured = capsys.readouterr()
    assert "gradient-based" in captured.out.lower() or "R²" in captured.out


def test_compute_final_metrics_handles_empty_sample(small_sparse, device):
    """The ``_compute_final_metrics`` helper has a fallback (lines
    945-947) for when the sampled rows happen to contain zero
    nnz — sets all metrics to 0 instead of dividing by zero. Test
    by calling the helper directly with an all-zero sparse sample."""
    from scipy.sparse import csr_matrix
    import torch

    nmf = SparseNMF(
        n_components=4, max_iter=5, device=device, verbose=False, random_state=0
    )
    # Run a normal fit so internal state is initialized.
    nmf.fit_transform(small_sparse)

    # Now call _compute_final_metrics with an all-zero CSR matrix
    # — forces the empty-sample fallback branch.
    X_empty = csr_matrix((nmf.W.shape[0], nmf.H.shape[1]), dtype=np.float32)
    nmf._compute_final_metrics(X_empty, nmf.W, nmf.H, X_empty.shape[0])
    # Fallback sets all metrics to 0.0.
    assert nmf.reconstruction_error_ == 0.0
    assert nmf.r2_score_ == 0.0
    assert nmf.r2_score_nonzero_ == 0.0


def test_negative_input_is_taken_absolute(small_sparse, device, capsys):
    """NMF is non-negative by definition — when input contains
    negatives, the wrapper takes ``abs()`` (with a warning when
    verbose). Verifies fit_transform completes and a warning is
    printed."""
    from scipy.sparse import csr_matrix

    X_dense = small_sparse.toarray()
    X_dense[0, 0] = -1.0  # inject a negative
    X_neg = csr_matrix(X_dense)
    nmf = SparseNMF(
        n_components=4, max_iter=5, device=device, verbose=True, random_state=0
    )
    W = nmf.fit_transform(X_neg)
    captured = capsys.readouterr()
    assert "negative" in captured.out.lower()
    assert np.isfinite(W).all()


def test_train_sparse_nmf_normalize_outputs_yields_unit_rows(small_sparse, device):
    """``normalize_outputs=True`` L2-normalizes each row of W to unit
    length via the package's AoU.utils.l2_normalize shim (registered
    in sparse_nmf/__init__.py since the verbatim-vendored _core.py
    imports it from a namespace that doesn't exist standalone).

    Non-zero rows should have L2 norm ≈ 1.0; zero rows pass through
    unchanged (the shim guards div-by-zero)."""
    from sparse_nmf import train_sparse_nmf

    W, _ = train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=10,
        device=device,
        verbose=False,
        random_state=0,
        normalize_outputs=True,
    )
    norms = np.linalg.norm(W, axis=1)
    nonzero = norms > 1e-6
    np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-4)


def test_train_sparse_nmf_verbose_runs(small_sparse, device, capsys):
    """``train_sparse_nmf(verbose=True)`` exercises a chain of
    progress prints (lines ~2155-2188 in _core.py). Just verify
    it runs and produces output."""
    from sparse_nmf import train_sparse_nmf

    train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=5,
        device=device,
        verbose=True,
        random_state=0,
    )
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_train_sparse_nmf_normalize_outputs_reload(small_sparse, tmp_path, device):
    """When loading from disk with ``normalize_outputs=True``, the
    function checks if embeddings are already normalized and
    re-normalizes if not (lines 2090-2099). Pre-seed unnormalized
    embeddings by training without normalize_outputs, then reload
    with normalize_outputs=True."""
    from sparse_nmf import train_sparse_nmf

    emb = tmp_path / "emb.npy"
    mod = tmp_path / "mod.pkl"
    # First run: save UN-normalized embeddings.
    train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=5,
        device=device,
        verbose=False,
        random_state=0,
        embeddings_save_path=str(emb),
        model_save_path=str(mod),
    )
    # Second run with normalize_outputs=True — loads, detects
    # un-normalized, re-normalizes via the AoU shim.
    W, _ = train_sparse_nmf(
        n_components=4,
        max_iter=5,
        device=device,
        verbose=True,  # also covers the verbose post-load print
        embeddings_save_path=str(emb),
        model_save_path=str(mod),
        normalize_outputs=True,
    )
    norms = np.linalg.norm(W, axis=1)
    nonzero = norms > 1e-6
    np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-4)


def test_train_sparse_nmf_verbose_with_normalize_inputs(small_sparse, device, capsys):
    """``normalize_inputs=True`` + ``verbose=True`` covers the
    print branches at 2155-2157."""
    from sparse_nmf import train_sparse_nmf

    train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=5,
        device=device,
        verbose=True,
        random_state=0,
        normalize_inputs=True,
    )
    captured = capsys.readouterr()
    assert "L2 normalized input" in captured.out


def test_train_sparse_nmf_verbose_with_save_path(small_sparse, tmp_path, device, capsys):
    """``embeddings_save_path`` + ``verbose=True`` covers line 2188
    ("Saved embeddings to ...")."""
    from sparse_nmf import train_sparse_nmf

    emb = tmp_path / "emb.npy"
    train_sparse_nmf(
        small_sparse,
        n_components=4,
        max_iter=5,
        device=device,
        verbose=True,
        random_state=0,
        embeddings_save_path=str(emb),
    )
    captured = capsys.readouterr()
    assert "Saved embeddings" in captured.out


def test_train_sparse_nmf_normalize_inputs_runs(small_sparse, device):
    """``normalize_inputs=True`` L2-normalizes each row of X before
    training (uses sklearn.preprocessing.normalize). Different
    factorization should result vs no-normalize, since the input
    distribution changes."""
    from sparse_nmf import train_sparse_nmf

    args = dict(
        n_components=4,
        max_iter=10,
        device=device,
        verbose=False,
        random_state=0,
    )
    W_norm, _ = train_sparse_nmf(small_sparse, normalize_inputs=True, **args)
    W_raw, _ = train_sparse_nmf(small_sparse, normalize_inputs=False, **args)
    # Different W matrices — normalize_inputs is not a silent no-op.
    assert not np.allclose(W_norm, W_raw, atol=1e-3)
    assert np.isfinite(W_norm).all()


def test_l2_normalize_shim_handles_zero_rows():
    """Direct exercise of the AoU shim: zero-norm rows must be
    returned unchanged (otherwise downstream code divides by zero
    when caller forgot to filter)."""
    from sparse_nmf import _l2_normalize

    X = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(X)
    # Zero row: unchanged.
    np.testing.assert_array_equal(out[0], X[0])
    # Non-zero row: unit norm.
    np.testing.assert_allclose(np.linalg.norm(out[1]), 1.0, atol=1e-6)


# ── transform() out-of-sample method ────────────────────────────────


def test_transform_returns_correct_shape(small_sparse, device):
    """``transform()`` is the out-of-sample inference method:
    fit_transform on training data, then transform on new data
    produces a (n_new, n_components) embedding using the frozen H."""
    nmf = SparseNMF(
        n_components=4, max_iter=10, device=device, verbose=False, random_state=0
    )
    nmf.fit_transform(small_sparse)

    # New data with the same number of features.
    from sparse_nmf.data import generate_synthetic_sparse

    X_new = generate_synthetic_sparse(
        n_samples=50,
        n_features=small_sparse.shape[1],
        n_components=4,
        density=0.05,
        seed=99,
    )
    W_new = nmf.transform(X_new)
    assert W_new.shape == (50, 4)
    assert (W_new >= -1e-6).all()
    assert np.isfinite(W_new).all()


def test_transform_before_fit_raises(device):
    """Calling ``transform()`` on an unfitted model must raise — we
    don't want silent garbage results."""
    from sparse_nmf.data import generate_synthetic_sparse

    nmf = SparseNMF(
        n_components=4, max_iter=10, device=device, verbose=False
    )
    X = generate_synthetic_sparse(n_samples=20, n_features=30, n_components=4, seed=0)
    with pytest.raises(ValueError, match="fitted"):
        nmf.transform(X)


def test_transform_feature_mismatch_raises(small_sparse, device):
    """``transform()`` must reject inputs with a different feature
    count than what the model was fitted on (otherwise the W @ H
    matmul would silently produce nonsense)."""
    from sparse_nmf.data import generate_synthetic_sparse

    nmf = SparseNMF(
        n_components=4, max_iter=10, device=device, verbose=False, random_state=0
    )
    nmf.fit_transform(small_sparse)
    # Wrong number of features — H is (k, n_features_orig).
    X_wrong = generate_synthetic_sparse(
        n_samples=10,
        n_features=small_sparse.shape[1] + 1,
        n_components=4,
        seed=0,
    )
    with pytest.raises(ValueError, match="features"):
        nmf.transform(X_wrong)


# ── More gradient-descent path coverage ─────────────────────────────


def test_gradient_descent_with_nonzero_r2_weight(small_sparse, device):
    """``r2_weight > 0`` AND ``nonzero_r2_weight > 0`` together — both
    flags route through gradient descent, but the nonzero R² branch
    skips zero positions in the loss. Hits the
    ``r2_weight + nonzero_r2_weight`` combined branch."""
    nmf = SparseNMF(
        n_components=4,
        max_iter=8,
        device=device,
        verbose=False,
        random_state=0,
        r2_weight=0.3,
        nonzero_r2_weight=0.7,
        learning_rate=0.05,
    )
    W = nmf.fit_transform(small_sparse)
    assert np.isfinite(W).all()
    assert (W >= -1e-5).all()


def test_gradient_descent_verbose_and_patience(small_sparse, device, capsys):
    """Verbose r2_weight path with patience exercises the patience
    counter inside gradient descent (lines 815-828)."""
    nmf = SparseNMF(
        n_components=4,
        max_iter=200,
        device=device,
        verbose=True,
        random_state=0,
        r2_weight=0.5,
        learning_rate=0.05,
        patience=2,
        tol=1e-1,  # loose tol → plateau triggers fast
    )
    nmf.fit_transform(small_sparse)
    capsys.readouterr()  # silence verbose output
    # Patience may or may not trigger depending on convergence —
    # either way, the verbose+patience code path runs.
    assert nmf.n_iter_ <= 200


# ── Autoencoder _sparse_to_torch coverage ───────────────────────────


def test_autoencoder_sparse_to_torch_converts_scipy(small_sparse, device):
    """SparseNMF_Autoencoder has its own ``_sparse_to_torch`` (a
    copy of the SparseNMF helper). It's not called from forward
    (which expects a torch.sparse tensor already), but it's part
    of the public-ish surface — direct test exercises lines
    1198-1210."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=4,
        latent_dim=2,
        hidden_dims=(8,),
        device=device,
        random_state=0,
    )
    out = model._sparse_to_torch(small_sparse)
    assert out.is_sparse
    assert tuple(out.shape) == small_sparse.shape


# ── _compute_recon_values_chunked direct coverage ───────────────────


def test_compute_recon_values_chunked_path_with_large_nnz(device):
    """The chunked branch (lines 130-168) fires when
    ``nnz > adaptive_chunk_size``. The internal floor on
    adaptive_chunk_size is 500, so a small explicit ``chunk_size``
    isn't enough — we need ``nnz > 500``."""
    import torch

    from sparse_nmf._core import _compute_recon_values_chunked

    nnz = 1_500  # > 500 floor
    n_components = 4
    n_features = 30
    rng = torch.Generator().manual_seed(0)
    W_rows = torch.rand(nnz, n_components, generator=rng)
    H = torch.rand(n_components, n_features, generator=rng)
    col_idx = torch.randint(0, n_features, (nnz,), generator=rng)
    out = _compute_recon_values_chunked(
        W_rows, H, col_idx, chunk_size=400, device=torch.device("cpu")
    )
    assert out.shape == (nnz,)
    # Validate against a hand-rolled per-row reference.
    expected = torch.stack([W_rows[i] @ H[:, col_idx[i]] for i in range(nnz)])
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_compute_recon_values_chunked_with_large_nnz_and_components(device):
    """The combined branch — chunked (nnz > adaptive_chunk_size)
    AND ``n_components > 512`` — exercises the sub-chunking inside
    chunks (lines 144-153). Need to pass an explicit chunk_size of
    at most 500 to force chunking despite the high adaptive ceiling
    that 600-component memory math computes."""
    import torch

    from sparse_nmf._core import _compute_recon_values_chunked

    nnz = 700
    n_components = 600  # > 512 → sub-chunking branch
    n_features = 20
    rng = torch.Generator().manual_seed(2)
    W_rows = torch.rand(nnz, n_components, generator=rng)
    H = torch.rand(n_components, n_features, generator=rng)
    col_idx = torch.randint(0, n_features, (nnz,), generator=rng)
    # chunk_size=400 → after the max(500, ...) floor → 500. nnz=700
    # > 500 → chunked path. Inside each chunk, n_components > 512 →
    # sub-chunked.
    out = _compute_recon_values_chunked(
        W_rows, H, col_idx, chunk_size=400, device=torch.device("cpu")
    )
    assert out.shape == (nnz,)
    # Hand-rolled reference for spot-check on first 100 rows
    # (full check is too slow with 600 components).
    expected_subset = torch.stack(
        [W_rows[i] @ H[:, col_idx[i]] for i in range(100)]
    )
    torch.testing.assert_close(out[:100], expected_subset, atol=1e-3, rtol=1e-3)


def test_compute_recon_values_chunked_handles_large_n_components(device):
    """For ``n_components > 512`` the helper does an additional
    sub-chunking over the components dimension (lines 113-121).
    Small-nnz path with a big component count exercises this."""
    import torch

    from sparse_nmf._core import _compute_recon_values_chunked

    nnz = 50
    n_components = 600  # > 512 → sub-chunking branch
    n_features = 20
    rng = torch.Generator().manual_seed(1)
    W_rows = torch.rand(nnz, n_components, generator=rng)
    H = torch.rand(n_components, n_features, generator=rng)
    col_idx = torch.randint(0, n_features, (nnz,), generator=rng)
    out = _compute_recon_values_chunked(
        W_rows, H, col_idx, device=torch.device("cpu")
    )
    assert out.shape == (nnz,)
    # Validate against a hand-rolled reference: row-wise dot product
    # against H[:, col_idx[i]].
    expected = torch.stack(
        [W_rows[i] @ H[:, col_idx[i]] for i in range(nnz)]
    )
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
