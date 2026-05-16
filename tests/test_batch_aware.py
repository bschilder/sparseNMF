"""Tests for :func:`sparse_nmf.train_sparse_nmf_batch_aware`.

The batch-aware variant decomposes ``X_c ≈ W_c · (H_shared + V[b])``
where V[b] is a per-batch additive correction. These tests cover the
shape/type contract, the per-batch grouping logic, the
``alignment_weight`` and ``patience`` knobs, and the determinism
guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sparse_nmf import BatchAwareResult, train_sparse_nmf_batch_aware


def _make_batched_sparse(n_cells=120, n_genes=200, n_batches=3, k_true=5, seed=0):
    """Tiny non-negative sparse counts with a clear per-batch additive shift
    on the first 20 genes. Returns (X_sparse, batch_labels)."""
    rng = np.random.default_rng(seed)
    W = np.abs(rng.normal(0, 1, (n_cells, k_true)))
    H = np.abs(rng.normal(0, 1, (k_true, n_genes)))
    X = W @ H + rng.normal(0, 0.05, (n_cells, n_genes))
    batch = np.repeat(np.array([f"b{i}" for i in range(n_batches)]),
                      n_cells // n_batches)
    if len(batch) < n_cells:
        batch = np.concatenate([batch, np.full(n_cells - len(batch), batch[-1])])
    # Add a per-batch additive shift to the first 20 genes — gives V[b]
    # something to absorb so the test can verify it does.
    for i, b in enumerate(np.unique(batch)):
        X[batch == b, :20] += float(i)
    X = np.clip(X, 0, None)
    return csr_matrix(X.astype(np.float32)), batch


def test_returns_batch_aware_result():
    X, batch = _make_batched_sparse()
    res = train_sparse_nmf_batch_aware(
        X, batch, n_components=6, max_iter=30, patience=5,
        device="cpu", verbose=False,
    )
    assert isinstance(res, BatchAwareResult)
    assert res.W.shape == (X.shape[0], 6)
    assert res.H_shared.shape == (6, X.shape[1])


def test_v_dict_keys_match_batches():
    X, batch = _make_batched_sparse(n_batches=4)
    res = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, max_iter=20, patience=3,
        device="cpu", verbose=False,
    )
    assert set(res.V.keys()) == set(np.unique(batch).astype(str))
    for v in res.V.values():
        assert v.shape == (4, X.shape[1])


def test_alignment_weight_extremes_shrink_v():
    """Very large alignment_weight should force V[b] near zero;
    very small should let it grow. The L2 penalty is the only thing
    constraining V[b]'s magnitude."""
    X, batch = _make_batched_sparse()
    res_strong = train_sparse_nmf_batch_aware(
        X, batch, n_components=5, alignment_weight=100.0,
        max_iter=40, patience=8, device="cpu", verbose=False, random_state=0,
    )
    res_weak = train_sparse_nmf_batch_aware(
        X, batch, n_components=5, alignment_weight=0.01,
        max_iter=40, patience=8, device="cpu", verbose=False, random_state=0,
    )
    strong_v_l2 = sum(np.linalg.norm(v) for v in res_strong.V.values())
    weak_v_l2 = sum(np.linalg.norm(v) for v in res_weak.V.values())
    assert strong_v_l2 < weak_v_l2, (
        f"strong α_v should shrink V more: {strong_v_l2=}, {weak_v_l2=}"
    )


def test_random_state_deterministic():
    X, batch = _make_batched_sparse()
    res_a = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, max_iter=20, patience=5,
        device="cpu", verbose=False, random_state=42,
    )
    res_b = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, max_iter=20, patience=5,
        device="cpu", verbose=False, random_state=42,
    )
    np.testing.assert_allclose(res_a.W, res_b.W, atol=1e-5)
    np.testing.assert_allclose(res_a.H_shared, res_b.H_shared, atol=1e-5)


def test_patience_early_stops():
    """patience=1 with tol=10.0 (impossible to beat) should stop after the
    first non-improvement check."""
    X, batch = _make_batched_sparse()
    res = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, max_iter=500, patience=1, tol=10.0,
        device="cpu", verbose=False,
    )
    assert res.n_iter < 50, f"early stop didn't fire; ran {res.n_iter} iters"


def test_max_iter_caps_when_patience_disabled():
    X, batch = _make_batched_sparse()
    res = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, max_iter=15, patience=None,
        device="cpu", verbose=False,
    )
    # max_iter=15 means iterations 0..14, n_iter is the count of the last one+1
    assert res.n_iter == 15


def test_rejects_dense_input():
    X = np.random.rand(50, 100).astype(np.float32)
    batch = np.repeat(["a", "b"], 25)
    with pytest.raises(TypeError, match="scipy.sparse"):
        train_sparse_nmf_batch_aware(X, batch, n_components=4, device="cpu")


def test_batch_shape_mismatch_errors():
    X, _ = _make_batched_sparse(n_cells=120)
    wrong_batch = np.repeat(["a", "b"], 30)  # 60 != 120
    with pytest.raises(ValueError, match="batch shape"):
        train_sparse_nmf_batch_aware(X, wrong_batch, n_components=4, device="cpu")


def test_normalize_inputs_toggle_changes_w():
    """Switching normalize_inputs should produce different embeddings
    (the L2 row-norm fundamentally changes the input scale)."""
    X, batch = _make_batched_sparse()
    res_norm = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, normalize_inputs=True,
        max_iter=20, patience=3, device="cpu", verbose=False, random_state=0,
    )
    res_raw = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, normalize_inputs=False,
        max_iter=20, patience=3, device="cpu", verbose=False, random_state=0,
    )
    diff = float(np.abs(res_norm.W - res_raw.W).mean())
    assert diff > 1e-3, "normalize_inputs toggle had no visible effect on W"


def test_single_batch_falls_through():
    """n_batches=1 still produces a result (the per-batch loop runs once
    with the whole dataset; V has one key)."""
    rng = np.random.default_rng(0)
    X = csr_matrix(np.abs(rng.normal(0, 1, (80, 150)).astype(np.float32)))
    batch = np.full(80, "only")
    res = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, max_iter=15, patience=3,
        device="cpu", verbose=False,
    )
    assert isinstance(res, BatchAwareResult)
    assert list(res.V.keys()) == ["only"]


def test_losses_recorded_when_verbose():
    X, batch = _make_batched_sparse()
    res = train_sparse_nmf_batch_aware(
        X, batch, n_components=4, max_iter=30, patience=5,
        device="cpu", verbose=True,
    )
    assert len(res.losses) > 0
    # Loss should generally decrease — at least the last value should
    # be < the first (allowing MU's transient bumps).
    assert res.losses[-1] < res.losses[0]


def test_cuda_unavailable_falls_back_to_cpu():
    """Asking for cuda on a CPU-only host should fall back gracefully
    rather than erroring."""
    X, batch = _make_batched_sparse(n_cells=60, n_genes=80)
    res = train_sparse_nmf_batch_aware(
        X, batch, n_components=3, max_iter=10, patience=3,
        device="cuda",  # may or may not exist
        verbose=False,
    )
    # If we got here, the fallback worked (either ran on cuda or cpu).
    assert res.W.shape == (60, 3)
