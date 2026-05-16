"""Tests for :func:`sparse_nmf.sweep_hyperparameters` and its helpers.

The sweep runs many small sparseNMF trainings to compare configs.
These tests use a tiny synthetic input so the whole module covers
fast on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sparse_nmf import SweepResult, sweep_hyperparameters
from sparse_nmf._hyper_sweep import _safe_silhouette, _sparsity


def _tiny_sparse(n_cells=80, n_genes=120, k_true=4, seed=0):
    rng = np.random.default_rng(seed)
    W = np.abs(rng.normal(0, 1, (n_cells, k_true)))
    H = np.abs(rng.normal(0, 1, (k_true, n_genes)))
    X = np.clip(W @ H + rng.normal(0, 0.05, (n_cells, n_genes)), 0, None)
    return csr_matrix(X.astype(np.float32))


# ── core sweep ───────────────────────────────────────────────────────


def test_sweep_standard_mode_basic():
    X = _tiny_sparse()
    labels = np.repeat([0, 1, 2, 3], 20)
    configs = [
        {"n_components": 3, "max_iter": 10, "patience": 3},
        {"n_components": 5, "max_iter": 10, "patience": 3},
    ]
    res = sweep_hyperparameters(
        X,
        configs,
        labels=labels,
        mode="standard",
        dataset_name="t1",
        device="cpu",
        verbose=False,
    )
    assert isinstance(res, SweepResult)
    assert len(res.df) == 2
    assert set(res.df["k"].values) == {3, 5}
    assert (res.df["dataset"] == "t1").all()
    assert (res.df["mode"] == "standard").all()


def test_sweep_batch_aware_mode():
    X = _tiny_sparse(n_cells=90)
    labels = np.repeat([0, 1, 2], 30)
    batch = np.tile([0, 1, 2], 30)
    configs = [
        {
            "n_components": 3,
            "alignment_weight": 1.0,
            "max_iter": 10,
            "patience": 3,
            "sparsity_weight": 0.01,
        },
    ]
    res = sweep_hyperparameters(
        X,
        configs,
        labels=labels,
        batch=batch,
        mode="batch_aware",
        dataset_name="t2",
        device="cpu",
        verbose=False,
    )
    assert len(res.df) == 1
    assert res.df["mode"].iloc[0] == "batch_aware"
    assert res.df["alignment_weight"].iloc[0] == 1.0


def test_sweep_batch_aware_requires_batch():
    X = _tiny_sparse()
    configs = [{"n_components": 3, "max_iter": 5}]
    with pytest.raises(ValueError, match="batch=..."):
        sweep_hyperparameters(
            X,
            configs,
            labels=None,
            batch=None,
            mode="batch_aware",
            device="cpu",
            verbose=False,
        )


def test_sweep_bad_mode_errors():
    X = _tiny_sparse()
    configs = [{"n_components": 3, "max_iter": 5}]
    with pytest.raises(ValueError, match="unknown mode"):
        sweep_hyperparameters(
            X,
            configs,
            mode="bogus",
            device="cpu",
            verbose=False,
        )


def test_sweep_missing_n_components_errors():
    X = _tiny_sparse()
    with pytest.raises(ValueError, match="missing 'n_components'"):
        sweep_hyperparameters(
            X,
            [{"max_iter": 5}],
            mode="standard",
            device="cpu",
            verbose=False,
        )


def test_sweep_records_train_seconds_and_iter():
    X = _tiny_sparse()
    res = sweep_hyperparameters(
        X,
        [{"n_components": 3, "max_iter": 10, "patience": 3}],
        device="cpu",
        verbose=False,
    )
    row = res.df.iloc[0]
    assert row["train_seconds"] > 0
    # n_iter may be int or None depending on which path runs; just
    # confirm the column is populated and not negative.
    assert row["n_iter"] is None or row["n_iter"] >= 1
    assert 0.0 <= row["W_sparsity"] <= 1.0


def test_sweep_no_labels_yields_nan_silhouette():
    X = _tiny_sparse()
    res = sweep_hyperparameters(
        X,
        [{"n_components": 3, "max_iter": 5, "patience": 2}],
        labels=None,
        batch=None,
        device="cpu",
        verbose=False,
    )
    assert np.isnan(res.df["silhouette_label"].iloc[0])
    assert np.isnan(res.df["silhouette_batch"].iloc[0])


def test_sweep_verbose_does_not_crash(capsys):
    X = _tiny_sparse(n_cells=40, n_genes=60)
    res = sweep_hyperparameters(
        X,
        [{"n_components": 3, "max_iter": 5, "patience": 2}],
        labels=np.repeat([0, 1], 20),
        device="cpu",
        verbose=True,
    )
    out = capsys.readouterr().out
    # Verbose mode emits a config-line + summary-line per config.
    assert "sil_label" in out
    assert len(res.df) == 1


def test_sweep_cuda_request_falls_back_on_cpu_host(monkeypatch):
    """Asking for device='cuda' on a CPU-only host should fall through
    to CPU without erroring."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    X = _tiny_sparse(n_cells=40, n_genes=60)
    res = sweep_hyperparameters(
        X,
        [{"n_components": 3, "max_iter": 5, "patience": 2}],
        device="cuda",
        verbose=True,
    )
    assert len(res.df) == 1


# ── helpers ─────────────────────────────────────────────────────────


def test_safe_silhouette_handles_single_label():
    """sklearn requires ≥2 distinct labels; our wrapper returns NaN
    instead of raising."""
    W = np.random.RandomState(0).normal(0, 1, (50, 8))
    labels = np.full(50, "only")
    val = _safe_silhouette(W, labels)
    assert np.isnan(val)


def test_safe_silhouette_subsamples_large_input():
    """Large inputs are downsampled to ``max_cells``. The function should
    still produce a finite number, not crash."""
    W = np.random.RandomState(0).normal(0, 1, (8000, 4))
    labels = np.random.RandomState(0).randint(0, 4, 8000)
    val = _safe_silhouette(W, labels, max_cells=500)
    assert np.isfinite(val)


def test_safe_silhouette_empty_input_returns_nan():
    W = np.empty((0, 4))
    assert np.isnan(_safe_silhouette(W, np.array([])))


def test_safe_silhouette_labels_none_returns_nan():
    W = np.random.RandomState(0).normal(0, 1, (20, 4))
    assert np.isnan(_safe_silhouette(W, None))


def test_sparsity_helper():
    W = np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 3.0]])
    # 3 of 6 entries are < 1e-3 → 0.5
    assert _sparsity(W) == 0.5
    # threshold=2.5: entries below are 0, 0, 1, 0, 2 (five of six) → 5/6
    assert _sparsity(W, threshold=2.5) == pytest.approx(5 / 6)


# ── plotting ────────────────────────────────────────────────────────


def test_sweep_result_plot_writes_files(tmp_path):
    """SweepResult.plot() should emit the 3 expected PNGs into the dir."""
    import matplotlib

    matplotlib.use("Agg")

    X = _tiny_sparse(n_cells=60, n_genes=100)
    labels = np.repeat([0, 1, 2], 20)
    res = sweep_hyperparameters(
        X,
        [
            {
                "n_components": 3,
                "normalize_inputs": True,
                "nonzero_mse_weight": 0.0,
                "max_iter": 5,
                "patience": 2,
            },
            {
                "n_components": 5,
                "normalize_inputs": False,
                "nonzero_mse_weight": 0.0,
                "max_iter": 5,
                "patience": 2,
            },
            {
                "n_components": 30,
                "normalize_inputs": True,
                "nonzero_mse_weight": 0.0,
                "max_iter": 5,
                "patience": 2,
            },
        ],
        labels=labels,
        dataset_name="d1",
        device="cpu",
        verbose=False,
    )
    paths = res.plot(tmp_path)
    assert (tmp_path / "sweep_k.png").exists()
    assert (tmp_path / "sweep_tradeoff.png").exists()
    # sweep_loss_norm is only emitted when there's at least one k=30 row
    assert "sweep_loss_norm" in paths
    assert (tmp_path / "sweep_loss_norm.png").exists()
