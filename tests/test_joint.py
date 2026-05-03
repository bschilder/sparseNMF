"""Smoke tests for the joint NMF + autoencoder model.

These tests don't try to verify training quality (too noisy in 1-2
epochs on a tiny matrix) — they verify that the public API stays
callable, returns the right shapes, and doesn't crash on edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

# train_joint_model + SparseNMF_Autoencoder are heavy enough that we
# import lazily inside each test, keeping ``test_data.py`` /
# ``test_core.py`` collection fast.


def test_imports_resolve():
    """The advertised public surface must be importable."""
    from sparse_nmf import (
        SparseNMF,
        SparseNMF_Autoencoder,
        compute_attention_correlation,
        compute_joint_loss,
        extract_attention_weights,
        plot_nmf_factor_distributions,
        sparse_nmf,
        trace_attention_to_genes,
        train_joint_model,
        train_sparse_nmf,
    )

    # silence "imported but unused"
    assert callable(SparseNMF)
    assert callable(train_joint_model)
    assert callable(train_sparse_nmf)
    assert callable(compute_joint_loss)
    assert callable(extract_attention_weights)
    assert callable(trace_attention_to_genes)
    assert callable(compute_attention_correlation)
    assert callable(plot_nmf_factor_distributions)
    assert callable(sparse_nmf)
    assert SparseNMF_Autoencoder is not None


@pytest.mark.slow
def test_joint_model_trains_for_one_epoch(small_sparse, device):
    """End-to-end: a single epoch on a 200×400 matrix should run to
    completion, return a (n_samples, latent_dim) embedding, and not
    produce NaNs."""
    from sparse_nmf import train_joint_model

    z, model = train_joint_model(
        small_sparse,
        n_samples=small_sparse.shape[0],
        n_features=small_sparse.shape[1],
        nmf_components=4,
        latent_dim=2,
        device=device,
        n_epochs=1,
        batch_size=64,
        verbose=False,
    )
    z = np.asarray(z)
    assert z.shape == (small_sparse.shape[0], 2), z.shape
    assert np.isfinite(z).all(), "embedding contains NaN/Inf"
    assert model is not None


def test_module_version_is_set():
    import sparse_nmf

    assert isinstance(sparse_nmf.__version__, str)
    assert sparse_nmf.__version__.count(".") >= 2  # ``MAJOR.MINOR.PATCH``
