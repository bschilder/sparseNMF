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


# ── SparseNMF_Autoencoder forward pass ──────────────────────────────


def _make_sparse_torch(X_csr, device):
    """Convert scipy CSR to torch.sparse_coo on the right device.

    Pulled out of every test so each test stays focused on behavior."""
    import torch

    coo = X_csr.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device)


def test_autoencoder_forward_returns_documented_5tuple(small_sparse, device):
    """Non-VAE forward must return ``(z, W_recon, X_recon, W, H)``
    with X_recon=None (computed on-demand in loss). Shape contract:
    z is (n_samples, latent_dim), W_recon and W are
    (n_samples, nmf_components), H is (nmf_components, n_features)."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    nmf_comp, latent_dim = 4, 2
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=nmf_comp,
        latent_dim=latent_dim,
        hidden_dims=(8, 4),
        device=device,
        random_state=0,
    )
    X_torch = _make_sparse_torch(small_sparse, device)
    out = model(X_torch)
    assert len(out) == 5, f"non-VAE forward should return 5 values, got {len(out)}"
    z, W_recon, X_recon, W, H = out
    assert z.shape == (n, latent_dim)
    assert W_recon.shape == (n, nmf_comp)
    assert X_recon is None  # computed on demand inside loss
    assert W.shape == (n, nmf_comp)
    assert H.shape == (nmf_comp, m)
    # All outputs must be finite — catches NaN-from-init regressions
    assert torch.isfinite(z).all()
    assert torch.isfinite(W_recon).all()


def test_autoencoder_vae_forward_returns_7tuple(small_sparse, device):
    """VAE branch returns ``(z, W_recon, X_recon, W, H, mu, logvar)``.
    mu and logvar parametrize the latent distribution; both must be
    (n_samples, latent_dim)."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    latent_dim = 3
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=4,
        latent_dim=latent_dim,
        hidden_dims=(8,),
        use_vae=True,
        device=device,
        random_state=0,
    )
    X_torch = _make_sparse_torch(small_sparse, device)
    out = model(X_torch)
    assert len(out) == 7, f"VAE forward should return 7 values, got {len(out)}"
    z, W_recon, X_recon, W, H, mu, logvar = out
    assert mu.shape == (n, latent_dim)
    assert logvar.shape == (n, latent_dim)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()


def test_autoencoder_normalize_nmf_components_changes_z(small_sparse, device):
    """``normalize_nmf_components=True`` L2-normalizes W before the
    encoder; the latent z must therefore differ from the un-normalized
    case. If they were identical the flag is a silent no-op."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    common = dict(
        n_samples=n,
        n_features=m,
        nmf_components=4,
        latent_dim=2,
        hidden_dims=(8,),
        device=device,
        random_state=0,
    )
    plain = SparseNMF_Autoencoder(**common, normalize_nmf_components=False)
    normd = SparseNMF_Autoencoder(**common, normalize_nmf_components=True)
    # Force both to start from identical W to isolate the
    # normalization effect (default init is the only difference).
    with torch.no_grad():
        normd.W.copy_(plain.W)
        normd.H.copy_(plain.H)

    X_torch = _make_sparse_torch(small_sparse, device)
    plain.train(False)
    normd.train(False)
    z_plain = plain(X_torch)[0]
    z_normd = normd(X_torch)[0]
    # Outputs should not coincide — the encoder sees different inputs.
    assert not torch.allclose(z_plain, z_normd, atol=1e-4)


def test_autoencoder_encode_eval_is_deterministic_for_vae(small_sparse, device):
    """In inference mode (``model.train(False)``) VAE.encode() returns
    ``mu`` directly — no sampling — so successive calls must be
    bit-identical."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=4,
        latent_dim=2,
        hidden_dims=(8,),
        use_vae=True,
        device=device,
        random_state=0,
    )
    X_torch = _make_sparse_torch(small_sparse, device)
    model.train(False)
    z1 = model.encode(X_torch)
    z2 = model.encode(X_torch)
    assert torch.equal(z1, z2)


# ── compute_joint_loss ──────────────────────────────────────────────


def _prep_joint_loss_inputs(small_sparse, device, *, use_vae=False, latent_dim=2):
    """Build a model + run forward to produce all the tensors that
    ``compute_joint_loss`` consumes. Pulling this out keeps each
    loss-shape test small."""
    from sparse_nmf import SparseNMF_Autoencoder

    n, m = small_sparse.shape
    model = SparseNMF_Autoencoder(
        n_samples=n,
        n_features=m,
        nmf_components=4,
        latent_dim=latent_dim,
        hidden_dims=(8,),
        use_vae=use_vae,
        device=device,
        random_state=0,
    )
    X_torch = _make_sparse_torch(small_sparse, device)
    out = model(X_torch)
    return model, X_torch, out


def test_compute_joint_loss_returns_scalar_with_grad(small_sparse, device):
    """Loss must be a 0-dim tensor with grad_fn so .backward() works.
    Also: every weighted component should appear in the dict."""
    from sparse_nmf import compute_joint_loss

    model, X_torch, (z, W_recon, X_recon, W, H) = _prep_joint_loss_inputs(small_sparse, device)
    loss, parts = compute_joint_loss(
        model=model,
        X_sparse_torch=X_torch,
        z=z,
        W_recon=W_recon,
        X_recon=X_recon,
        W=W,
        H=H,
    )
    assert loss.dim() == 0
    assert loss.requires_grad
    assert loss.grad_fn is not None
    # Default flags include nmf, ae, contrastive, dim_reg (no kl since not VAE).
    assert {"nmf", "ae", "contrastive", "dim_reg"} <= set(parts.keys())
    assert "kl" not in parts


def test_compute_joint_loss_vae_includes_kl(small_sparse, device):
    """When the model is a VAE, the loss dict must include ``kl``
    (KL divergence between latent posterior and unit Gaussian)."""
    from sparse_nmf import compute_joint_loss

    model, X_torch, out = _prep_joint_loss_inputs(small_sparse, device, use_vae=True)
    z, W_recon, X_recon, W, H, mu, logvar = out
    loss, parts = compute_joint_loss(
        model=model,
        X_sparse_torch=X_torch,
        z=z,
        W_recon=W_recon,
        X_recon=X_recon,
        W=W,
        H=H,
        mu=mu,
        logvar=logvar,
    )
    assert "kl" in parts
    assert torch.isfinite(parts["kl"])
    assert torch.isfinite(loss)


def test_compute_joint_loss_skips_contrastive_when_disabled(small_sparse, device):
    """``use_contrastive=False`` must remove the contrastive term
    entirely — both from the loss arithmetic and the parts dict."""
    from sparse_nmf import compute_joint_loss

    model, X_torch, (z, W_recon, X_recon, W, H) = _prep_joint_loss_inputs(small_sparse, device)
    _, parts = compute_joint_loss(
        model=model,
        X_sparse_torch=X_torch,
        z=z,
        W_recon=W_recon,
        X_recon=X_recon,
        W=W,
        H=H,
        use_contrastive=False,
    )
    assert "contrastive" not in parts


def test_compute_joint_loss_dense_path_runs(small_sparse, device):
    """``use_sparse_loss=False`` switches to the dense MSE path
    (materializes X_recon). Verify it works on a small matrix."""
    from sparse_nmf import compute_joint_loss

    model, X_torch, (z, W_recon, _, W, H) = _prep_joint_loss_inputs(small_sparse, device)
    loss, parts = compute_joint_loss(
        model=model,
        X_sparse_torch=X_torch,
        z=z,
        W_recon=W_recon,
        X_recon=None,  # exercises the on-the-fly mm path inside dense branch
        W=W,
        H=H,
        use_sparse_loss=False,
        use_contrastive=False,
        dimension_reg_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert "nmf" in parts


def test_compute_joint_loss_mse_ae_path(small_sparse, device):
    """``use_cosine_loss=False`` switches AE loss from cosine to MSE.
    Different number → confirms the branch was taken."""
    from sparse_nmf import compute_joint_loss

    model, X_torch, (z, W_recon, _, W, H) = _prep_joint_loss_inputs(small_sparse, device)
    _, parts_cos = compute_joint_loss(
        model=model, X_sparse_torch=X_torch, z=z, W_recon=W_recon,
        X_recon=None, W=W, H=H, use_cosine_loss=True, use_contrastive=False,
        dimension_reg_weight=0.0,
    )
    _, parts_mse = compute_joint_loss(
        model=model, X_sparse_torch=X_torch, z=z, W_recon=W_recon,
        X_recon=None, W=W, H=H, use_cosine_loss=False, use_contrastive=False,
        dimension_reg_weight=0.0,
    )
    # Cosine in [0,2], MSE on tiny tensors typically << 1 — they should
    # not coincide unless one branch silently fell through to the other.
    assert not torch.allclose(parts_cos["ae"], parts_mse["ae"])


def test_compute_joint_loss_dim_reg_zero_weight_skips_term(small_sparse, device):
    """``dimension_reg_weight=0`` should skip the dim-reg branch
    entirely — keeping it out of the dict so consumers iterating
    over parts don't have to filter zero terms."""
    from sparse_nmf import compute_joint_loss

    model, X_torch, (z, W_recon, _, W, H) = _prep_joint_loss_inputs(small_sparse, device)
    _, parts = compute_joint_loss(
        model=model, X_sparse_torch=X_torch, z=z, W_recon=W_recon,
        X_recon=None, W=W, H=H, dimension_reg_weight=0.0,
        use_contrastive=False,
    )
    assert "dim_reg" not in parts


# ── train_joint_model deeper coverage ───────────────────────────────


@pytest.mark.slow
def test_train_joint_model_vae_runs_one_epoch(small_sparse, device):
    """VAE-mode end-to-end: 1 epoch must run and return finite z of
    the requested latent dim."""
    from sparse_nmf import train_joint_model

    z, model = train_joint_model(
        small_sparse,
        n_samples=small_sparse.shape[0],
        n_features=small_sparse.shape[1],
        nmf_components=4,
        latent_dim=3,
        device=device,
        n_epochs=1,
        batch_size=64,
        verbose=False,
        use_vae=True,
        random_state=0,
    )
    z = np.asarray(z)
    assert z.shape == (small_sparse.shape[0], 3)
    assert np.isfinite(z).all()


@pytest.mark.slow
def test_train_joint_model_different_seeds_diverge(small_sparse, device):
    """A 2-epoch run on a tiny matrix should at minimum complete and
    return a different W than the random init — confirms gradients
    actually flow back through the model.

    We check by training two models with different seeds and ensuring
    their embeddings differ. If gradients didn't flow, both models
    would just be returning their (different) random inits, so this
    test is actually weakly sensitive to learning. The strong claim
    is just: training completes and produces seed-dependent output."""
    from sparse_nmf import train_joint_model

    args = dict(
        n_samples=small_sparse.shape[0],
        n_features=small_sparse.shape[1],
        nmf_components=4,
        latent_dim=2,
        device=device,
        n_epochs=2,
        batch_size=64,
        verbose=False,
    )
    z_a, _ = train_joint_model(small_sparse, random_state=0, **args)
    z_b, _ = train_joint_model(small_sparse, random_state=99, **args)
    z_a = np.asarray(z_a)
    z_b = np.asarray(z_b)
    assert z_a.shape == z_b.shape
    assert not np.allclose(z_a, z_b, atol=1e-3)


# ``import torch`` at module scope so the helper functions and tests
# above can use it without each repeating the import.
import torch  # noqa: E402  (intentional — used by helpers above)
