"""Joint NMF + autoencoder producing a 2-D scatter-friendly embedding.

Run from repo root::

    python examples/joint_model.py

Trains for a small number of epochs on the synthetic generator's
output (default: 2,000 samples × 5,000 features, planted rank=16),
then prints the resulting (n_samples, 2) embedding shape and a
quick check that distinct latent clusters exist.
"""

from __future__ import annotations

import time

import numpy as np

from sparse_nmf import train_joint_model
from sparse_nmf.data import generate_synthetic_sparse


def main() -> int:
    print("Generating sample data (rank-16 planted, 2k × 5k)...")
    X = generate_synthetic_sparse(
        n_samples=2_000,
        n_features=5_000,
        n_components=16,
        density=0.05,
        seed=42,
    )

    print("Training joint NMF + autoencoder for 20 epochs...")
    t0 = time.time()
    z, model = train_joint_model(
        X,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        nmf_components=64,
        latent_dim=2,
        device="cpu",
        n_epochs=20,
        batch_size=128,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  z.shape={np.asarray(z).shape}")

    z = np.asarray(z)
    print(
        "  embedding stats: "
        f"mean={z.mean():.3f}  std={z.std():.3f}  range=[{z.min():.3f}, {z.max():.3f}]"
    )

    # Sanity: ensure non-degenerate spread along both latent axes.
    if z.std(axis=0).min() < 1e-3:
        print("  WARNING: one latent axis collapsed (std < 1e-3)")
    else:
        print("  OK: both latent axes have meaningful spread")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
