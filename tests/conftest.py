"""Pytest fixtures shared across the suite.

The synthetic sparse matrix used by the unit tests is small (500 × 1k
@ 5% density) so the whole suite finishes in a few seconds even on a
laptop CPU. Tests that legitimately need a GPU are marked with
``@pytest.mark.gpu`` and skipped when one isn't available.
"""

from __future__ import annotations

import pytest

from sparse_nmf.data import generate_synthetic_sparse


@pytest.fixture(scope="session")
def small_sparse():
    """Tiny rank-8 sparse matrix; same content every session for determinism."""
    return generate_synthetic_sparse(
        n_samples=200,
        n_features=400,
        n_components=8,
        density=0.05,
        seed=7,
    )


@pytest.fixture(scope="session")
def device():
    """Pick CUDA when present, else CPU. Lets the same test exercise both
    when run on different machines without a code change."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.gpu`` tests when no CUDA device is visible."""
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device — skipping GPU-only test")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
