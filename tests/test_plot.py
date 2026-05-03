"""Tests for ``plot_nmf_factor_distributions``.

The plot function is a 230-line wrapper around matplotlib that
visualizes per-factor distributions in a faceted histogram grid.
We test by calling with various flag combinations and checking the
returned Figure has the expected structure (subplot count, axis
properties) — no need to render pixels.

matplotlib's ``Agg`` backend is set at import time so these tests
work headlessly on CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must come before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest

from sparse_nmf import plot_nmf_factor_distributions


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any open figures after each test so we don't leak
    matplotlib state across the suite."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def factor_matrix():
    """A small (n=200, k=6) factor matrix with a mix of distributions
    — some skewed, some near-zero — so the plot exercises both KDE
    and histogram paths and the per-subplot stats text doesn't crash
    on flat input."""
    rng = np.random.RandomState(0)
    return np.abs(rng.randn(200, 6)).astype(np.float32) + 0.01


def test_plot_returns_figure_when_requested(factor_matrix):
    """``return_fig=True`` must return a Figure with one subplot per
    factor (in a 2-row × max_cols layout when n_factors > max_cols)."""
    fig = plot_nmf_factor_distributions(factor_matrix, return_fig=True)
    assert fig is not None
    # 6 factors → 2 rows × min(4, 6)=4 cols, but with 6 factors that
    # leaves 2 unused subplots. Ensure ≥ 6 axes total (the function
    # may add an extra title axis or a colorbar — only assert the
    # lower bound).
    assert len(fig.axes) >= 6


def test_plot_returns_none_when_return_fig_false(factor_matrix):
    """Default ``return_fig=False`` should return None — the function
    is expected to call plt.show() and let the caller move on."""
    out = plot_nmf_factor_distributions(factor_matrix, return_fig=False)
    assert out is None


def test_plot_with_n_factors_to_plot_subset(factor_matrix):
    """Limiting to a subset of factors should produce fewer subplots."""
    fig = plot_nmf_factor_distributions(
        factor_matrix, n_factors_to_plot=3, return_fig=True
    )
    # At least 3 axes for the 3 factors — possibly more if the
    # function pads to a full grid row.
    assert len(fig.axes) >= 3


def test_plot_kde_disabled(factor_matrix):
    """``kde=False`` skips the seaborn KDE overlay — different code
    path than the default. Plot should still complete and return a
    valid Figure."""
    fig = plot_nmf_factor_distributions(factor_matrix, kde=False, return_fig=True)
    assert fig is not None


def test_plot_log_scales(factor_matrix):
    """Both ``log_x`` and ``log_y`` should switch the axis scale.
    We just verify the call completes — exact tick-locator inspection
    is brittle across matplotlib versions."""
    fig = plot_nmf_factor_distributions(
        factor_matrix, log_x=True, log_y=True, return_fig=True
    )
    assert fig is not None


def test_plot_filter_zeros(factor_matrix):
    """``filter_zeros=True`` excludes near-zero values from the plot
    distribution (but stats are still computed on the full data)."""
    # Inject some explicit zeros so the filter has work to do.
    W = factor_matrix.copy()
    W[:50, 0] = 0.0
    fig = plot_nmf_factor_distributions(
        W, filter_zeros=True, zero_threshold=1e-6, return_fig=True
    )
    assert fig is not None


def test_plot_with_factor_names(factor_matrix):
    """Custom ``factor_names`` should appear as subplot titles. We
    verify by inspecting axis titles after plotting."""
    names = [f"comp_{i}" for i in range(factor_matrix.shape[1])]
    fig = plot_nmf_factor_distributions(
        factor_matrix, factor_names=names, return_fig=True
    )
    titles = [ax.get_title() for ax in fig.axes]
    # At least one of the configured names should appear in the
    # subplot titles (the function may decorate with extra info,
    # but the name itself should be present).
    assert any("comp_" in t for t in titles), titles


def test_plot_overall_title(factor_matrix):
    """Custom overall ``title`` should land as the figure suptitle."""
    fig = plot_nmf_factor_distributions(
        factor_matrix, title="Custom Suptitle 123", return_fig=True
    )
    # suptitle is a Text object with a non-empty string
    suptitle = fig._suptitle
    assert suptitle is not None
    assert "Custom Suptitle 123" in suptitle.get_text()


def test_plot_accepts_torch_tensor(factor_matrix):
    """W can be a torch.Tensor; the function must convert internally
    via ``.detach().cpu().numpy()`` rather than crashing on
    dtype mismatch."""
    import torch

    W_torch = torch.from_numpy(factor_matrix)
    fig = plot_nmf_factor_distributions(W_torch, return_fig=True)
    assert fig is not None


def test_plot_subsamples_for_large_n(factor_matrix):
    """For n_samples > max_samples the function subsamples for the
    histogram (full-data stats are kept). We pass max_samples=50 and
    a 200-row matrix to force the subsample branch."""
    fig = plot_nmf_factor_distributions(
        factor_matrix, max_samples=50, return_fig=True
    )
    assert fig is not None


def test_plot_custom_figsize_and_max_cols(factor_matrix):
    """User-supplied ``figsize`` should be honored exactly, and
    ``max_cols`` should change the grid layout."""
    fig = plot_nmf_factor_distributions(
        factor_matrix, figsize=(10, 6), max_cols=2, return_fig=True
    )
    assert fig is not None
    np.testing.assert_allclose(fig.get_size_inches(), (10, 6))


def test_plot_shared_axes(factor_matrix):
    """``sharey=True`` is a non-default branch — exercises the
    plt.subplots(sharey=True) path. Plot must still complete."""
    fig = plot_nmf_factor_distributions(
        factor_matrix, sharex=True, sharey=True, return_fig=True
    )
    assert fig is not None


def test_plot_handles_more_factors_than_default_cols(factor_matrix):
    """The default grid is max_cols=4. A 9-component matrix needs
    a second row — exercises the multi-row layout branch."""
    rng = np.random.RandomState(1)
    W_wide = np.abs(rng.randn(150, 9)).astype(np.float32) + 0.01
    fig = plot_nmf_factor_distributions(W_wide, return_fig=True)
    assert fig is not None
    assert len(fig.axes) >= 9


def test_plot_handles_all_zero_factor():
    """When a factor column is entirely below ``zero_threshold`` and
    ``filter_zeros=True``, the function shows an "All zeros" text
    annotation in that subplot instead of trying to plot an empty
    histogram (lines 3727-3733)."""
    n, k = 100, 4
    W = np.abs(np.random.RandomState(2).randn(n, k).astype(np.float32)) + 0.1
    # Force one factor to be all-zero.
    W[:, 0] = 0.0
    fig = plot_nmf_factor_distributions(
        W, filter_zeros=True, zero_threshold=1e-3, return_fig=True
    )
    assert fig is not None
