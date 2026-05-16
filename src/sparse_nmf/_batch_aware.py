"""Batch-aware sparse NMF — iNMF-inspired factorization.

Standard NMF (and sparseNMF) doesn't model batch effects: a method
trained on multi-protocol scRNA-seq will load batch-specific biology
into the shared factor space, polluting the cell embeddings. This
module adds an explicit *per-batch* additive correction to the
factor loadings, inspired by LIGER's integrative NMF (Welch et al.
2019, Cell).

The factorization splits the loadings matrix into a **shared** part
and a **batch-specific** part:

    X_c ≈ W_c · (H_shared + V[batch(c)])

where:
- ``W`` (n_cells × k) — cell embeddings; downstream consumer uses
  this matrix.
- ``H_shared`` (k × n_genes) — batch-invariant factor loadings.
- ``V[b]`` (k × n_genes) — additive per-batch correction. Bounded
  in magnitude by an L2 penalty (``alignment_weight``) so that
  most signal is forced into the shared part.

Objective::

    L = Σ_c ‖X_c − W_c (H_shared + V[batch(c)])‖²
        + α_v · Σ_b ‖V[b]‖_F²
        + α_w · ‖W‖_1

Subject to W, H_shared, V[b] ≥ 0.

The L2 penalty α_v controls the trade-off:
- α_v → ∞: V[b] → 0, recovers standard NMF (no batch correction).
- α_v → 0: V[b] absorbs all signal; W loses meaning.
- Moderate α_v: ~LIGER recommendation of 5–10× the per-batch
  reconstruction loss magnitude.

Multiplicative updates (Lee-Seung style, batched in the cell axis)::

    H_shared ← H_shared ⊙ (Wᵀ X) / Σ_b (W_bᵀ W_b (H_shared + V_b))
    V[b]     ← V[b]     ⊙ (W_bᵀ X_b) / (W_bᵀ W_b (H_shared + V_b) + α_v V_b)
    W[c]     ← W[c]     ⊙ (X_c H_effᵀ) / (W_c H_eff H_effᵀ + α_w)

where ``H_eff = H_shared + V[batch(c)]`` is the cell-specific
effective loading matrix.

Note we keep the entrywise additive form of iNMF (not the
multiplicative ``W · V_b · H`` form some variants use). The
additive form preserves the MU positivity guarantee (Lee-Seung
2001) directly: numerator and denominator stay non-negative for
non-negative initialization, so the updates can never produce
negative entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.sparse import issparse, spmatrix


def _to_torch_dense(X, device: str, dtype=torch.float32) -> torch.Tensor:
    """Convert a (sparse or dense) numpy/scipy matrix to a dense torch
    tensor on ``device``. Used for the small batched slices that fit
    in GPU memory comfortably."""
    if issparse(X):
        X = X.toarray()
    return torch.as_tensor(np.asarray(X), dtype=dtype, device=device)


def _maybe_log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


@dataclass
class BatchAwareResult:
    """Returned by ``train_sparse_nmf_batch_aware``.

    Attributes
    ----------
    W
        Cell embeddings, shape (n_cells, k). The matrix downstream
        consumers should use.
    H_shared
        Shared factor loadings, shape (k, n_genes). Interpretable as
        the batch-invariant gene programs.
    V
        Per-batch additive corrections. ``V[b]`` has shape
        (k, n_genes); ``V`` itself is a dict keyed by batch label.
        Useful for inspecting which factors carry batch signal.
    losses
        Reconstruction loss per logged iteration. Empty when
        ``verbose=False``.
    n_iter
        Number of MU iterations actually run (≤ ``max_iter``).
    """

    W: np.ndarray
    H_shared: np.ndarray
    V: dict
    losses: list[float]
    n_iter: int


def train_sparse_nmf_batch_aware(
    X_sparse: spmatrix,
    batch: np.ndarray,
    n_components: int,
    *,
    max_iter: int = 500,
    sparsity_weight: float = 0.01,
    alignment_weight: float = 1.0,
    device: str = "cuda",
    random_state: Optional[int] = None,
    patience: Optional[int] = 10,
    tol: float = 1e-5,
    normalize_inputs: bool = True,
    verbose: bool = True,
) -> BatchAwareResult:
    """Train a batch-aware sparse NMF.

    Parameters
    ----------
    X_sparse
        (n_cells, n_genes) scipy.sparse matrix.
    batch
        (n_cells,) array of batch labels. Any hashable values; will
        be canonicalized internally.
    n_components
        Factor count (``k``).
    sparsity_weight
        L1 penalty on W. Larger → more zeros in cell embeddings.
        Default 0.01, matching the rough scale ``train_sparse_nmf``
        uses implicitly via its MU update.
    alignment_weight
        L2 penalty on the per-batch corrections V[b]. Larger →
        more aggressive batch alignment (V[b] shrinks toward 0,
        recovering standard NMF in the limit). The "default 1.0"
        is a moderate setting; the right value scales with the
        per-batch reconstruction loss magnitude — empirically 0.5–10
        is the useful range.
    device
        ``"cuda"`` or ``"cpu"``. Auto-detects CUDA availability.
    random_state
        Seed for W / H_shared initialization.
    patience
        Number of error-report intervals to wait without improvement
        before early-stopping. ``None`` disables (runs to ``max_iter``
        or tolerance).
    tol
        Relative-improvement tolerance for early convergence.
    normalize_inputs
        L2-row-normalize X before factorization (depth normalization,
        matching ``train_sparse_nmf``'s default). Strongly recommended
        for single-cell counts.
    verbose
        Print progress + loss every 10 iterations.

    Returns
    -------
    BatchAwareResult
        See class docstring.
    """
    if not issparse(X_sparse):
        raise TypeError("X_sparse must be scipy.sparse")
    n_cells, n_genes = X_sparse.shape
    if batch.shape != (n_cells,):
        raise ValueError(
            f"batch shape {batch.shape} doesn't match n_cells={n_cells}"
        )
    k = int(n_components)

    # Canonicalize batch labels → integer 0..(n_batches-1).
    uniq_batches, batch_idx = np.unique(batch, return_inverse=True)
    n_batches = len(uniq_batches)
    _maybe_log(verbose,
        f"  batch-aware sparseNMF: n_cells={n_cells} n_genes={n_genes} "
        f"k={k} n_batches={n_batches} α_w={sparsity_weight} α_v={alignment_weight}"
    )

    if not torch.cuda.is_available() and device.startswith("cuda"):
        _maybe_log(verbose, "  cuda not available — falling back to cpu")
        device = "cpu"

    # Convert X to CSR; precompute per-batch row slices (sorted by batch).
    X_csr = X_sparse.tocsr().astype(np.float32)

    if normalize_inputs:
        # L2-row-normalize directly on the CSR (preserves sparsity).
        # sparse_nmf.utils.l2_normalize is dense-only; using sklearn's
        # `normalize` which has a fast sparse path.
        from sklearn.preprocessing import normalize

        X_csr = normalize(X_csr, norm="l2", axis=1).astype(np.float32)

    # Group cell indices by batch for fast per-batch slicing during MU.
    cells_in_batch: list[np.ndarray] = [
        np.where(batch_idx == b)[0] for b in range(n_batches)
    ]

    # Initialize W, H_shared, V[b] with non-negative random values.
    rng = np.random.default_rng(random_state)
    scale = float(np.sqrt(X_csr.mean() / k + 1e-8))
    W = torch.tensor(rng.random((n_cells, k)).astype(np.float32) * scale, device=device)
    H_shared = torch.tensor(rng.random((k, n_genes)).astype(np.float32) * scale, device=device)
    V = [torch.tensor(rng.random((k, n_genes)).astype(np.float32) * scale * 0.1, device=device)
         for _ in range(n_batches)]

    # Precompute torch tensors for each batch's X_b (kept on device).
    # For very large datasets, this could blow VRAM — but for our
    # target scale (≤100k cells × 2k HVGs ≤ ~800 MB float32) it's fine.
    X_b_torch: list[torch.Tensor] = []
    for b in range(n_batches):
        idx = cells_in_batch[b]
        if idx.size == 0:
            X_b_torch.append(torch.zeros((0, n_genes), dtype=torch.float32, device=device))
        else:
            X_b_torch.append(_to_torch_dense(X_csr[idx], device))

    eps = 1e-10
    losses: list[float] = []
    best_loss = float("inf")
    patience_counter = 0

    for it in range(max_iter):
        # H_shared update.
        # Numerator: Σ_c W_c.T X_c = (Σ_b W_b.T X_b).
        # Denominator: Σ_b W_b.T W_b (H_shared + V[b]).
        H_num = torch.zeros((k, n_genes), device=device)
        H_den = torch.zeros((k, n_genes), device=device)
        for b in range(n_batches):
            idx = cells_in_batch[b]
            if idx.size == 0:
                continue
            W_b = W[idx]
            WbTWb = W_b.t() @ W_b
            H_num += W_b.t() @ X_b_torch[b]
            H_den += WbTWb @ (H_shared + V[b])
        H_shared = H_shared * (H_num / (H_den + eps))
        H_shared = torch.clamp(H_shared, min=eps)

        # V[b] updates — same numerator pattern, but with the L2 penalty
        # term ``alignment_weight * V[b]`` in the denominator.
        for b in range(n_batches):
            idx = cells_in_batch[b]
            if idx.size == 0:
                continue
            W_b = W[idx]
            WbTWb = W_b.t() @ W_b
            V_num = W_b.t() @ X_b_torch[b]
            V_den = WbTWb @ (H_shared + V[b]) + alignment_weight * V[b]
            V[b] = V[b] * (V_num / (V_den + eps))
            V[b] = torch.clamp(V[b], min=eps)

        # W update — batch-sliced so each cell's H_eff is its batch's.
        # Sparsity penalty α_w lands in the denominator (L1-on-W is the
        # gradient sign-preserving subgradient at W>0, so the MU
        # denominator picks up a constant α_w term).
        for b in range(n_batches):
            idx = cells_in_batch[b]
            if idx.size == 0:
                continue
            H_eff = H_shared + V[b]
            W_b = W[idx]
            num = X_b_torch[b] @ H_eff.t()
            den = (W_b @ H_eff) @ H_eff.t() + sparsity_weight
            W[idx] = W_b * (num / (den + eps))
            # No clamp on W — sparsity from α_w pushes entries toward 0.

        # Loss + early stopping every 10 iters.
        if (it + 1) % 10 == 0 or it == 0:
            loss = 0.0
            for b in range(n_batches):
                idx = cells_in_batch[b]
                if idx.size == 0:
                    continue
                recon = W[idx] @ (H_shared + V[b])
                loss += float(((X_b_torch[b] - recon) ** 2).sum().item())
            v_pen = float(sum((alignment_weight * (Vb ** 2).sum()).item() for Vb in V))
            w_pen = float((sparsity_weight * W.abs().sum()).item())
            total = loss + v_pen + w_pen
            losses.append(total)
            _maybe_log(verbose,
                f"  iter {it+1:4d} loss={total:.4f} (recon={loss:.4f} V_l2={v_pen:.4f} W_l1={w_pen:.4f})"
            )

            if total < best_loss * (1 - tol):
                best_loss = total
                patience_counter = 0
            else:
                patience_counter += 1
            if patience is not None and patience_counter >= patience:
                _maybe_log(verbose,
                    f"  early stop at iter {it+1}: no rel-improvement > {tol} for {patience} checks"
                )
                break

    # Materialize numpy results.
    W_np = W.detach().cpu().numpy()
    H_np = H_shared.detach().cpu().numpy()
    V_dict = {
        str(uniq_batches[b]): V[b].detach().cpu().numpy()
        for b in range(n_batches)
    }
    return BatchAwareResult(
        W=W_np, H_shared=H_np, V=V_dict,
        losses=losses, n_iter=it + 1,
    )
