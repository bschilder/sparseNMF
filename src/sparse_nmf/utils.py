"""Shared helpers for ``sparse_nmf``.

These utilities are general enough that more than one module in this
package (and downstream consumers) will need them, so they live in a
dedicated submodule rather than the package ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import torch

    ArrayLike = Union[np.ndarray, "torch.Tensor"]


def l2_normalize(embeddings):
    """L2-normalize embedding rows to unit length.

    Each row is divided by its L2 norm; zero-norm rows pass through
    unchanged so callers never get a div-by-zero NaN. Accepts numpy
    arrays or torch tensors and returns the same type — for tensors,
    dtype and device are preserved.

    Parameters
    ----------
    embeddings : np.ndarray or torch.Tensor
        Shape ``(n_samples, embedding_dim)``. Each row is a single
        embedding vector.

    Returns
    -------
    np.ndarray or torch.Tensor
        Same shape and type as the input, with rows scaled to unit
        L2 norm. Zero-norm rows are returned as-is.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(embeddings, torch.Tensor):
        return F.normalize(embeddings, p=2, dim=1)

    arr = np.asarray(embeddings, dtype=np.float32)
    # ``torch.from_numpy`` requires a writable buffer; the
    # ``.copy()`` guards against arrays that arrived as views of
    # read-only memory (e.g. mmap'd .npy files).
    if not arr.flags.writeable:
        arr = arr.copy()
    return F.normalize(torch.from_numpy(arr), p=2, dim=1).numpy()


__all__ = ["l2_normalize"]
