"""Build + execute the tutorial notebooks under ``docs/notebooks/``.

Run from repo root::

    python scripts/build_notebooks.py

Each notebook is constructed programmatically (so the source-of-truth
lives as Python code, easy to diff and review) and then executed
inline via ``nbclient`` so the rendered ``.ipynb`` ships with real
plots / prints baked in. ``docs/conf.py`` sets ``nb_execution_mode =
"off"`` so RTD doesn't re-execute on every doc build.

The script is idempotent — re-running overwrites the .ipynb files
in-place. Outputs are deterministic given the same package version
(seeded RNG everywhere) so re-running shouldn't churn the diff.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

OUT = Path(__file__).resolve().parents[1] / "docs" / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return nbformat.v4.new_markdown_cell(text)


def code(src: str) -> dict:
    return nbformat.v4.new_code_cell(src)


def write_nb(path: Path, cells: list[dict], title: str) -> None:
    nb = nbformat.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "title": title,
    }
    nb["cells"] = cells
    print(f"executing {path.name}...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)
    print(f"  wrote {path}")


# ── 01 — basic NMF ──────────────────────────────────────────────────
write_nb(
    OUT / "01_basic_nmf.ipynb",
    [
        md(
            "# Tutorial 1 — Standalone sparse NMF\n\n"
            "This notebook walks through the simplest use of\n"
            "`sparse_nmf.SparseNMF` on the bundled synthetic dataset.\n"
            "We fit a rank-8 model, then check the recovered factors\n"
            "and reconstruction quality.\n\n"
            "Run order: top to bottom. All cells should execute in a\n"
            "few seconds on CPU."
        ),
        code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from sparse_nmf import SparseNMF\n"
            "from sparse_nmf.data import generate_synthetic_sparse\n"
            "\n"
            "np.random.seed(0)\n"
            "X = generate_synthetic_sparse(\n"
            "    n_samples=500, n_features=1_000, n_components=8,\n"
            "    density=0.05, seed=0,\n"
            ")\n"
            "print(f'shape={X.shape}  nnz={X.nnz:,}  density={X.nnz / (X.shape[0]*X.shape[1]):.3%}')"
        ),
        md(
            "## Fit the model\n\n"
            "200 multiplicative-update iterations is enough for this\n"
            "small matrix. On real data you'll typically want 500 +\n"
            "and a CUDA device."
        ),
        code(
            "nmf = SparseNMF(n_components=8, max_iter=200, device='cpu', verbose=False)\n"
            "W = nmf.fit_transform(X)\n"
            "H = nmf.H.detach().cpu().numpy()\n"
            "print(f'W.shape={W.shape}  H.shape={H.shape}')\n"
            "print(f'W non-negative: {(W >= 0).all()}')\n"
            "print(f'H non-negative: {(H >= 0).all()}')"
        ),
        md(
            "## Reconstruction quality\n\n"
            "On a rank-8 planted matrix with light noise, NMF should\n"
            "land within ~30-40% relative Frobenius error."
        ),
        code(
            "X_dense = X.toarray()\n"
            "recon = W @ H\n"
            "rel_err = np.linalg.norm(X_dense - recon) / np.linalg.norm(X_dense)\n"
            "print(f'relative reconstruction error: {rel_err:.4f}')"
        ),
        md(
            "## Visualize the per-sample codes\n\n"
            "The columns of `W` are the latent factor activations.\n"
            "Heatmap of the first 50 samples × 8 factors:"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "im = ax.imshow(W[:50], aspect='auto', cmap='viridis')\n"
            "ax.set_xlabel('factor')\n"
            "ax.set_ylabel('sample')\n"
            "ax.set_title('W (per-sample factor activations, first 50 samples)')\n"
            "plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Visualize the per-feature loadings\n\n"
            "The rows of `H` show which features each factor activates.\n"
            "We pick the top 30 features per factor for readability:"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "top_features = np.argsort(H.sum(axis=0))[::-1][:30]\n"
            "im = ax.imshow(H[:, top_features], aspect='auto', cmap='viridis')\n"
            "ax.set_xlabel('top-30 feature')\n"
            "ax.set_ylabel('factor')\n"
            "ax.set_title('H (per-feature loadings, top-30 features)')\n"
            "plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Next\n\n"
            "Tutorial 2 (`02_joint_model.ipynb`) wraps this same NMF\n"
            "factorization inside an autoencoder so you get a\n"
            "low-dimensional embedding directly, without a separate\n"
            "post-NMF dimensionality-reduction step."
        ),
    ],
    title="Tutorial 1 — Standalone sparse NMF",
)


# ── 02 — joint model ────────────────────────────────────────────────
write_nb(
    OUT / "02_joint_model.ipynb",
    [
        md(
            "# Tutorial 2 — Joint NMF + autoencoder\n\n"
            "This notebook trains the joint model end-to-end and uses\n"
            "the resulting 2-D embedding for visualization. Useful when\n"
            "the downstream task is plotting / clustering rather than\n"
            "interpreting the W/H factors directly.\n\n"
            "We keep the run short (10 epochs) so the notebook executes\n"
            "in under a minute on CPU. Production runs typically use\n"
            "100+ epochs and a CUDA device."
        ),
        code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "np.random.seed(0)\n"
            "\n"
            "from sparse_nmf import train_joint_model\n"
            "from sparse_nmf.data import generate_synthetic_sparse\n"
            "\n"
            "X = generate_synthetic_sparse(\n"
            "    n_samples=600, n_features=800, n_components=8,\n"
            "    density=0.05, seed=0,\n"
            ")\n"
            "print(f'shape={X.shape}  nnz={X.nnz:,}')"
        ),
        md(
            "## Train\n\n"
            "`nmf_components` is the dimensionality of the NMF stage's\n"
            "output (the autoencoder's input). `latent_dim` is the\n"
            "bottleneck size — pick 2 for visualization, 64-256 for\n"
            "downstream retrieval."
        ),
        code(
            "z, model = train_joint_model(\n"
            "    X,\n"
            "    n_samples=X.shape[0],\n"
            "    n_features=X.shape[1],\n"
            "    nmf_components=32,\n"
            "    latent_dim=2,\n"
            "    device='cpu',\n"
            "    n_epochs=10,\n"
            "    batch_size=128,\n"
            "    verbose=False,\n"
            ")\n"
            "z = np.asarray(z)\n"
            "print(f'embedding shape: {z.shape}')\n"
            "print(f'mean={z.mean():.3f}  std={z.std():.3f}')"
        ),
        md(
            "## Plot the 2-D embedding\n\n"
            "Color samples by the dominant factor in the synthetic\n"
            "data — i.e. which of the 8 planted clusters they came\n"
            "from. We compute the assignment from the original W (per-\n"
            "sample mixture weights) so the colors reflect biology, not\n"
            "the model's prediction."
        ),
        code(
            "# Recover the planted cluster assignment for coloring.\n"
            "rng = np.random.default_rng(0)\n"
            "W_planted = rng.gamma(2.0, 1.0, (X.shape[0], 8)).astype(np.float32)\n"
            "labels = W_planted.argmax(axis=1)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 5))\n"
            "scatter = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)\n"
            "ax.set_xlabel('latent_0')\n"
            "ax.set_ylabel('latent_1')\n"
            "ax.set_title('Joint NMF + autoencoder — 2-D embedding')\n"
            "plt.colorbar(scatter, ax=ax, label='planted cluster')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "Even after only 10 epochs on CPU, the embedding shows\n"
            "structure that lines up with the planted clusters. With\n"
            "more epochs and real data, this becomes the input to\n"
            "downstream tasks — clustering, retrieval, classification."
        ),
    ],
    title="Tutorial 2 — Joint NMF + autoencoder",
)

print("\nAll tutorials built and executed.")
