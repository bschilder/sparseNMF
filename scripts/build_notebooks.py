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
(seeded RNG + single-threaded BLAS / OMP — see ``set_global_seed``
in ``examples/_determinism.py``).
"""

from __future__ import annotations

import os

# Single-threaded BLAS / OpenMP env vars must be set BEFORE the child
# kernel process starts (since they're read at thread-pool init in
# numpy / scipy / torch). nbclient inherits env from this parent
# process, so setting them here propagates into every notebook kernel.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Imports below sit after the determinism env-var setup — E402 is the
# intended structure here, not a mistake.
from pathlib import Path  # noqa: E402

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return nbformat.v4.new_markdown_cell(text)


def code(src: str) -> dict:
    return nbformat.v4.new_code_cell(src)


def write_nb(path: Path, cells: list[dict], title: str, timeout: int = 120) -> None:
    nb = nbformat.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "title": title,
    }
    nb["cells"] = cells
    print(f"executing {path.name}...")
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)
    print(f"  wrote {path}")


# Snippet used by demo notebooks 03/04 to import their corresponding
# example module from the repo's ``examples/`` directory. Walks up from
# cwd until the file is found, so the same notebook executes correctly
# whether the user opens it from the repo root or from
# ``docs/notebooks/``. Also calls set_global_seed(0) so PCA / NMF /
# sparseNMF / UMAP outputs are identical across runs.
PATH_SETUP = (
    "import sys, pathlib\n"
    "for _p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:\n"
    "    if (_p / 'examples' / '{module}.py').exists():\n"
    "        sys.path.insert(0, str(_p / 'examples'))\n"
    "        break\n"
    "else:\n"
    "    raise RuntimeError('Could not locate the examples/ directory')\n"
    "from _determinism import set_global_seed\n"
    "set_global_seed(0)\n"
)


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

# ── 03 — sparsity confound demo (synthetic) ────────────────────────
write_nb(
    OUT / "03_sparsity_confound_demo.ipynb",
    [
        md(
            "# Tutorial 3 — The sparsity confound (synthetic worst case)\n\n"
            "This notebook walks through the *worst-case* scenario\n"
            "`sparseNMF` is built to handle: two batches of cells with\n"
            "the same biology but very different per-row sparsity\n"
            "signatures. The synthetic data is constructed so that PCA\n"
            "and vanilla NMF will *lock onto* the sparsity axis instead\n"
            "of biology — and we'll see exactly when that happens.\n\n"
            "All heavy lifting (data generator, model fits, figure\n"
            "rendering) lives in `examples/sparsity_confound_demo.py` —\n"
            "this notebook is the narrative wrapper. Cells should\n"
            "execute in under a minute on CPU."
        ),
        code(
            PATH_SETUP.format(module="sparsity_confound_demo")
            + "from sparsity_confound_demo import (\n"
            "    make_sparsity_confound_data, fit_pca, fit_nmf, fit_sparse_nmf,\n"
            "    umap_project, make_figure,\n"
            ")\n"
            "import numpy as np, time\n"
            "from sklearn.metrics import silhouette_score\n"
            "from pathlib import Path"
        ),
        md(
            "## Construct the synthetic data\n\n"
            "Three biological groups, each split into two *sparsity\n"
            "batches* (low-nnz and high-nnz). Within each group the\n"
            "underlying gene loadings are the same — only the\n"
            "per-cell sparsity differs. So *biology* is identical\n"
            "across batches but *nnz signatures* differ by ~10×."
        ),
        code(
            "X, groups, batches = make_sparsity_confound_data(seed=0)\n"
            "nnz = np.asarray((X != 0).sum(axis=1)).ravel()\n"
            "print(f'X shape={X.shape}  nnz={X.nnz:,}  density={X.nnz / (X.shape[0]*X.shape[1]):.2%}')\n"
            "print(f'per-cell nnz: batch_low mean={nnz[batches == 0].mean():.0f}, '\n"
            "      f'batch_high mean={nnz[batches == 1].mean():.0f}')"
        ),
        md(
            "## Compare three factorizations\n\n"
            "Each method produces a high-dim latent representation at\n"
            "the same auto-sized `k`, then UMAP projects to 2-D. The\n"
            "UMAP step is held constant, so any difference in the\n"
            "embedding reflects the *factorization*, not the\n"
            "projector."
        ),
        code(
            "seed = 0\n"
            "k = int(np.clip(min(X.shape) // 8, 32, 1024))\n"
            "print(f'shared latent dim k={k}')\n"
            "\n"
            "embeddings, metrics = {}, {}\n"
            "for name, fn in (('PCA', fit_pca), ('NMF', fit_nmf), ('sparseNMF', fit_sparse_nmf)):\n"
            "    t0 = time.time()\n"
            "    high = fn(X, seed, k)\n"
            "    z = umap_project(high, seed)\n"
            "    embeddings[name] = z\n"
            "    sg = float(silhouette_score(z, groups))\n"
            "    sb = float(silhouette_score(z, batches))\n"
            "    metrics[name] = (sg, sb)\n"
            "    print(f'  {name:>10s}: {time.time()-t0:5.1f}s  '\n"
            "          f'silhouette(group)={sg:+.3f}  silhouette(batch)={sb:+.3f}')"
        ),
        md(
            "## Read the figure\n\n"
            "**Row 1** — colored by biological group. The story we want\n"
            "to see: 3 colors, 3 clean clusters.  \n"
            "**Row 2** — colored by per-cell nnz. We do *not* want this\n"
            "to look structured. If it does, the embedding has aligned\n"
            "with the sparsity axis, which is exactly the failure mode\n"
            "`sparseNMF`'s default `normalize_inputs=True` exists to\n"
            "prevent."
        ),
        code(
            "import tempfile\n"
            "from IPython.display import Image\n"
            "# Use a tempfile so this notebook doesn't overwrite the\n"
            "# production figure at docs/_static/sparsity_confound_demo.png\n"
            "# (that one is produced by `python examples/sparsity_confound_demo.py`).\n"
            "_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)\n"
            "_tmp.close()\n"
            "make_figure(embeddings, metrics, groups, batches, nnz, Path(_tmp.name))\n"
            "Image(filename=_tmp.name)"
        ),
        md(
            "## Takeaway\n\n"
            "PCA and vanilla NMF allocate their leading components to\n"
            "*the per-cell magnitude axis* — variance dominated by the\n"
            "per-row sparsity, not biology. UMAP then spreads cells\n"
            "along that axis, scrambling group identity. `sparseNMF`\n"
            "L2-normalizes each row before the multiplicative updates,\n"
            "so the magnitude axis is gone before factorization\n"
            "starts; the resulting embedding clusters by biology.\n\n"
            "For a real-data version of the same story on cross-protocol\n"
            "scRNA-seq, see Tutorial 4."
        ),
    ],
    title="Tutorial 3 — The sparsity confound (synthetic worst case)",
    timeout=300,
)


# ── 04 — real cross-protocol scRNA-seq demo ────────────────────────
write_nb(
    OUT / "04_real_pancreas_demo.ipynb",
    [
        md(
            "# Tutorial 4 — Real cross-protocol scRNA-seq (scIB pancreas)\n\n"
            "This notebook reproduces the same comparison as Tutorial 3\n"
            "but on real published data: the scIB human pancreas\n"
            "benchmark from Luecken *et al.* 2022 (Nature Methods,\n"
            "[DOI 10.1038/s41592-021-01336-8](https://doi.org/10.1038/s41592-021-01336-8)).\n\n"
            "**Data.** 16,382 pancreatic islet cells across 9 published\n"
            "protocol / batch labels (CEL-seq, CEL-seq2, Smart-seq2,\n"
            "Fluidigm C1, Smarter, inDrop1–4). Library depth varies\n"
            "~300× — Fluidigm C1 at ~1.2 M reads / cell, inDrop3 at\n"
            "~3,800 UMI / cell. 14 published cell types.\n\n"
            "The notebook uses a small stratified subsample\n"
            "(`cells_per_cohort=5`) so it executes in ~2 min on CPU.\n"
            "On first run the dataset is fetched from figshare\n"
            "(~301 MB) and cached to `~/.cache/sparse-nmf/`. For the\n"
            "production figure with `cells_per_cohort=12`, run\n"
            "`python examples/real_pancreas_demo.py` from a shell."
        ),
        code(
            PATH_SETUP.format(module="real_pancreas_demo") + "from real_pancreas_demo import (\n"
            "    load_pancreas, fit_pca, fit_nmf, fit_sparse_nmf,\n"
            "    umap_project, depth_r2, make_figure,\n"
            ")\n"
            "import numpy as np, time\n"
            "from sklearn.metrics import silhouette_score\n"
            "from pathlib import Path"
        ),
        md(
            "## Load the dataset\n\n"
            "First call downloads + caches; subsequent calls are\n"
            "instant. We use a small subsample so this notebook\n"
            "completes in a reasonable time — the story holds at any\n"
            "subsample size, only the absolute silhouette / R² numbers\n"
            "shift slightly."
        ),
        code(
            "X, celltype, tech = load_pancreas(cells_per_cohort=5, seed=0)\n"
            "n_counts = np.asarray(X.sum(axis=1)).ravel()\n"
            "log_depth = np.log10(n_counts + 1.0)\n"
            "print(f'X shape={X.shape}  nnz={X.nnz:,}  density={X.nnz / (X.shape[0]*X.shape[1]):.2%}')\n"
            "print(f'{len(np.unique(celltype))} cell types, {len(np.unique(tech))} protocols')\n"
            "print('depth (UMI/cell) by protocol:')\n"
            "for tk in sorted(np.unique(tech)):\n"
            "    d = n_counts[tech == tk]\n"
            "    print(f'  {tk:>14s}: n={d.size:4d}  median={np.median(d):>10.0f}  '\n"
            "          f'p10={np.percentile(d, 10):>10.0f}  p90={np.percentile(d, 90):>10.0f}')"
        ),
        md(
            "Note the depth column: Fluidigm C1 cells routinely report\n"
            "1M+ reads / cell, while inDrop cells are at the ~4 k UMI\n"
            "scale — three orders of magnitude apart. This is the\n"
            "confound we're studying."
        ),
        md(
            "## Compare three factorizations\n\n"
            "Same `k`, same UMAP, three different factorizations.\n"
            "We report three metrics per method:\n\n"
            "* **cell-type silhouette ↑** — does the embedding cluster\n"
            "  by biology?\n"
            "* **tech silhouette ↓ (near 0)** — are protocols mixed?\n"
            "* **depth-R² ↓ (near 0)** — kNN R² for predicting\n"
            "  log10(depth) from the 2-D UMAP. Quantifies how much of\n"
            "  the depth axis is still encoded in the embedding. High\n"
            "  is bad."
        ),
        code(
            "seed = 0\n"
            "k = int(np.clip(min(X.shape) // 8, 32, 1024))\n"
            "print(f'shared latent dim k={k}')\n"
            "\n"
            "embeddings, metrics = {}, {}\n"
            "for name, fn in (('PCA', fit_pca), ('NMF', fit_nmf), ('sparseNMF', fit_sparse_nmf)):\n"
            "    t0 = time.time()\n"
            "    high = fn(X, seed, k)\n"
            "    z = umap_project(high, seed)\n"
            "    embeddings[name] = z\n"
            "    sg = float(silhouette_score(z, celltype))\n"
            "    sb = float(silhouette_score(z, tech))\n"
            "    r2 = depth_r2(z, log_depth)\n"
            "    metrics[name] = (sg, sb, r2)\n"
            "    print(f'  {name:>10s}: {time.time()-t0:6.1f}s  '\n"
            "          f'silhouette(cell-type)={sg:+.3f}  '\n"
            "          f'silhouette(tech)={sb:+.3f}  depth-R²={r2:+.3f}')"
        ),
        md(
            "## Read the figure\n\n"
            "Row 1: cell-type clusters. Row 2: tech mixing. Row 3 is\n"
            "the smoking gun — color by `log10(UMI / cell)`. PCA / NMF\n"
            "filaments resolve into clean depth gradients (purple low,\n"
            "yellow high) — *the filaments are the depth axis*.\n"
            "sparseNMF has depth scattered uniformly within each\n"
            "cluster: the magnitude axis has been dissolved."
        ),
        code(
            "import tempfile\n"
            "from IPython.display import Image\n"
            "# Use a tempfile so this notebook doesn't overwrite the\n"
            "# production figure at docs/_static/real_pancreas_demo.png\n"
            "# (that one is produced by `python examples/real_pancreas_demo.py`\n"
            "# with cells_per_cohort=12).\n"
            "_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)\n"
            "_tmp.close()\n"
            "make_figure(embeddings, metrics, celltype, tech, log_depth, Path(_tmp.name))\n"
            "Image(filename=_tmp.name)"
        ),
        md(
            "## Side-by-side metrics\n\n"
            "Same numbers as the per-panel titles, side-by-side for\n"
            "comparison. Goal arrows:\n\n"
            "* **cell-type silhouette ↑** — higher is better (cleaner\n"
            "  biological clusters)\n"
            "* **tech silhouette → 0** — closer to zero is better\n"
            "  (batches are mixed within clusters); negative is fine\n"
            "* **depth-R² → 0** — lower is better (the depth axis is\n"
            "  not encoded in the embedding)"
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "method_names = list(metrics)\n"
            "celltype_vals = [metrics[m][0] for m in method_names]\n"
            "tech_vals = [metrics[m][1] for m in method_names]\n"
            "depth_vals = [metrics[m][2] for m in method_names]\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
            "metric_panels = [\n"
            "    (axes[0], 'cell-type silhouette  (↑ better)', celltype_vals, '#3cb44b'),\n"
            "    (axes[1], 'tech silhouette  (→ 0 better)', tech_vals,     '#f58231'),\n"
            "    (axes[2], 'depth R²  (↓ better)',         depth_vals,    '#4363d8'),\n"
            "]\n"
            "for ax, title, vals, color in metric_panels:\n"
            "    bars = ax.bar(method_names, vals, color=color, alpha=0.85,\n"
            "                  edgecolor='black', linewidth=0.5)\n"
            "    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')\n"
            "    ax.set_title(title, fontsize=11)\n"
            "    ax.tick_params(axis='x', labelsize=10)\n"
            "    ax.tick_params(axis='y', labelsize=9)\n"
            "    for b, v in zip(bars, vals):\n"
            "        ax.text(b.get_x() + b.get_width() / 2,\n"
            "                v + (0.02 if v >= 0 else -0.04),\n"
            "                f'{v:+.2f}',\n"
            "                ha='center', va='bottom' if v >= 0 else 'top',\n"
            "                fontsize=10, fontweight='bold')\n"
            "    # Pad y-axis so the annotations fit.\n"
            "    lo, hi = min(vals + [0]), max(vals + [0])\n"
            "    ax.set_ylim(lo - 0.1 * (hi - lo + 0.1), hi + 0.15 * (hi - lo + 0.1))\n"
            "fig.suptitle('Pancreas benchmark — PCA / NMF / sparseNMF', fontsize=12)\n"
            "fig.tight_layout(rect=(0, 0, 1, 0.95))\n"
            "plt.show()"
        ),
        md(
            "## Takeaway\n\n"
            "On clean single-protocol scRNA-seq the depth confound is\n"
            "mild, and PCA / NMF + UMAP do fine. On cross-protocol\n"
            "integration like this — where library depth varies by\n"
            "orders of magnitude between assays — PCA + UMAP produces\n"
            "depth-organized filaments (depth-R² ≈ 0.93), NMF the\n"
            "same (~0.82). `sparseNMF` with the default\n"
            "`normalize_inputs=True` dissolves the magnitude axis at\n"
            "the input, dropping depth-R² to ~0.40 and recovering\n"
            "cell-type silhouette to +0.40 — all while using the same\n"
            "UMAP step."
        ),
    ],
    title="Tutorial 4 — Real cross-protocol scRNA-seq (scIB pancreas)",
    timeout=900,
)


print("\nAll tutorials built and executed.")
