<p align="center">
  <img src="docs/_static/logo.svg" alt="sparseNMF logo" width="480"/>
</p>

<h1 align="center">sparseNMF</h1>

<p align="center">
  <em>GPU-accelerated sparse non-negative matrix factorization with PyTorch.</em>
</p>

<p align="center">
  <a href="https://github.com/bschilder/sparseNMF/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/bschilder/sparseNMF/actions/workflows/ci.yml/badge.svg"/></a>
  <a href="https://github.com/bschilder/sparseNMF/actions/workflows/docker.yml"><img alt="Docker" src="https://github.com/bschilder/sparseNMF/actions/workflows/docker.yml/badge.svg"/></a>
  <a href="https://github.com/bschilder/sparseNMF/blob/main/coverage.svg"><img alt="Coverage" src="https://raw.githubusercontent.com/bschilder/sparseNMF/main/coverage.svg"/></a>
  <a href="https://sparseNMF.readthedocs.io/en/latest/"><img alt="Docs" src="https://readthedocs.org/projects/sparsenmf/badge/?version=latest"/></a>
  <a href="https://github.com/bschilder/sparseNMF/releases"><img alt="Releases" src="https://img.shields.io/badge/releases-on_GitHub-blue?logo=github&logoColor=white"/></a>
  <a href="https://github.com/bschilder/sparseNMF/pkgs/container/sparsenmf"><img alt="Container" src="https://img.shields.io/badge/ghcr.io-sparsenmf-2496ED?logo=docker&logoColor=white"/></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue"/>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"/></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"/></a>
</p>

<!--
  Note on badges while the repo is still private:
  - CI / Docker badges are GitHub-native and work for authenticated viewers.
  - Coverage badge is committed in-tree as ``coverage.svg`` (auto-updated by CI on push to main).
  - Other badges are static images so they work regardless of repo visibility.
  - When the repo goes public, swap the static "Releases" badge above for the
    dynamic ``img.shields.io/github/v/release/...`` shield that pulls actual version data.
-->

---

## Why this exists

Non-negative matrix factorization (NMF) is a workhorse for biomedical
data: it produces interpretable parts-based decompositions of count or
abundance matrices (gene-association, phenotype, single-cell, document-
term, …). Off-the-shelf NMF implementations either materialize a dense
copy of the input (memory-prohibitive for very sparse, high-dimensional
data — think 100k samples × 30k features at 0.5% density) or never
touch the GPU.

`sparseNMF` keeps the input on the device as a `torch.sparse` tensor,
processes it in mini-batches, and runs the multiplicative updates
end-to-end on CUDA. It also ships an optional **joint NMF +
autoencoder** model that learns the factorization and a low-dimensional
embedding in a single pass — useful when downstream tasks
(visualization, clustering, retrieval) want the latent code rather
than the W/H matrices directly.

A deeper survey of prior NMF implementations and where this package
sits among them lives in the docs:
[**Prior works** →](https://sparseNMF.readthedocs.io/en/latest/prior_works.html)

## Install

```bash
# from PyPI (when released)
pip install sparse-nmf

# from source
pip install git+https://github.com/bschilder/sparseNMF.git

# with viz extras
pip install "sparse-nmf[viz]"
```

GPU acceleration requires a PyTorch build with CUDA. CPU-only works for
correctness checks and small data.

## Quick start

### Standalone NMF

```python
from scipy.sparse import csr_matrix
from sparse_nmf import SparseNMF

X = csr_matrix(...)                                 # (n_samples, n_features)

nmf = SparseNMF(n_components=256, max_iter=500, device="cuda")
X_reduced = nmf.fit_transform(X)                    # → (n_samples, 256) dense
H = nmf.components_                                  # → (256, n_features) dense
```

### Joint NMF + autoencoder (recommended for downstream embeddings)

```python
from sparse_nmf import train_joint_model

z, model = train_joint_model(
    X,
    n_samples=X.shape[0],
    n_features=X.shape[1],
    nmf_components=256,
    latent_dim=2,
    device="cuda",
    n_epochs=100,
)
# z is (n_samples, 2) — drop into UMAP / scatterplot directly.
```

See [`examples/`](examples/) for runnable end-to-end scripts and the
[**API reference**](https://sparseNMF.readthedocs.io/en/latest/api.html)
for every public function.

## Sample data

The package bundles a small synthetic sparse matrix (rank-8 plus
noise, 500 × 1k, 5% density) for tests and quickstart:

```python
from sparse_nmf.data import load_synthetic_sparse, generate_synthetic_sparse

X = load_synthetic_sparse()
# or programmatically:
X = generate_synthetic_sparse(n_samples=10_000, n_features=5_000, n_components=16, seed=42)
```

## Container

A pinned CUDA image is published to GHCR on every release:

```bash
docker pull ghcr.io/bschilder/sparsenmf:latest
docker run --gpus all --rm -it ghcr.io/bschilder/sparsenmf:latest python -c "from sparse_nmf import SparseNMF; print(SparseNMF.__doc__)"
```

## Contributing

```bash
git clone https://github.com/bschilder/sparseNMF.git
cd sparseNMF
pip install -e ".[dev]"
pytest                       # tests + coverage
ruff check . && ruff format --check .
sphinx-build docs docs/_build/html
```

CI runs lint + types + tests + coverage on every PR. Releases are
tag-triggered (`v*`) and publish the wheel + Docker image
automatically.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If `sparseNMF` is useful in published work, please cite:

```bibtex
@software{schilder_sparsenmf_2026,
  author = {Schilder, Brian},
  title  = {sparseNMF: GPU-accelerated sparse non-negative matrix factorization},
  url    = {https://github.com/bschilder/sparseNMF},
  year   = {2026},
}
```
