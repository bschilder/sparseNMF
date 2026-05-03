Changelog
=========

Changes are tracked in the GitHub Releases page:
https://github.com/bschilder/sparseNMF/releases

The release workflow auto-generates notes from merged PRs on every
``v*`` tag, so the canonical changelog is whatever Releases shows.
This page is kept here so ``intersphinx`` / cross-references resolve.

v0.1.0 (initial)
----------------

* First public release.
* :class:`sparse_nmf.SparseNMF` — GPU-accelerated multiplicative-update
  NMF over ``torch.sparse`` inputs.
* :func:`sparse_nmf.train_joint_model` — joint NMF + autoencoder
  pipeline producing a low-dimensional embedding.
* Attention-extraction utilities for tracing latent dimensions back
  to feature columns.
* Bundled synthetic sample data + matching ``data.generate_synthetic_sparse``
  programmatic factory.
* CI (lint + types + multi-version pytest), GHCR Docker image,
  Read the Docs site.
