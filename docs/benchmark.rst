Benchmarks
==========

scIB-style integration benchmark
--------------------------------

We evaluate ``sparseNMF`` against the standard scRNA-seq integration
methods on the canonical scIB benchmark datasets (Luecken *et al.*
2022, *Nature Methods*,
`DOI 10.1038/s41592-021-01336-8 <https://doi.org/10.1038/s41592-021-01336-8>`__).
The benchmark setup, methods, and metric definitions are identical
to the scIB paper's; we just add ``sparseNMF`` (with the default
``normalize_inputs=True``) and a ``sparseNMF + nonzero_mse_weight=1.0``
variant to the method list.

**Datasets**

- **Pancreas** — 16,382 cells × 19,093 genes across 9 protocols
  (CEL-seq, CEL-seq2, Smart-seq2, Fluidigm C1, Smarter, inDrop 1–4).
  Library depth varies ~300× between protocols. 14 cell types.
- **Immune** — 33k cells × 12k genes across ~10 donor/study batches.
  Heterogeneous PBMC + BMMC data. ~16 cell types.

Both are auto-fetched from figshare on first run. The benchmark
is **subprocess-isolated**: each method runs in its own Python
process so framework state (CUDA contexts, JAX devices, Lightning
trainer globals) can't leak between methods. Layout:

- `benchmarks/run_benchmark.py
  <https://github.com/bschilder/sparseNMF/blob/main/benchmarks/run_benchmark.py>`__
  — orchestrator (forks subprocesses, aggregates results)
- `benchmarks/methods/
  <https://github.com/bschilder/sparseNMF/tree/main/benchmarks/methods>`__
  — one module per method (PCA, NMF, sparseNMF, Harmony, scVI)
- `benchmarks/metrics/
  <https://github.com/bschilder/sparseNMF/tree/main/benchmarks/metrics>`__
  — two scoring impls: ``scib_yosef`` (JAX rewrite, default) and
  ``scib_original`` (canonical Theis-lab)
- `benchmarks/io.py
  <https://github.com/bschilder/sparseNMF/blob/main/benchmarks/io.py>`__
  — shared dataset loaders, preprocessing helpers, embedding I/O

**Methods**

- **PCA** — sklearn (baseline, no batch correction)
- **NMF** — sklearn (baseline, no batch correction)
- **sparseNMF** — this package, defaults (``normalize_inputs=True``,
  ``patience=10``, ``n_components=auto``)
- **sparseNMF + nonzero** — sparseNMF with
  ``nonzero_mse_weight=1.0`` (loss on observed entries only)
- **Harmony** — Korsunsky *et al.* 2019 (``harmonypy`` Python port)
- **scVI** — Lopez *et al.* 2018 (``scvi-tools``)

**Metrics**

scIB defines two metric families:

- *Bio conservation* (cell-type signal preservation): NMI / ARI /
  ASW(label) / cLISI / isolated-label F1 + ASW
- *Batch correction* (batch / protocol mixing): graph connectivity
  / iLISI

Composite score is :math:`0.6 \cdot \text{bio} + 0.4 \cdot \text{batch}`
per the scIB paper. Higher is better on both axes.

Results
~~~~~~~

.. note::

   These numbers are produced by a single full-dataset run on a
   Linux x86_64 GPU pod (NVIDIA RTX A4000, scvi-tools 1.4.x). The
   table is auto-populated by ``benchmarks/run_benchmark.py``
   into ``benchmarks/results.md``; see the
   `results.csv <https://github.com/bschilder/sparseNMF/blob/main/benchmarks/results.csv>`__
   for the unrounded per-metric values.

.. include:: ../benchmarks/results.md
   :parser: myst_parser.sphinx_

Reproducing
~~~~~~~~~~~

CPU pilot (fast — subsampled to 50 cells per batch × cell-type cohort)::

    pip install "sparse-nmf[viz]" harmonypy scvi-tools scib-metrics
    python -m benchmarks.run_benchmark --no-lisi   # arm64 Macs need --no-lisi

Full run on Linux x86_64 GPU (scIB papers' setup; ~1 hr total)::

    python -m benchmarks.run_benchmark --full

Pick the metrics implementation::

    # YosefLab JAX rewrite (default; works without compiled binaries)
    python -m benchmarks.run_benchmark --metrics-impl scib_yosef

    # Original Theis-lab impl (canonical; needs a one-shot rebuild
    # of scib's LISI binary against host glibc)
    pip install scib
    bash benchmarks/scripts/rebuild_scib_lisi.sh
    python -m benchmarks.run_benchmark --metrics-impl scib_original

Notes
~~~~~

- The scIB LISI metric (``scib_original``) ships as a precompiled
  binary requiring glibc 2.38+. ``benchmarks/scripts/rebuild_scib_lisi.sh``
  recompiles it against the host glibc — works on Ubuntu 22.04
  (glibc 2.35) and macOS provided a C++11 toolchain is present.
- ``--metrics-impl scib_yosef`` (default) avoids the compiled-binary
  dance entirely. Composite aggregation differs from ``scib_original``
  by ~hundredths; do not compare composites across impls without
  recalibrating.
- kBET / PCR / HVG-conservation / cell-cycle / trajectory metrics
  are off by default — they need either R (kBET) or counts-layer
  references / cycle genes / pseudotime inputs that aren't always
  available. Toggle them on per dataset if you have the inputs.
- scVI is the only method here that doesn't share the
  factorization → UMAP recipe; it learns its own variational
  latent. The scIB metrics work on any 2-D-projectable embedding,
  so the comparison is still well-defined.
