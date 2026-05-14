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

Both are auto-fetched from figshare on first run; the benchmark
code lives at `benchmarks/scib_benchmark.py
<https://github.com/bschilder/sparseNMF/blob/main/benchmarks/scib_benchmark.py>`__
and the driver at `benchmarks/run_benchmark.py
<https://github.com/bschilder/sparseNMF/blob/main/benchmarks/run_benchmark.py>`__.

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

    pip install "sparse-nmf[viz]" harmonypy scvi-tools scib
    python -m benchmarks.run_benchmark --no-lisi   # arm64 Macs need --no-lisi

Full run on Linux x86_64 GPU (scIB papers' setup; ~1 hr total)::

    python -m benchmarks.run_benchmark --full

Notes
~~~~~

- The scIB LISI metric ships as a pre-compiled x86_64 .o binary; it
  does *not* load on arm64 (Apple Silicon). The driver has a
  ``--no-lisi`` flag that skips it and runs the rest of the suite.
  Full-suite numbers are only available from a Linux x86_64 run.
- kBET / PCR / HVG-conservation / cell-cycle / trajectory metrics
  are off by default — they need either R (kBET) or counts-layer
  references / cycle genes / pseudotime inputs that aren't always
  available. Toggle them on per dataset if you have the inputs.
- scVI is the only method here that doesn't share the
  factorization → UMAP recipe; it learns its own variational
  latent. The scIB metrics work on any 2-D-projectable embedding,
  so the comparison is still well-defined.
