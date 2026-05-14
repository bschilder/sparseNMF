Prior works
===========

A survey of NMF implementations and **methodological approaches** that
tackle the same problem ``sparseNMF`` is built around: factorizing
high-dimensional, very sparse, magnitude-heterogeneous matrices
(typically single-cell or phenome-scale count data) into interpretable
parts. This page covers two axes:

1. **Implementations** — existing software packages, their compute
   model, and where they break under the workloads that motivated
   this package.
2. **Methodological prior art** — published methods that attempted
   what ``sparseNMF`` does (whether or not they shipped a Python
   package), with concrete statistical / algorithmic differences
   against our approach.

Each entry below carries a link to the **paper** (DOI when published)
and the **distribution** (GitHub / Bioconductor / equivalent) so the
reader can drill down.

----

Classic implementations
-----------------------

scikit-learn — :class:`sklearn.decomposition.NMF`
   Reference implementation; multiplicative updates and coordinate
   descent solvers, well-tested. **Limitation**: input is densified
   to a NumPy ndarray internally for many code paths, so a 100k × 30k
   sparse matrix at 0.5 % density (~15 M non-zeros) explodes to a
   ~12 GB dense float32 array before the first iteration runs.

   Code: https://github.com/scikit-learn/scikit-learn

nimfa
   Pure-Python multi-algorithm NMF library (LSNMF, BMF, BD, ICM, …).
   Wide algorithm coverage and great for methods research, but no
   GPU path and again densifies internally for sparse inputs.

   Code: https://github.com/marinkaz/nimfa

scanpy / muon
   Use NMF (via either scikit-learn or nimfa under the hood) for
   topic-model-style decompositions of cell × gene matrices. Inherits
   the same memory characteristics; works for typical single-cell
   sizes (10k cells × 20k genes) but breaks at the 100k+ cohorts that
   motivated this package.

   Code: https://github.com/scverse/scanpy

GPU implementations
-------------------

cuML's ``cuml.NMF``
   GPU-native NMF inside the RAPIDS stack. Fast on dense input but
   doesn't support ``cupyx.scipy.sparse`` matrices as a first-class
   input — sparse data must be densified to a CuPy array first, which
   is the same memory wall as scikit-learn just with a different fail
   mode.

   Code: https://github.com/rapidsai/cuml

torchnmf
   PyTorch implementation of several NMF variants. Inputs are
   ``torch.Tensor``; sparse-tensor support is partial and limited to
   specific layouts. Closer in spirit to ``sparseNMF`` but the
   training loop densifies for the multiplicative updates.

   Code: https://github.com/yoyolicoris/torchnmf

NMF-GPU
   Earlier CUDA C++ implementation focused on bioinformatics.
   Excellent throughput for the cases it supports but requires
   building from source against a specific CUDA version, and the
   input format is a custom file layout rather than a standard sparse
   matrix.

   Code: https://github.com/bioinfo-cnio/NMF-GPU

Methodological prior art
------------------------

The papers below introduced methods that aim at one or more of the
problems ``sparseNMF`` addresses: factor-level sparsity, library-depth
removal, batch integration, count-data fidelity, or scale. Each entry
identifies the camp the method belongs to and how it differs from
``sparseNMF`` in **what** is done, not just **how** it is implemented.

Factor-sparseness constraints (the classical "sparse NMF")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hoyer 2004 — "Non-negative Matrix Factorization with Sparseness Constraints"
   The foundational paper on what the NMF literature calls *sparse
   NMF*: imposes a user-specified sparseness target on the columns of
   :math:`W` and/or :math:`H`. Sparseness is measured as

   .. math::

      \sigma(x) = \frac{\sqrt{n} - \|x\|_1 / \|x\|_2}{\sqrt{n} - 1}
      \quad \in [0, 1]

   where :math:`n` is the vector dimension. After each gradient /
   multiplicative-update step, each column / row of the factors is
   *projected* onto the nearest non-negative vector with the desired
   :math:`L_1` and :math:`L_2` norms (and thus the target
   :math:`\sigma`). The input :math:`X` is column-normalized to unit
   Euclidean length; otherwise no special treatment of sparse inputs.

   **Diff vs sparseNMF.** This is a name collision worth flagging: in
   the Hoyer line of work "sparse" refers to *sparsity constraints on
   the factors* (W / H rows have many zeros). ``sparseNMF`` has no
   :math:`\sigma` knob and no projection step — "sparse" here refers
   to the *input matrix format* (``torch.sparse`` COO/CSR kept on
   device end-to-end). Imposing factor sparseness via projection is
   orthogonal to handling sparse inputs efficiently, and the two ideas
   compose.

   Paper: https://www.jmlr.org/papers/v5/hoyer04a/hoyer04a.pdf
   (JMLR 5, 1457–1469, 2004). No first-party software package
   shipped; the algorithm is implemented in several downstream
   libraries (e.g. ``nimfa``'s ``SNMF``).

Consensus & robustness on top of standard NMF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kotliar et al. 2019 — cNMF (consensus NMF), eLife
   Runs scikit-learn NMF :math:`R` times with different random seeds,
   :math:`L_2`-normalizes each of the :math:`R \cdot K` spectra,
   removes outliers by mean Euclidean distance to nearest neighbours
   (:math:`L = \rho R`, :math:`\rho = 0.3`), :math:`K`-means clusters
   the rest, and takes the per-cluster median as the consensus spectra
   — then refits usages once with spectra fixed. Pre-processing is
   *per-gene unit-variance scaling* on highly-variable genes; no
   log-transform ("requires an arbitrary pseudo-count") and
   **explicitly no per-cell row normalization**: "cells with more
   counts can contribute more information." Library depth is meant to
   be absorbed by the usage matrix, not preprocessed away.

   **Diff vs sparseNMF.** Polar-opposite stance on row normalization:
   ``sparseNMF`` defaults to L2 row-normalize precisely so the
   factorization can't latch onto the library-depth axis (see the
   "Why sparseNMF?" demo in the README); cNMF treats that axis as
   informative and lets a usage component carry it. Both can be
   right — the choice depends on whether per-cell totals are
   biological (cNMF) or technical (sparseNMF, single-cell with strong
   protocol confounds). Implementation-wise: dense sklearn NMF on
   CPU, no batching, K must be hand-swept.

   Paper: https://doi.org/10.7554/eLife.43803 (eLife 8:e43803, 2019;
   PMID 31282856; PMC6639075).
   Code: https://github.com/dylkot/cNMF

Stein-O'Brien et al. 2017 — PatternMarkers & GWCoGAPS, *Bioinformatics*
   PatternMarkers ranks genes per pattern :math:`k` by

   .. math::

      s_{ij} = \sum_k \left(\frac{A_{ik}}{\max_i A_{ik}} -
      \bar{w}_{jk}\right)^2

   with :math:`\bar{w}_j` a one-hot pattern indicator — smaller
   :math:`s` ⇒ more uniquely associated with pattern :math:`k`,
   yielding unsupervised marker selection without external labels.
   GWCoGAPS is the genome-wide distributed mode of CoGAPS.

   Paper: https://doi.org/10.1093/bioinformatics/btx058 (PMID
   28174896; PMC5860188). Companion to the CoGAPS entry below.

Bayesian / probabilistic NMF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CoGAPS (Sherman et al. 2020, *Bioinformatics*; Stein-O'Brien et al. 2017)
   Bayesian sparse NMF: decomposes :math:`D \approx AP` via a
   **Markov-chain Monte Carlo Gibbs sampler** with an **atomic prior**
   on both :math:`A` and :math:`P`. Atoms are placed on a 1-D domain
   via a Poisson process and mapped to matrix positions, inducing a
   prior that favours **exact zeros** in the factors (true sparsity,
   not :math:`L_1` shrinkage). CoGAPS 3 added sparse-matrix input
   (HDF5Array / CSC), single-cell mode (``sparseOptimization=TRUE``),
   asynchronous Gibbs sampling, and a distributed mode (GWCoGAPS /
   scCoGAPS) that partitions genes or cells across processes.
   Gaussian likelihood; no Poisson count model. CPU only.

   **Diff vs sparseNMF.** Different paradigms. CoGAPS returns a full
   *posterior* on :math:`A`, :math:`P` with uncertainty; ``sparseNMF``
   returns a point-estimate W, H via multiplicative updates. CoGAPS
   gives PatternMarkers (unsupervised gene selection) and exact-zero
   factor sparsity from the atomic prior — neither in ``sparseNMF``.
   On the other hand, CoGAPS is famously slow (hours-to-days on bulk;
   distributed mode required for scRNA-seq), CPU-only, while
   ``sparseNMF`` runs a 100k+ cell sparse matrix on a single GPU in
   minutes.

   Papers: https://doi.org/10.1093/bioinformatics/btaa1042 (Sherman
   et al. 2020; PMID 31764967) and the PatternMarkers companion
   above.
   Code: https://github.com/FertigLab/CoGAPS (R, Bioconductor:
   https://bioconductor.org/packages/CoGAPS/), and
   https://github.com/FertigLab/PyCoGAPS (Python wrapper).

Johnson et al. 2023 — CoGAPS protocol, *Nature Protocols*
   A how-to-run-CoGAPS paper covering PyCoGAPS / Docker / R /
   GenePattern. Explicitly recommends **log-normalized** input
   (``sc.pp.log1p`` after ``.todense()``), warns against upstream batch
   correction ("CoGAPS concurrently learns technical and biological
   signals"), and leaves cell/gene filtering, HVG selection, and
   imputation to the user.

   **Diff vs sparseNMF.** The protocol surfaces what's implicit in
   most NMF tooling: a CoGAPS workflow front-loads several
   preprocessing decisions (densify, log, transpose, gene filter)
   onto the analyst. ``sparseNMF`` inverts this — defaults
   (``normalize_inputs=True``, auto-sized :math:`k`, sparse input
   accepted directly) are tuned so a single zero-config call lands
   in the empirically-good region for typical sparse count data.

   Paper: https://doi.org/10.1038/s41596-023-00821-y (PMID 37989764;
   PMC10961825).
   Code: see CoGAPS / PyCoGAPS entries above.

Argelaguet et al. 2018 (MOFA) and 2020 (MOFA+), *Molecular Systems Biology* & *Genome Biology*
   Bayesian sparse factor analysis (not NMF) for multi-omics:
   :math:`Y_{gm} = Z_g W_m^\top + \varepsilon_{gm}` per modality
   :math:`m` and group :math:`g`. Per-feature likelihoods are
   Gaussian (continuous), Poisson (counts), or Bernoulli (binary);
   non-Gaussian cases use local variational bounds. Sparsity is
   enforced on the *loadings* :math:`W` via a two-level prior:
   **automatic relevance determination (ARD)** per factor / view
   pushes whole factors off in views where they aren't needed, and a
   **spike-and-slab** prior on individual weights zeros out features
   within a factor. Inference: stochastic variational Bayes
   maximizing the ELBO with mini-batches; GPU acceleration when data
   fits memory.

   **Diff vs sparseNMF.** (1) **Sign**: MOFA+ factors and loadings
   are real-valued; ``sparseNMF`` enforces non-negativity through
   multiplicative updates. (2) **Sparsity locus**: MOFA+ induces
   sparsity on *loadings* via spike-and-slab + ARD; ``sparseNMF``
   ingests sparse *data* — no factor-level sparsity prior. (3)
   **Library depth**: MOFA+ recommends external size-factor
   normalization + variance stabilization for RNA-seq (or the Poisson
   likelihood without an offset); ``sparseNMF`` handles it inside the
   one-liner with default L2 row-normalization. (4) **Output**: MOFA+
   returns a variational *posterior*; ``sparseNMF`` returns a point
   estimate.

   Papers: https://doi.org/10.1186/s13059-020-02015-1 (MOFA+, 2020;
   PMID 32393329; PMC7212577) and
   https://doi.org/10.15252/msb.20178124 (MOFA, 2018; PMID 29925568;
   PMC6010767).
   Code: https://github.com/bioFAM/MOFA2 (R + Python wrappers),
   https://github.com/bioFAM/mofapy2 (Python backend).

Batch-aware factorization
~~~~~~~~~~~~~~~~~~~~~~~~~

LIGER (Welch et al. 2019, *Cell*) and online iNMF (Gao et al. 2021, *Nature Biotechnology*)
   Integrative NMF: jointly factorizes :math:`K` datasets
   :math:`X_i \in \mathbb{R}^{m \times n_i}_+` as

   .. math::

      \min_{W, V_i, H_i \ge 0}
      \sum_i \|X_i - (W + V_i) H_i^\top\|_F^2
           + \lambda \|V_i H_i^\top\|_F^2

   :math:`W \in \mathbb{R}^{m \times K}_+` is the **shared** metagene
   matrix (signal common across datasets);
   :math:`V_i \in \mathbb{R}^{m \times K}_+` is a per-dataset
   *deviation* metagene matrix (dataset-specific signal, regularized
   by :math:`\lambda`); :math:`H_i` is per-cell factor loadings. The
   2021 online extension keeps :math:`X_i` on disk (HDF5 chunks),
   streams mini-batches, and maintains sufficient statistics
   :math:`A_i = \sum H H^\top`, :math:`B_i = \sum X H` (memory
   :math:`O(K^2)` and :math:`O(mK)` instead of :math:`O(\text{cells})`).
   Solvers: **block-principal-pivoting NNLS** for :math:`H_i` per
   cell, **HALS** column updates for :math:`W` and :math:`V_i`.
   Preprocessing: per-cell size-normalize → HVG selection per dataset
   → divide each gene by its root-mean-square (no centring — preserves
   :math:`X \ge 0`). Post-factorization: quantile-aligns :math:`H_i`
   across datasets for the final integrated embedding.

   **Diff vs sparseNMF.** LIGER bakes batch correction into the
   *model* (the per-dataset :math:`V_i` term + quantile alignment);
   ``sparseNMF`` has no :math:`V_i` analogue and pushes batch handling
   into *preprocessing* (L2 row-normalize). Both compose: LIGER's
   :math:`W` could in principle be initialized from a ``sparseNMF``
   fit. Implementation-wise: LIGER scales-without-centering, which
   densifies selected genes; ``sparseNMF`` keeps everything on the
   GPU as a sparse tensor. LIGER requires a hand-chosen :math:`K`
   (typically 20–40); ``sparseNMF`` auto-sizes from input shape.

   Papers: https://doi.org/10.1038/s41587-021-00867-x (online iNMF,
   Gao 2021; PMID 33875866; PMC8355612) and
   https://doi.org/10.1016/j.cell.2019.05.006 (LIGER, Welch 2019).
   Code: https://github.com/welch-lab/liger (R: ``rliger`` on CRAN;
   Python: ``pyliger``).

Normalization as a stand-in for factorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hafemeister & Satija 2019 — sctransform, *Genome Biology*
   Not a factorization. Fits a regularized negative-binomial GLM per
   gene:

   .. math::

      \log \mathbb{E}[x_{ij}] = \beta_{0, i} +
        \beta_{1, i} \log_{10}(m_j),
      \quad
      \mathrm{Var}(x_{ij}) = \mu_{ij} + \mu_{ij}^2 / \theta_i

   with :math:`m_j` the total UMI in cell :math:`j`. Per-gene
   parameters :math:`(\beta_0, \beta_1, \theta)` are first estimated
   independently, then regularized by **kernel smoothing** (bandwidth
   from ``bw.SJ``, BAF = 3) as smooth functions of mean expression —
   pooling info across genes of similar abundance to prevent
   overfitting of low-abundance genes. Output: **Pearson residuals**
   :math:`z_{ij} = (x_{ij} - \mu_{ij}) / \sqrt{\mu_{ij} +
   \mu_{ij}^2/\theta_i}`, clipped to :math:`\pm\sqrt{N}`.

   **Diff vs sparseNMF's** ``normalize_inputs=True``. ``sparseNMF``'s
   L2 row-normalize applies *one global scalar per cell* to all
   genes; sctransform's :math:`\beta_{1, i}` is **gene-specific** and
   smoothly varies with mean expression. Hafemeister & Satija show
   that simple row-normalization only de-trends mid-abundance genes —
   high-abundance ones remain depth-correlated; sctransform's
   gene-specific slope is the whole point. The price is (a) ~100–1000×
   slower (per-gene GLMs + kernel smoothing) and (b) **Pearson
   residuals can be negative**, breaking NMF's non-negativity
   invariant. Practical workflow: residuals go into PCA / UMAP /
   clustering, not NMF. For NMF + depth correction, the principled
   bridge is GLM-PCA or count-NMF variants, not sctransform-then-NMF.

   Paper: https://doi.org/10.1186/s13059-019-1874-1 (Genome Biology
   20:296, 2019; PMID 31870423; PMC6927181).
   Code: https://github.com/satijalab/sctransform (also wrapped in
   Seurat as ``SCTransform()``).

Supervised / reference-anchored NMF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hevdeli et al. 2025 — CellMentor, *Nature Communications*
   Supervised NMF for scRNA-seq dimensionality reduction. Adds a
   Fisher-style discriminant term (within-class compactness
   :math:`S_{\widetilde W}` + between-class separation
   :math:`S_{\widetilde B}`) to the standard reconstruction loss
   :math:`\tfrac{1}{2}\|X - WH\|_F^2`, with :math:`L_1` sparsity on
   :math:`H` (:math:`\gamma \sum H`) and an orthogonality penalty on
   :math:`W` (:math:`\tfrac{\delta}{2}\,\mathrm{tr}(W^\top W M)`).
   Two-phase fit: (1) learn :math:`W`, :math:`H_\mathrm{ref}` from a
   labeled reference; (2) project the query onto :math:`W` via
   non-negative least squares. Preprocessing:
   :math:`\log(\mathrm{count}/\mathrm{lib} \times 10^4 + 1)` + RMSE
   scaling. CPU only.

   **Diff vs sparseNMF.** CellMentor occupies the opposite design
   point: **supervised**, requires a pre-labeled reference atlas, and
   is sensitive to reference-label quality. ``sparseNMF`` is
   unsupervised and targets exploration where no trusted atlas
   exists. Different niches — CellMentor is closer to a supervised
   projection / label-transfer tool than to a discovery NMF.

   Paper: https://doi.org/10.1038/s41467-025-67088-7 (PMID 41381456;
   PMC12796484).
   Code: https://github.com/petrenkokate/CellMentor (Apache-2.0).

High-performance sparse-input NMF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeBruine et al. 2021 — RcppML, bioRxiv preprint
   The closest *engineering* parallel to ``sparseNMF``. R + Rcpp /
   Eigen C++ implementation with **OpenMP** threading and an
   optional CUDA path; solvers are **alternating least squares**,
   **coordinate descent**, and **Cholesky NNLS** (auto-selected via
   ``solver`` argument) — no Lee–Seung multiplicative updates as the
   primary path. Sparse-input-aware: detects column-compressed
   sparse (``dgCMatrix``) and dispatches optimized sparse routines.
   Headline claim: 10–20× speedups on large matrices vs CPU NMF
   baselines.

   **Diff vs sparseNMF.** Same "sparse" semantics (input format, not
   factor constraint), different stack: R + Rcpp/Eigen (+ optional
   CUDA) vs Python + PyTorch (GPU-first sparse tensors). RcppML's
   primary path is CPU + OpenMP; ``sparseNMF`` is GPU + mini-batched
   sparse-tensor MU. Algorithm-wise: ALS / coordinate descent /
   Cholesky NNLS vs multiplicative updates. Both are designed for
   genuinely sparse single-cell counts; they're complementary rather
   than competing.

   Paper: https://doi.org/10.1101/2021.09.01.458620 (bioRxiv
   preprint).
   Code: https://github.com/zdebruine/RcppML

Alternative strategies for removing the depth confound
------------------------------------------------------

Depth, sparsity, and how they relate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two related-but-distinct per-cell quantities show up throughout this
discussion:

- **Library depth** :math:`d_i = \sum_j X_{ij}` — the row sum.
  *How many counts* were captured for cell :math:`i`.
- **Per-cell sparsity** :math:`s_i = 1 - \mathrm{nnz}(X_{i,\cdot}) /
  n_\mathrm{features}` — the fraction of zero entries in cell
  :math:`i`'s row. *How many distinct features* are detected at all.

In count data these are tightly but **non-linearly** coupled
(rarefaction / species-accumulation behaviour): adding a single read
to a low-depth cell almost always lands on a new feature
(non-zero count goes up by 1, sparsity drops); adding a read to a
high-depth cell mostly lands on an already-detected feature (depth
goes up, sparsity barely moves). Empirically:

- At very low :math:`d_i`, :math:`\mathrm{nnz}_i \approx d_i` — every
  read = one new gene.
- At high :math:`d_i`, :math:`\mathrm{nnz}_i` saturates toward the
  gene catalogue size, while :math:`d_i` keeps growing.

In the README demo, the two batches were constructed with 10× different
*depth* and the figure colours by *nnz* — they're nearly
interchangeable summaries at the scale used.

The confound from a factorization's point of view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depth is largely *technical* (how well a particular cell was
sequenced) but it dominates the variance of any individual gene, so
unsupervised factorizations tend to allocate their leading components
to "tracking the row sum" rather than to biology. Sparsity is the
secondary expression of the same technical axis. The two coupled
effects:

1. **Magnitude**: row totals differ by orders of magnitude → cells
   with more counts dominate the squared-error loss.
2. **Sparsity pattern**: cells with deeper sequencing have a different
   *set of zero entries*; same biology can produce two cells that
   look very different just because one had more reads and so more
   genes broke above zero.

L2 row-normalization (Strategy 3 below) collapses (1) exactly, but
only attenuates (2): an :math:`L_2`-normalized low-depth cell has each
of its few non-zero entries scaled *up* to compensate, while a
high-depth cell's many entries are scaled *down*. That's why the
``nonzero_mse_weight`` knob exists — if the residual sparsity pattern
is still a meaningful driver of variance after L2-norm, evaluating
the loss only on observed (non-zero) entries can remove the second
order of confound too.

Three strategies
~~~~~~~~~~~~~~~~

Three broad strategies for removing the depth/sparsity confound are
in common use. ``sparseNMF`` takes the third — but it's worth being
explicit about the tradeoffs.

Strategy 1 — Explicit regression of depth (or batch) as a covariate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each gene :math:`g`, fit a generative model with depth (or batch,
or any other named nuisance) as a covariate, and downstream
analysis runs on the **residuals**:

.. math::

   \log \mathbb{E}[x_{ij}] = \beta_{0, j} + \beta_{1, j} \log d_i +
     \beta_{2, j} \, b_i + \ldots,
   \quad
   z_{ij} = (x_{ij} - \hat\mu_{ij}) / \sigma_{ij}

This is the path taken by **sctransform** (per-gene NB GLM,
regularized via kernel smoothing; see the dedicated entry above),
**Seurat's** ``ScaleData(vars.to.regress = ...)`` (per-gene linear
regression on user-named covariates), **ComBat** / **ComBat-seq**
(empirical-Bayes batch correction, Gaussian or NB), and **limma**'s
``removeBatchEffect`` (linear regression of nuisance design
matrices). **Harmony** is a related-but-different idea: it runs the
correction in PCA *embedding* space, iteratively maximum-diversity-
clustering and centring within batches, rather than on the count
matrix.

**Pros.** Statistically principled — there's an explicit generative
model of the nuisance. Targeted — removes only the variance
associated with the named covariate. Composes with downstream
methods that accept real-valued input (PCA, UMAP, clustering).

**Cons.** Requires the analyst to *name* the covariate up front; if
the confound isn't perfectly correlated with depth or with the labelled
batch, the regression misses it. Can over-correct — biological signal
that genuinely correlates with depth (e.g. cell-type-specific
transcriptional output) gets removed too. Per-gene model fitting is
slow (sctransform is ~100–1000× slower than L2 row-norm) and the
residuals are **real-valued, often negative** — incompatible with
downstream NMF, which requires :math:`X \ge 0`. So for NMF
specifically, this path forces a chain of "regress → PCA / GLM-PCA →
clustering", not "regress → NMF".

References:

- sctransform (per-gene NB regression):
  Hafemeister & Satija 2019,
  `10.1186/s13059-019-1874-1 <https://doi.org/10.1186/s13059-019-1874-1>`__,
  code https://github.com/satijalab/sctransform.
- ComBat (empirical-Bayes batch correction):
  Johnson, Li & Rabinovic 2007 *Biostatistics* 8(1):118–127,
  `10.1093/biostatistics/kxj037 <https://doi.org/10.1093/biostatistics/kxj037>`__,
  PMID 16632515. Bioconductor: ``sva``,
  https://bioconductor.org/packages/sva/.
- ComBat-seq (NB version for count data):
  Zhang, Parmigiani & Johnson 2020 *NAR Genom Bioinform* 2(3):lqaa078,
  `10.1093/nargab/lqaa078 <https://doi.org/10.1093/nargab/lqaa078>`__,
  PMID 33015620.
- limma ``removeBatchEffect``:
  Ritchie *et al.* 2015 *Nucleic Acids Res* 43(7):e47,
  `10.1093/nar/gkv007 <https://doi.org/10.1093/nar/gkv007>`__,
  PMID 25605792. Bioconductor: ``limma``,
  https://bioconductor.org/packages/limma/.
- Harmony (embedding-space iterative correction):
  Korsunsky *et al.* 2019 *Nature Methods* 16(12):1289–1296,
  `10.1038/s41592-019-0619-0 <https://doi.org/10.1038/s41592-019-0619-0>`__,
  PMID 31740819. Code https://github.com/immunogenomics/harmony.

Strategy 2 — Drop the top N principal components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Folklore in single-cell pipelines: run PCA on the (normalized) count
matrix, inspect PC1 (and sometimes PC2..N) for correlation with total
counts, then proceed with the *later* PCs (``X_pca[:, k:]``) for
downstream clustering / UMAP. Implemented as a manual slicing step
in essentially every Scanpy / Seurat tutorial and as the
``n_pcs`` / ``use_pcs`` argument in many wrappers.

**Pros.** Trivial: a single array slice. No covariate label needed —
PCA finds whatever axis dominates and the user can decide whether to
keep it.

**Cons.** All-or-nothing per PC: throws away whatever biological
signal is co-mingled in PC1 (in real data, PC1 is rarely "purely"
library depth — it's a linear combination of depth, total
transcriptional activity, dominant cell type fractions, etc.). The
depth axis may also not be cleanly aligned with PC1 — it can split
across several PCs, leaving residual contamination after any fixed
cutoff. Requires manual inspection (correlate each PC with
:math:`d_i` and threshold); no principled stopping rule. Specifically
unhelpful for **NMF**: NMF doesn't produce orthogonal axes you can
sort by variance, so "drop the top N components" has no analogous
operation.

This strategy is widely used because it's easy, not because it's
right. The methodologically-careful path is one of the other two.

Strategy 3 — Input-level row normalization (``sparseNMF``'s default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply a deterministic transform to the *input* :math:`X` that
collapses the per-row magnitude axis before factorization. The
specific transform in this package is :math:`L_2` row normalization:
each cell vector is scaled to unit Euclidean length, so the row sum
is no longer a degree of freedom in the factorization.

.. math::

   \widetilde{X}_{ij} = X_{ij} \,/\, \|X_{i,\cdot}\|_2

After this, any signal that was per-cell magnitude (library depth,
total transcriptional output) is gone; only the *direction* of
expression (which genes are co-expressed and in what relative
proportions) survives. Equivalent in spirit to running NMF on cosine
similarities. This is the route ``sparseNMF`` takes by default
(``normalize_inputs=True``).

**Pros.** Zero-config — no covariate label required. Deterministic
and :math:`O(\text{nnz})` cheap. Preserves non-negativity, so
downstream NMF / NNLS runs natively. Removes **all** of the magnitude
axis at once, including parts that don't perfectly correlate with the
"batch" label (which Strategy 1 would miss). Compatible with sparse
storage end-to-end.

**Cons.** Coarse — discards genuinely biological per-cell magnitude
(e.g. cell-cycle state, total metabolic activity, secretory cells
that produce more mRNA than quiescent ones). All-or-nothing on the
magnitude axis: unlike sctransform's gene-specific
:math:`\beta_{1, j}`, every gene gets de-trended by the same per-cell
scalar. Doesn't help with **non-magnitude** confounds — if Batch A
has a gene specifically upregulated that doesn't track its cells'
totals, L2-norm leaves that batch effect untouched. The user must
know that depth / magnitude is the dominant nuisance for this
strategy to be the right call.

How to choose: concrete scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The right strategy depends on (a) what's *driving* the depth/sparsity
variation in your data, and (b) what downstream method you're feeding.
Five common scenarios with explicit recommended calls:

**Scenario A — Multi-protocol scRNA-seq integration.** Cells assayed
with different protocols (e.g. 10x v2 ~500 UMIs/cell vs Smart-seq2
~5,000 UMIs/cell), same biological cell types. Depth varies by ~10×
with batch and is mostly technical.

   Use ``sparseNMF`` with defaults — ``train_sparse_nmf(X)`` already
   has ``normalize_inputs=True``, which dissolves the magnitude axis.
   If you see residual batch sub-clusters per group (the demo figure
   shows what this looks like), add ``nonzero_mse_weight=1.0`` to
   stop the model from also predicting the *pattern* of zeros. If
   batch labels are reliable and you'd rather correct in the
   embedding, run ``sparseNMF`` first then ``Harmony`` on the
   resulting embedding — they compose.

**Scenario B — Single-protocol scRNA-seq across cell types with
genuinely different mRNA content.** All cells run on one 10x lane,
but you're profiling e.g. neurons (transcriptionally hyperactive,
~5,000 UMIs) alongside resting lymphocytes (~500 UMIs). The depth
gradient *is* biology — you want to keep it.

   Pass ``normalize_inputs=False`` to ``sparseNMF``. One factor will
   pick up the per-cell magnitude as a "total activity" signal,
   leaving the others free to encode cell type. Alternatively use
   **cNMF** (`Kotliar 2019
   <https://doi.org/10.7554/eLife.43803>`__) — same philosophical
   stance, built around per-gene variance scaling rather than
   per-cell row-norm.

**Scenario C — Single-protocol scRNA-seq with technical batches you
want gone.** All 10x, but assayed across 3 wet-lab runs that
introduce a per-batch shift in *which genes* are detectable (not
just per-cell depth — different reagents preferentially capture
different transcripts).

   This is the case where Strategy 3 alone is insufficient — the
   confound is batch-specific *direction*, not magnitude. Use
   **LIGER / online iNMF** (`Welch 2019 / Gao 2021
   <https://doi.org/10.1038/s41587-021-00867-x>`__) which factorizes
   :math:`X_i \approx (W + V_i) H_i` with batch-specific
   :math:`V_i`, or run ``sparseNMF`` then apply **Harmony**
   (`Korsunsky 2019 <https://doi.org/10.1038/s41592-019-0619-0>`__)
   on the resulting embedding.

**Scenario D — scATAC-seq (or any near-binary count matrix).** Counts
are effectively 0/1, sparsity per cell is extreme and varies strongly
with sequencing depth.

   Same as Scenario A but more important: depth differences here
   manifest almost entirely as sparsity-pattern differences. Pass
   ``normalize_inputs=True`` and consider also binarizing the input
   (``X.data = (X.data > 0).astype(np.float32)``). The
   ``nonzero_mse_weight`` knob is particularly useful here. **CoGAPS**
   in single-cell mode (``sparseOptimization=TRUE``) is a slower but
   more principled alternative if you want posterior uncertainty.

**Scenario E — Sparse non-genomics data (phenome × disease, document
× term, user × item).** Row sums vary because of the underlying
process (more disease codes for sicker patients, longer documents,
more active users), not technical capture. You typically want to
compare *patterns* not magnitudes.

   ``sparseNMF`` defaults are designed for this. The L2 row-norm
   makes each row a unit-magnitude direction, equivalent in spirit
   to comparing rows by cosine similarity. Set
   ``nonzero_mse_weight=0`` (the default) so the zeros — which here
   *are* informative ("this patient does not have this code") —
   continue to contribute to the loss.

When NOT to use ``sparseNMF``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The downstream task is **differential expression**, not embedding /
  clustering. Use sctransform Pearson residuals + a count GLM
  (DESeq2, edgeR, limma), not NMF.
- You need **posterior uncertainty** on W and H. Use CoGAPS.
- You have a **labeled reference atlas** and want a supervised
  projection. Use CellMentor or Seurat's transfer-learning
  utilities.
- You need a **likelihood-based generative model** (e.g. for
  imputation or counterfactual queries). Use scVI / scANVI.

Beyond single-cell: factorization of gene-signature matrices
------------------------------------------------------------

The depth-confound framing above is written in single-cell terms
because that's where this package was first stress-tested, but the
input shape that ``sparseNMF`` is good at — a **sparse, non-negative
matrix with magnitude-heterogeneous rows** — generalizes to a much
wider class of biological data. Four common shapes where the same
machinery works without modification:

Pathway × gene membership matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Rows**: pathways / gene sets (MSigDB, Reactome, KEGG, GO BP).
- **Columns**: genes.
- **Entries**: :math:`0/1` (membership) or continuous :math:`\in
  [0, 1]` (weighted membership / topic-model-style soft assignment).
- **"Row sum"** :math:`d_i` = pathway *size* (number of member
  genes), which varies wildly — MSigDB pathways range from ~5 genes
  to ~5000.

Without normalization, factorization would weight large pathways
much more heavily. With ``normalize_inputs=True``, every pathway
contributes equally and ``sparseNMF`` discovers **pathway themes** —
latent metafeatures grouping pathways that share gene-overlap
structure ("metabolism", "immune", "cell cycle" emerge as their own
factors). Useful for cross-database integration (KEGG ↔ Reactome)
or for pruning a redundant gene-set collection down to a small
non-overlapping basis (analogous to the hallmark approach of
Liberzon et al. 2015).

References:

- GSEA / MSigDB:
  Subramanian *et al.* 2005 *PNAS* 102(43):15545–15550,
  `10.1073/pnas.0506580102 <https://doi.org/10.1073/pnas.0506580102>`__,
  PMID 16199517.
- MSigDB hallmark gene-set collection:
  Liberzon *et al.* 2015 *Cell Systems* 1(6):417–425,
  `10.1016/j.cels.2015.12.004 <https://doi.org/10.1016/j.cels.2015.12.004>`__,
  PMID 26771021.
- MSigDB: https://www.gsea-msigdb.org/gsea/msigdb/
- Reactome: https://reactome.org/

Differential-expression signature × gene matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Rows**: signatures, one per study × contrast × threshold (sources:
  GEO-curated signature collections, LINCS L1000 / CMap perturbation
  signatures, published DGE supplementary tables).
- **Columns**: genes.
- **Entries**: :math:`0` (not significant), :math:`1` (significantly
  up), :math:`-1` (significantly down) — or continuous log2 fold
  change.
- **"Row sum"** :math:`d_i` = number of significantly differentially
  expressed genes, which is dominated by **study power** (sample
  size, effect size, multiple-testing correction choice) far more
  than by biology.

NMF needs non-negativity, so signed signatures should be **split
into "up" and "down" half-rows** before factorization (one signature
becomes two rows: gene memberships at the positive tail, and at the
negative tail, separately). With that split applied,
``train_sparse_nmf(X)`` defaults find **shared transcriptional
programs across heterogeneous DGE studies**, even when those studies
used different platforms, thresholds, and statistical methods — the
L2 row-norm puts a low-power 50-DEG study and a high-power
500-DEG study on equal footing as long as the *direction* of
expression is similar.

This is genuinely hard to do with sctransform / DESeq2 / similar
pipelines because they operate on raw counts, not on
already-summarized signature matrices. ``sparseNMF`` operates at the
summarized level.

GWAS-derived gene lists
~~~~~~~~~~~~~~~~~~~~~~~

- **Rows**: traits / studies.
- **Columns**: genes (mapped from significant SNPs via MAGMA / FUMA /
  PoPS / nearest-gene).
- **Entries**: :math:`0/1` (gene mapped from a significant locus) or
  continuous (Z-score, posterior probability, PoPS score).
- **"Row sum"** :math:`d_i` = number of gene-level hits, which depends
  on the study's :math:`N`, the trait's heritability, its
  polygenicity, the ancestry-matched LD reference used — almost all
  technical / study-design factors rather than the trait's
  biology.

L2 row-norm makes a small high-effect-size autoimmune-disease GWAS
comparable to a million-sample polygenic-trait GWAS for the purposes
of finding **pleiotropic gene modules** — sets of genes that recur
across phenotypically-distinct traits. Useful for drug repurposing,
target prioritization, and trait-clustering studies. This is closely
related to (but cheaper than) genetic-correlation methods like LDSC
when the goal is gene-level rather than SNP-level pleiotropy.

References:

- GWAS Catalog (knowledgebase of published associations):
  Sollis *et al.* 2023 *Nucleic Acids Res* 51(D1):D977–D985,
  `10.1093/nar/gkac1010 <https://doi.org/10.1093/nar/gkac1010>`__,
  PMID 36350656.
- Catalog browser: https://www.ebi.ac.uk/gwas/
- Open Targets Genetics (gene-mapping aggregation):
  https://genetics.opentargets.org/

Disease-gene association matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Rows**: diseases (OMIM, DisGeNET, Open Targets, MONDO).
- **Columns**: genes.
- **Entries**: continuous association score (text-mining frequency,
  curator confidence, integrated evidence score).
- **"Row sum"** :math:`d_i` = study volume / curation depth per
  disease, which differs by orders of magnitude (cancer has thousands
  of associated genes in DisGeNET; rare diseases often have under
  10).

Same logic: L2 row-norm levels the playing field for finding
cross-disease gene modules. Useful for nosology refinement and
drug-target sharing across disease groups.

References:

- DisGeNET knowledge platform:
  Piñero *et al.* 2020 *Nucleic Acids Res* 48(D1):D845–D855,
  `10.1093/nar/gkz1021 <https://doi.org/10.1093/nar/gkz1021>`__,
  PMID 31680165. Platform: https://www.disgenet.org/
- Open Targets Platform: https://platform.opentargets.org/

What changes about the choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mapping the single-cell knobs to the signature-matrix setting:

- **"Depth"** stops meaning "library depth" and just means
  :math:`d_i = \sum_j X_{ij}` — the row sum. The semantic
  interpretation differs by source (pathway size, DEG count,
  GWAS-hit count, curation depth) but the mathematical role is
  identical.
- **``normalize_inputs=True``** is almost always the right default
  for these matrices: it converts "compare row patterns" into the
  factorization objective (akin to running on cosine similarities).
  The exception is when *row size itself* is the biological signal
  you want (e.g., "broadly pleiotropic trait" vs "narrowly pleiotropic
  trait" might be a feature, not a confound — in that case use
  ``normalize_inputs=False``).
- **``nonzero_mse_weight``** behaves differently than in single-cell:
  for binary or score-thresholded signature matrices, **zero
  literally means "not in this set"** and is informative — leave
  ``nonzero_mse_weight=0`` (the default). The "treat zeros as
  missing" case only applies when low row sum is technical
  (e.g., severely under-powered GWAS where a zero plausibly hides a
  true association).
- **No batch correction needed.** These matrices don't have the
  per-batch protocol-shift problem that LIGER/iNMF/Harmony solve. If
  you're integrating multiple databases (KEGG + Reactome + GO), the
  union of their gene-set rows is still a single sparse matrix —
  just stack them.

Minimal example — pathway × gene factorization::

   from sparse_nmf import train_sparse_nmf
   from scipy.sparse import csr_matrix
   import umap

   # X[pathway_i, gene_j] = 1 if gene_j is a member of pathway_i.
   # Build from GMT files (MSigDB), Reactome dumps, etc.
   X = csr_matrix(...)  # shape (n_pathways, n_genes), binary or weighted

   W, model = train_sparse_nmf(X)  # defaults: normalize_inputs=True,
                                   # n_components auto-sized
   # W[pathway_i, k] = how strongly pathway_i loads on metafeature k
   # Cluster pathways by their W rows -> pathway "themes".
   z = umap.UMAP(n_components=2, random_state=0).fit_transform(W)

The same call works unchanged for the DGE / GWAS / disease shapes
above — only the row semantics change.

Adjacent: deep generative factorization
---------------------------------------

scVI / scANVI / Poisson-NMF
   Probabilistic generative models that do an NMF-flavored
   decomposition as part of a deeper VAE. Different problem framing
   (likelihood-based, not multiplicative updates) and different
   output (a posterior, not point-estimate W / H). Worth knowing about
   for downstream analysis but not a drop-in NMF replacement.

   Code: https://github.com/scverse/scvi-tools

scGPT / single-cell foundation models
   Treat the cell × gene matrix as a sequence and learn embeddings
   via masked-token prediction. Solves a related problem (per-cell
   embedding) without producing the parts-based interpretable factors
   that NMF provides. ``sparseNMF`` and these are complements, not
   substitutes — the joint model in this package borrows the
   autoencoder bottleneck idea but keeps NMF as the interpretable
   front end.

   Code: https://github.com/bowang-lab/scGPT

A note on the word "sparse"
---------------------------

The NMF literature uses "sparse" for two distinct things and they
sometimes get conflated:

* **Factor sparseness** (Hoyer 2004 line of work, CoGAPS atomic
  priors, MOFA+ spike-and-slab) — the *learned matrices* :math:`W` /
  :math:`H` are encouraged or required to have many zeros.
  Motivation: interpretability — each component lights up only a few
  features. Imposed via constrained projection (Hoyer), atomic priors
  (CoGAPS), or spike-and-slab (MOFA+).
* **Sparse-input I/O** (this package, RcppML, NMF-GPU) — the *input
  matrix* :math:`X` is stored and operated on in a sparse layout
  (CSR / COO / CSC) so the implementation never materializes a dense
  copy. Motivation: scale — typical single-cell or phenome-scale
  matrices are :math:`< 5 \%` non-zero, and densifying them is
  memory-prohibitive.

The two are orthogonal. ``sparseNMF`` is firmly in the second camp.
Adding factor-sparseness constraints à la Hoyer on top of the
sparse-input MU loop is a natural future extension and is not
ruled out by the current design.

Where this package fits
-----------------------

``sparseNMF`` is targeted at the specific pain point that's not
covered cleanly by any of the above. The package was first
stress-tested on **single-cell** count matrices, but the input shape
it's built for — *sparse, non-negative, magnitude-heterogeneous
rows* — is shared by a broader family of biological matrices
(pathway × gene, DGE-signature × gene, GWAS-trait × gene,
disease × gene; see "Beyond single-cell" above). The criteria
where ``sparseNMF`` is the right tool:

1. *Input is sparse and large enough that densification is
   prohibitive* (≥ 50k × 20k @ < 5 % density — modern bio-cohort
   sizes for single-cell, or wide signature-database integrations).
2. *A GPU is available* — typical lab / cloud setup.
3. *Interpretable parts-based factors are wanted* (i.e. NMF's
   non-negativity constraint, not a black-box VAE).
4. *The dominant nuisance is per-row magnitude — library depth for
   single-cell, study power / pathway size / curation depth for
   signature matrices — not batch identity per se.* If batch
   identity is the primary confound, LIGER / iNMF's model-based
   correction is more principled (single-cell); if rows should keep
   their magnitude signal as information, cNMF's "do not normalize"
   stance is more principled. ``sparseNMF``'s default :math:`L_2`
   row-normalize sits between these, and is the right default for
   the signature-matrix shapes above as well.

The implementation keeps the input on the device as a
``torch.sparse`` tensor, runs multiplicative updates in mini-batches
(so the working set stays in VRAM regardless of total dataset size),
defaults to L2 row-normalization for library-depth removal, and adds
an optional joint autoencoder head when a low-dimensional embedding
is what you actually want at the end.

Distribution: https://github.com/bschilder/sparseNMF
