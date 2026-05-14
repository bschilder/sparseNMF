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
covered cleanly by any of the above:

1. *Input is sparse and large enough that densification is
   prohibitive* (≥ 50k × 20k @ < 5 % density, modern bio-cohort
   sizes).
2. *A GPU is available* — typical lab / cloud setup.
3. *Interpretable parts-based factors are wanted* (i.e. NMF's
   non-negativity constraint, not a black-box VAE).
4. *The dominant nuisance is per-row magnitude / library depth, not
   batch identity per se.* If batch identity is the primary
   confound, LIGER / iNMF's model-based correction is more
   principled; if cells should keep their depth signal as
   information, cNMF's "do not normalize" stance is more principled.
   ``sparseNMF``'s default :math:`L_2` row-normalize sits between
   these.

The implementation keeps the input on the device as a
``torch.sparse`` tensor, runs multiplicative updates in mini-batches
(so the working set stays in VRAM regardless of total dataset size),
defaults to L2 row-normalization for library-depth removal, and adds
an optional joint autoencoder head when a low-dimensional embedding
is what you actually want at the end.

Distribution: https://github.com/bschilder/sparseNMF
