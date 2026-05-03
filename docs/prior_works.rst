Prior works
===========

A survey of NMF implementations and where ``sparseNMF`` sits among
them. The motivation: every existing NMF library either makes you
densify the input (memory-prohibitive at modern bio-data scale) or
runs only on CPU (throughput-prohibitive). This page is the deeper
"why" behind the README's two-paragraph pitch.

----

Classic implementations
-----------------------

scikit-learn — :class:`sklearn.decomposition.NMF`
   Reference implementation; multiplicative updates and coordinate
   descent solvers, well-tested. **Limitation**: input is densified
   to a NumPy ndarray internally for many code paths, so a 100k × 30k
   sparse matrix at 0.5 % density (~15 M non-zeros) explodes to a
   ~12 GB dense float32 array before the first iteration runs.

nimfa — https://github.com/marinkaz/nimfa
   Pure-Python multi-algorithm NMF library (LSNMF, BMF, BD, ICM, …).
   Wide algorithm coverage and great for methods research, but no
   GPU path and again densifies internally for sparse inputs.

scanpy / muon — single-cell ecosystem
   Use NMF (via either scikit-learn or nimfa under the hood) for
   topic-model-style decompositions of cell × gene matrices.
   Inherits the same memory characteristics; works for typical
   single-cell sizes (10k cells × 20k genes) but breaks at the
   100k+ cohorts that motivated this package.

GPU implementations
-------------------

cuML's ``cuml.NMF`` — https://github.com/rapidsai/cuml
   GPU-native NMF inside the RAPIDS stack. Fast on dense input but
   doesn't support ``cupyx.scipy.sparse`` matrices as a first-class
   input — sparse data must be densified to a CuPy array first,
   which is the same memory wall as scikit-learn just with a
   different fail mode.

torchnmf — https://github.com/yoyolicoris/torchnmf
   PyTorch implementation of several NMF variants. Inputs are
   ``torch.Tensor``; sparse-tensor support is partial and limited
   to specific layouts. Closer in spirit to ``sparseNMF`` but the
   training loop densifies for the multiplicative updates.

NMF-GPU — https://github.com/bioinfo-cnio/NMF-GPU
   Earlier CUDA C++ implementation focused on bioinformatics.
   Excellent throughput for the cases it supports but requires
   building from source against a specific CUDA version, and the
   input format is a custom file layout rather than a standard
   sparse matrix.

Adjacent: deep generative factorization
---------------------------------------

scVI / scANVI / poisson-NMF
   Probabilistic generative models that do an NMF-flavored
   decomposition as part of a deeper VAE. Different problem framing
   (likelihood-based, not multiplicative updates) and different
   output (a posterior, not point-estimate W/H). Worth knowing about
   for downstream analysis but not a drop-in NMF replacement.

scGPT / single-cell foundation models
   Treat the cell × gene matrix as a sequence and learn embeddings
   via masked-token prediction. Solves a related problem (per-cell
   embedding) without producing the parts-based interpretable
   factors that NMF provides. ``sparseNMF`` and these are
   complements, not substitutes — the joint model in this package
   borrows the autoencoder bottleneck idea but keeps NMF as the
   interpretable front end.

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

The implementation keeps the input on the device as a
``torch.sparse`` tensor, runs multiplicative updates in mini-batches
(so the working set stays in VRAM regardless of total dataset size),
and adds an optional joint autoencoder head when a low-dimensional
embedding is what you actually want at the end.

Unpublished / internal references
---------------------------------

Several of the design choices were informed by unpublished work
inside Standard Model Bio's phenome stack (``AoU/phenome/``) and the
companion ``protoforge`` proteomics foundation model. Those repos
are not yet public; this package isolates the reusable NMF kernel
from that lineage.
