Examples
========

Runnable end-to-end scripts under ``examples/`` in the repo, plus
their companion Jupyter notebooks under ``docs/notebooks/`` (rendered
on this site with outputs baked in). Clone the repo, ``pip install
"sparse-nmf[viz]"``, and run any script with ``python examples/<name>.py``.

Quickstart examples
-------------------

``examples/basic_usage.py``
   Standalone :class:`sparse_nmf.SparseNMF` on the bundled synthetic
   data. Prints reconstruction error per iteration as a sanity check
   on convergence. Companion notebook:
   :doc:`Tutorial 1 — Standalone sparse NMF <notebooks/01_basic_nmf>`.

   * Script: `examples/basic_usage.py
     <https://github.com/bschilder/sparseNMF/blob/main/examples/basic_usage.py>`__

``examples/joint_model.py``
   End-to-end joint NMF + autoencoder producing a 2-D embedding
   suitable for ``matplotlib.pyplot.scatter``. Illustrates how to pick
   ``nmf_components`` and ``latent_dim`` for typical bio-data sizes.
   Companion notebook:
   :doc:`Tutorial 2 — Joint NMF + autoencoder <notebooks/02_joint_model>`.

   * Script: `examples/joint_model.py
     <https://github.com/bschilder/sparseNMF/blob/main/examples/joint_model.py>`__

Method-comparison demos
-----------------------

These compare ``sparseNMF`` against PCA and vanilla NMF on the same
input, all projected to 2-D via the same UMAP step — so any
difference in the resulting embedding reflects the *factorization*,
not the projector.

``examples/sparsity_confound_demo.py``
   Synthetic *worst-case* sparsity confound: three biological groups,
   each split into two batches with ~10× different non-zero gene
   counts. Shows PCA and vanilla NMF locking onto the per-cell
   magnitude axis instead of biology, while ``sparseNMF``'s default
   ``normalize_inputs=True`` recovers the three groups cleanly.
   Companion notebook:
   :doc:`Tutorial 3 — The sparsity confound (synthetic) <notebooks/03_sparsity_confound_demo>`.

   * Script: `examples/sparsity_confound_demo.py
     <https://github.com/bschilder/sparseNMF/blob/main/examples/sparsity_confound_demo.py>`__

``examples/real_pancreas_demo.py``
   The same comparison on real cross-protocol scRNA-seq: the scIB
   human pancreas benchmark (Luecken *et al.* 2022, 9 protocols,
   library depth varies ~300×). Reports the **depth-R²** metric that
   directly quantifies how much of the library-depth axis each
   embedding still encodes. Auto-fetches the dataset from figshare
   (~301 MB) into ``~/.cache/sparse-nmf/``. Companion notebook:
   :doc:`Tutorial 4 — Real cross-protocol scRNA-seq <notebooks/04_real_pancreas_demo>`.

   * Script: `examples/real_pancreas_demo.py
     <https://github.com/bschilder/sparseNMF/blob/main/examples/real_pancreas_demo.py>`__

The notebook builders (``scripts/build_notebooks.py``) are the
source of truth for the rendered tutorials — re-run them after
editing a script if you want the notebook outputs to stay in sync.
