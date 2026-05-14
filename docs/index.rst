sparseNMF
=========

.. image:: _static/logo.svg
   :alt: sparseNMF logo
   :align: center
   :width: 480

GPU-accelerated sparse non-negative matrix factorization with PyTorch.

----

**sparseNMF** factorizes very-large sparse non-negative matrices on
the GPU without materializing a dense copy. Designed for biomedical
data: gene-association counts, phenotype matrices, single-cell
expression — anything that's both *sparse* and *too big to fit
densely in VRAM*.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install
   quickstart
   tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   prior_works
   benchmark
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   license

Two flavors
-----------

**Standalone** :class:`sparse_nmf.SparseNMF`
   Drop-in replacement for ``sklearn.decomposition.NMF`` that runs on
   GPU and never densifies the input. Use it when you want ``W`` and
   ``H`` matrices directly.

**Joint** :func:`sparse_nmf.train_joint_model`
   End-to-end training of an NMF + autoencoder pipeline. Use it when
   you want a low-dimensional embedding (e.g., 2-D for plotting,
   192-D for downstream retrieval) rather than the full ``W`` matrix.

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
