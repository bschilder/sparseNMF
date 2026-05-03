API reference
=============

Public surface re-exported from :mod:`sparse_nmf`.

Standalone NMF
--------------

.. autoclass:: sparse_nmf.SparseNMF
   :members:
   :show-inheritance:

.. autofunction:: sparse_nmf.sparse_nmf

.. autofunction:: sparse_nmf.train_sparse_nmf

Joint model
-----------

.. autoclass:: sparse_nmf.SparseNMF_Autoencoder
   :members:
   :show-inheritance:

.. autofunction:: sparse_nmf.train_joint_model

.. autofunction:: sparse_nmf.compute_joint_loss

Attention analysis
------------------

.. autofunction:: sparse_nmf.extract_attention_weights

.. autofunction:: sparse_nmf.extract_and_aggregate_attention

.. autofunction:: sparse_nmf.trace_attention_to_genes

.. autofunction:: sparse_nmf.compute_attention_correlation

Visualization
-------------

.. autofunction:: sparse_nmf.plot_nmf_factor_distributions

Sample data
-----------

.. autofunction:: sparse_nmf.data.generate_synthetic_sparse

.. autofunction:: sparse_nmf.data.load_synthetic_sparse
