Quickstart
==========

A 2-minute tour. Both modes use the same input — a SciPy
``csr_matrix`` — and produce dense outputs ready for whatever you'd
do next (cluster, plot, feed into a downstream model, …).

Standalone NMF
--------------

.. code-block:: python

   from scipy.sparse import csr_matrix
   from sparse_nmf import SparseNMF

   X = csr_matrix(...)                       # (n_samples, n_features), sparse, non-negative

   nmf = SparseNMF(n_components=64, max_iter=200, device="cuda")
   W = nmf.fit_transform(X)                  # (n_samples, 64) dense — the per-sample code
   H = nmf.components_                       # (64, n_features) dense — the per-feature loadings

``W`` and ``H`` are both non-negative and approximate ``W @ H ≈ X``.
The reconstruction error is exposed on the model object after a fit.

Joint NMF + autoencoder
-----------------------

.. code-block:: python

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

``z`` is an ``(n_samples, latent_dim)`` embedding. With
``latent_dim=2``, drop it directly into a scatterplot. With
``latent_dim=64-256``, feed it into a downstream classifier or
retrieval index.

Sample data
-----------

For a runnable end-to-end check:

.. code-block:: python

   from sparse_nmf import SparseNMF
   from sparse_nmf.data import generate_synthetic_sparse

   X = generate_synthetic_sparse(
       n_samples=2000, n_features=5000, n_components=16, seed=42,
   )
   W = SparseNMF(n_components=16, max_iter=100, device="cuda").fit_transform(X)
   print(W.shape)  # (2000, 16)

The synthetic generator builds a controllable rank-K plus noise
matrix so you can verify NMF is recovering the planted structure.
