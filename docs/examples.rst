Examples
========

Runnable scripts under ``examples/`` in the repo:

``examples/basic_usage.py``
   Standalone NMF on the bundled synthetic data. Plots
   reconstruction error vs iteration so you can sanity-check
   convergence.

``examples/joint_model.py``
   End-to-end joint NMF + autoencoder, producing a 2-D embedding
   suitable for ``matplotlib.pyplot.scatter``. Illustrates how to
   pick ``nmf_components`` and ``latent_dim`` for typical bio data
   sizes.

Each example is self-contained — clone the repo, run
``python examples/<name>.py``, no further setup beyond installing
``sparse-nmf[viz]``.
