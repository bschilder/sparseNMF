Tutorials
=========

Hands-on notebooks shipping with the package. All cells are
pre-executed (outputs and plots are baked into the rendered docs)
so you can read through start-to-finish without setting up a Python
environment first.

.. toctree::
   :maxdepth: 1

   notebooks/01_basic_nmf
   notebooks/02_joint_model
   notebooks/03_sparsity_confound_demo
   notebooks/04_real_pancreas_demo

Each notebook lives at ``docs/notebooks/`` in the source tree —
clone the repo and run them in Jupyter to experiment with parameter
choices on your own data. Re-render outputs with::

   python scripts/build_notebooks.py
