Contributing
============

Local setup
-----------

.. code-block:: bash

   git clone https://github.com/bschilder/sparseNMF.git
   cd sparseNMF
   pip install -e ".[dev]"

   pytest                              # tests + coverage
   ruff check . && ruff format --check .
   sphinx-build docs docs/_build/html  # build the docs site locally

CI runs lint + tests on every PR. PRs that drop coverage by more
than 1% against ``main`` will be flagged by Codecov.

Filing an issue
---------------

Open issues at https://github.com/bschilder/sparseNMF/issues.
Please include:

* A minimal repro — sparse matrix shape, density, dtype.
* The exact command + traceback.
* PyTorch version and CUDA capability (if applicable).

Releases
--------

Tag-triggered:

.. code-block:: bash

   git tag v0.2.0
   git push --tags

The ``release.yml`` workflow builds the wheel + sdist, creates a
GitHub Release with auto-generated notes, and the ``docker.yml``
workflow simultaneously publishes a tagged GHCR image.
