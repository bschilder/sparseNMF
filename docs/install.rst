Installation
============

From PyPI (recommended)
-----------------------

.. code-block:: bash

   pip install sparse-nmf

With the visualization extras (``matplotlib``, ``seaborn``):

.. code-block:: bash

   pip install "sparse-nmf[viz]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/bschilder/sparseNMF.git
   cd sparseNMF
   pip install -e ".[dev]"

GPU support
-----------

``sparseNMF`` requires PyTorch. To run on a GPU, install a CUDA-enabled
build of ``torch`` *before* installing this package, otherwise pip will
pull the CPU-only wheel from PyPI.

.. code-block:: bash

   pip install --index-url https://download.pytorch.org/whl/cu124 torch
   pip install sparse-nmf

CPU-only is fully supported (correctness is unchanged; throughput
drops by ~10-50× depending on shape).

Container
---------

A pinned CUDA image is published to GHCR on every release:

.. code-block:: bash

   docker pull ghcr.io/bschilder/sparsenmf:latest
   docker run --gpus all --rm -it ghcr.io/bschilder/sparsenmf:latest python
