Installation
============

Requirements
------------

- Python 3.11 or later
- PyTorch (CPU or CUDA)

Basic Install
-------------

.. code-block:: bash

   pip install medical-image-std

With ``uv`` (recommended for faster resolution):

.. code-block:: bash

   uv pip install medical-image-std

Optional Dependencies
---------------------

Install extras for development, GPU, or Qt visualization:

.. code-block:: bash

   # Development tools (pytest, black, ruff, mypy)
   pip install "medical-image-std[dev]"

   # GPU support (explicit torch + torchvision)
   pip install "medical-image-std[gpu]"

   # Documentation building
   pip install "medical-image-std[docs]"

   # Everything
   pip install "medical-image-std[all]"

From Source
-----------

.. code-block:: bash

   git clone https://github.com/LATIS-medical-Imaging/medical-image-std.git
   cd medical-image-std
   pip install -e ".[dev]"

Verify Installation
-------------------

.. code-block:: python

   import medical_image
   from medical_image import DicomImage, Filters, FebdsAlgorithm
   print("medical-image-std installed successfully")