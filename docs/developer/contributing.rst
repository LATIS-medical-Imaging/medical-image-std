Contributing
============

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/LATIS-medical-Imaging/medical-image-std.git
   cd medical-image-std
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

Or with ``uv``:

.. code-block:: bash

   uv pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   pytest medical_image/tests/ -v

The test suite includes 136+ tests covering:

- Image loading and format handling
- Processing operations (filters, thresholds, morphology)
- Algorithm correctness (FEBDS, K-Means, FCM, PFCM, Deep Segmentation)
- Patch grid creation and reconstruction
- Dataset loading and sample format
- Annotation serialization
- Mammography preprocessing

Code Style
----------

The project uses **Black** for formatting and **Ruff** for linting:

.. code-block:: bash

   black medical_image/
   ruff check medical_image/

CI runs Black and pytest on Python 3.11 and 3.12.

Pull Request Workflow
---------------------

1. Fork the repository.
2. Create a feature branch from ``master``.
3. Write tests for new functionality.
4. Ensure ``pytest`` and ``black --check`` pass.
5. Open a pull request with a clear description.

Building Documentation
----------------------

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` to preview.