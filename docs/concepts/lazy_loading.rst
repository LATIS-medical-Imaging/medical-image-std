Lazy Loading
============

Lazy loading is a central design principle: image objects are **lightweight handles** until you explicitly call ``.load()``.

Why Lazy Loading?
-----------------

Medical imaging workflows often reference thousands of images but only process a subset. Lazy loading avoids:

- Loading gigabytes of pixel data into memory upfront
- Blocking the main thread on I/O during dataset traversal
- Wasting GPU memory on images that may never be processed

How It Works
------------

.. code-block:: python

   from medical_image import DicomImage

   # Step 1: Create — stores path only, no I/O
   image = DicomImage("mammogram.dcm")
   assert image.pixel_data is None  # nothing loaded yet

   # Step 2: Load — reads file, populates pixel_data
   image.load()
   assert image.pixel_data is not None  # torch.Tensor ready

   # Step 3: Use
   print(image.pixel_data.shape)  # e.g., (3328, 2560)

Deferred Device Migration
--------------------------

You can specify the target device **before** loading. The tensor will be placed on the correct device after ``load()`` completes:

.. code-block:: python

   image = DicomImage("mammogram.dcm")
   image.to("cuda")    # caches target device
   image.load()        # pixel_data goes directly to GPU

Lightweight Cloning
-------------------

``clone()`` creates a copy of the pixel data tensor but **does not** deep-copy heavy backing objects (pydicom Dataset, PIL Image). This makes it efficient for creating output images:

.. code-block:: python

   output = image.clone()
   # output.pixel_data — new tensor (independent of original)
   # image.dicom_data  — NOT copied (memory efficient)

The ``@requires_loaded`` Decorator
-----------------------------------

Processing operations are decorated with ``@requires_loaded``, which validates that all ``Image`` arguments have non-None ``pixel_data`` before the operation runs. If an image hasn't been loaded, it raises ``DicomDataNotLoadedError``.

Datasets and Lazy Loading
--------------------------

Dataset classes (INbreast, CBIS-DDSM) inherit this pattern:

- ``__init__`` scans the directory structure and builds a sample list (metadata only).
- ``__getitem__`` loads a single sample on demand.
- ``__len__`` returns the count without loading any images.

This integrates with PyTorch's ``DataLoader`` for efficient batched loading with multiple workers.