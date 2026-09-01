Medical Image Standard
======================

.. image:: _static/logo.png
   :alt: Medical Image Standard
   :width: 300px
   :align: center

.. raw:: html

   <p class="hero-tagline" style="text-align: center;">
   A standardized Python framework for medical image processing with GPU acceleration, built around PyTorch tensors.
   </p>

   <div class="quick-nav" style="justify-content: center;">
      <a href="getting_started/quickstart.html" class="primary">Get Started</a>
      <a href="api/index.html" class="secondary">API Reference</a>
      <a href="https://github.com/LATIS-medical-Imaging/medical-image-std" class="secondary">GitHub</a>
   </div>

   <hr class="section-divider">

Install
-------

.. code-block:: bash

   pip install medical-image-std

Quick Example
-------------

.. code-block:: python

   from medical_image import DicomImage, FebdsAlgorithm

   # Load a mammogram
   image = DicomImage("mammogram.dcm")
   image.load()

   # Run microcalcification detection
   output = image.clone()
   algo = FebdsAlgorithm(method="dog", device="cuda")
   algo(image=image, output=output)

   # Extract binary mask
   mask = output.pixel_data.cpu().numpy()

Key Features
------------

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <strong>Image Abstractions</strong>
       <p>Unified API across DICOM, PNG, and in-memory images with lazy loading and automatic format handling.</p>
     </div>
     <div class="feature-card">
       <strong>GPU Acceleration</strong>
       <p>Transparent device management with automatic inference, mixed precision, OOM fallback, and multi-GPU support.</p>
     </div>
     <div class="feature-card">
       <strong>Extensible Algorithms</strong>
       <p>Composable algorithm framework with 11 built-in implementations for segmentation, filtering, and enhancement.</p>
     </div>
     <div class="feature-card">
       <strong>Dataset Integration</strong>
       <p>PyTorch-compatible dataset classes for INbreast and CBIS-DDSM with lazy loading and on-the-fly transforms.</p>
     </div>
   </div>

Architecture Overview
---------------------

The framework is built around five core layers:

1. **Data Layer** --- Abstract :class:`~medical_image.data.image.Image` class with concrete implementations for DICOM, PNG, and in-memory formats. All pixel data stored as :class:`torch.Tensor`.

2. **Processing Layer** --- Stateless operations (filters, thresholds, morphology, metrics) applied via static methods on loaded images.

3. **Algorithm Layer** --- :class:`~medical_image.algorithms.algorithm.Algorithm` base class using the Template Method pattern. Algorithms compose processing operations into pipelines.

4. **Dataset Layer** --- PyTorch-compatible :class:`~medical_image.datasets.base_dataset.BaseDataset` with lazy sample loading and standard dict output.

5. **Utilities** --- Device management, precision control, GPU memory tools, and export helpers.

.. code-block:: text

     Image (DICOM / PNG / InMemory)
         |
         v
     Processing (Filters, Threshold, Morphology)
         |
         v
     Algorithm (FEBDS, FCM, DeepSeg, ...)
         |
         v
     Output (binary mask + annotations)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/images
   user_guide/processing
   user_guide/algorithms
   user_guide/datasets
   user_guide/gpu
   user_guide/patches

.. toctree::
   :maxdepth: 2
   :caption: Concepts
   :hidden:

   concepts/architecture
   concepts/lazy_loading
   concepts/device_management

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer/contributing
   developer/extending