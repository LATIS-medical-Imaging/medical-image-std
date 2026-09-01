Quick Start
===========

This guide walks through the core workflow: load an image, process it, and extract results.

Loading a Medical Image
-----------------------

All images follow the **lazy loading** pattern --- the constructor stores metadata, and ``.load()`` reads the pixel data into a PyTorch tensor.

.. code-block:: python

   from medical_image import DicomImage

   image = DicomImage("mammogram.dcm")
   image.load()

   print(image.width, image.height)        # Image dimensions
   print(image.pixel_data.shape)            # torch.Tensor shape
   print(image.pixel_data.device)           # cpu or cuda

You can also work with PNG images or raw arrays:

.. code-block:: python

   from medical_image import PNGImage, InMemoryImage
   import numpy as np

   # From PNG file
   png = PNGImage("scan.png")
   png.load()

   # From numpy array
   arr = np.random.rand(256, 256).astype(np.float32)
   mem = InMemoryImage(array=arr)

Applying Processing Operations
-------------------------------

Processing operations are **stateless static methods**. They read from an input image and write to an output image:

.. code-block:: python

   from medical_image import Filters, Threshold

   # Always clone before processing
   output = image.clone()

   # Gaussian blur
   Filters.gaussian_filter(image, output, sigma=2.0)

   # Otsu thresholding
   binary = output.clone()
   Threshold.otsu_threshold(output, binary)

   # Result is a binary mask tensor
   mask = binary.pixel_data  # torch.Tensor with values 0.0 and 1.0

Running an Algorithm
--------------------

Algorithms encapsulate multi-step pipelines. They follow the same ``(image, output)`` interface:

.. code-block:: python

   from medical_image import FebdsAlgorithm

   algo = FebdsAlgorithm(method="dog", device="cpu")
   output = image.clone()
   algo(image=image, output=output)

   # Extract numpy mask
   mask_np = output.pixel_data.detach().cpu().numpy()

Using Deep Learning Models
--------------------------

Download and run pretrained segmentation models directly:

.. code-block:: python

   from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm

   # Discover available models
   models = DeepSegmentationAlgorithm.list_available_models()
   for m in models:
       print(f"{m['name']} ({m['architecture']}, patch={m['patch_size']})")

   # Download and load a model
   algo = DeepSegmentationAlgorithm.from_pretrained(
       "unetpp_bce_dice_32_inbreast", device="cuda"
   )

   # Run inference
   output = image.clone()
   algo(image=image, output=output)

   # Binary mask + per-lesion annotations
   mask_np = output.pixel_data.cpu().numpy()
   for ann in output.annotations:
       print(ann.label, ann.metadata["confidence"], ann.metadata["area"])

Working with Patches
--------------------

Large medical images can be split into a grid of patches:

.. code-block:: python

   from medical_image import PatchGrid

   grid = PatchGrid(image, patch_size=(128, 128))
   print(f"{len(grid.patches)} patches")

   # Process each patch
   for patch in grid.patches:
       patch_img = patch.to_image()
       # ... process patch_img ...

   # Reconstruct full image
   full = grid.reconstruct()  # torch.Tensor

GPU Acceleration
----------------

Move images to GPU and process with mixed precision:

.. code-block:: python

   from medical_image import DeviceContext, Precision

   image.to("cuda")

   with DeviceContext("cuda") as ctx:
       algo = FebdsAlgorithm(method="dog", device="cuda")
       algo.precision = Precision.HALF
       output = image.clone()
       algo(image=image, output=output)

See :doc:`/user_guide/gpu` for OOM fallback, multi-GPU, and async pipeline details.