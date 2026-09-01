Core Concepts
=============

Understanding these concepts will help you use the framework effectively.

Everything is a Tensor
----------------------

All pixel data in the framework is stored as a :class:`torch.Tensor`. Whether you load a DICOM file, a PNG image, or create an in-memory array, the result is a tensor that can be:

- Moved to GPU with ``image.to("cuda")``
- Processed with PyTorch operations
- Converted to numpy with ``image.pixel_data.numpy()``

This design unifies CPU and GPU workflows under a single interface.

The Clone-Process Pattern
-------------------------

Processing operations read from an **input** image and write to an **output** image. The standard pattern is:

.. code-block:: python

   output = image.clone()           # lightweight copy (clones tensor, not DICOM data)
   some_operation(image, output)    # reads image, writes output
   result = output.pixel_data       # processed result

This keeps the original image unchanged and avoids accidental in-place mutation.

Algorithms vs Processing
------------------------

The framework distinguishes between two levels of abstraction:

**Processing** (``medical_image.process``):
   Stateless, single-step operations --- a Gaussian filter, a threshold, a morphological closing. These are static methods on utility classes.

**Algorithms** (``medical_image.algorithms``):
   Stateful, multi-step pipelines that compose processing operations. Algorithms inherit from :class:`~medical_image.algorithms.algorithm.Algorithm`, store configuration, and are callable objects.

.. code-block:: python

   # Processing: one step
   Filters.gaussian_filter(image, output, sigma=2.0)

   # Algorithm: multi-step pipeline
   algo = FebdsAlgorithm(method="dog")
   algo(image=image, output=output)

Annotations
-----------

Algorithms that detect structures (e.g., lesions, microcalcifications) attach :class:`~medical_image.data.annotation.Annotation` objects to the output image. Each annotation stores:

- **Shape** --- rectangle, ellipse, or polygon
- **Coordinates** --- geometry-specific coordinate list
- **Label** --- e.g., ``"microcalcification"``
- **Metadata** --- confidence, area, bounding box, etc.

.. code-block:: python

   for ann in output.annotations:
       print(ann.label, ann.shape, ann.metadata["confidence"])
       bbox = ann.get_bounding_box()  # [x_min, y_min, x_max, y_max]

Device Inference
----------------

Most operations accept an optional ``device`` parameter. When omitted, the device is **inferred from the input image**:

.. code-block:: python

   image.to("cuda")
   Filters.gaussian_filter(image, output, sigma=2.0)  # runs on CUDA automatically

See :doc:`/concepts/device_management` for details.