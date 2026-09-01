Device Management
=================

The framework provides automatic and explicit device management for seamless CPU/GPU workflows.

Device Resolution
-----------------

The :func:`~medical_image.utils.device.resolve_device` function determines the target device using this priority:

1. **Explicit** ``device=`` parameter (highest priority)
2. **Image device** --- inferred from ``image.pixel_data.device``
3. **CPU fallback** (lowest priority)

.. code-block:: python

   from medical_image.utils.device import resolve_device

   # Inferred from image
   device = resolve_device(image)                    # image's device
   device = resolve_device(image, explicit="cuda")   # explicit override

Every processing method and algorithm respects this convention.

Precision Control
-----------------

The :class:`~medical_image.utils.device.Precision` enum controls floating-point precision:

.. code-block:: python

   from medical_image import Precision, set_default_precision, get_default_precision

   # Per-algorithm
   algo = FebdsAlgorithm(precision=Precision.HALF)  # float16

   # Global default
   set_default_precision(Precision.BFLOAT16)
   print(get_default_precision())  # Precision.BFLOAT16

.. list-table::
   :header-rows: 1

   * - Precision
     - dtype
     - Use Case
   * - ``FULL``
     - float32
     - Default; best accuracy
   * - ``HALF``
     - float16
     - 2x memory savings on GPU
   * - ``BFLOAT16``
     - bfloat16
     - Better dynamic range than float16

DeviceContext
-------------

A context manager for GPU memory lifecycle:

.. code-block:: python

   from medical_image import DeviceContext

   with DeviceContext("cuda", verbose=True) as ctx:
       print(ctx.device)
       stats = ctx.memory_stats()
       # ... processing ...
   # torch.cuda.empty_cache() called on exit

The ``@gpu_safe`` Decorator
----------------------------

Catches ``torch.cuda.OutOfMemoryError`` and retries the decorated function on CPU:

.. code-block:: python

   from medical_image import gpu_safe

   @gpu_safe
   def my_processing(image, output, device=None):
       # If CUDA runs out of memory, automatically retried on CPU
       ...

Memory Estimation
-----------------

Check whether an image will fit on the GPU before loading:

.. code-block:: python

   from medical_image import estimate_image_bytes, check_gpu_budget

   bytes_needed = estimate_image_bytes(3328, 2560)
   fits = check_gpu_budget(bytes_needed)