GPU Acceleration
================

The framework provides transparent GPU support. Most operations work on both CPU and CUDA devices without code changes.

Automatic Device Inference
--------------------------

When ``device=None`` (the default), the device is **inferred from the input image**:

.. code-block:: python

   from medical_image import Filters

   image.to("cuda")
   output = image.clone()
   Filters.gaussian_filter(image, output, sigma=2.0)  # runs on CUDA automatically

The resolution priority is:

1. Explicit ``device=`` parameter (highest)
2. Input image device (``image.pixel_data.device``)
3. CPU fallback (lowest)

DeviceContext Manager
---------------------

Manage GPU memory with automatic cleanup:

.. code-block:: python

   from medical_image import DeviceContext

   with DeviceContext("cuda", verbose=True) as ctx:
       print(ctx.device)          # torch.device('cuda:0')
       stats = ctx.memory_stats() # {'allocated': ..., 'reserved': ...}
       # ... processing ...
   # torch.cuda.empty_cache() called automatically on exit

Mixed Precision
---------------

Reduce memory usage and increase throughput with half-precision:

.. code-block:: python

   from medical_image import Precision, set_default_precision

   # Per-algorithm
   algo = FebdsAlgorithm(device="cuda", precision=Precision.HALF)
   algo(image, output)

   # Or globally
   set_default_precision(Precision.BFLOAT16)

Available precisions: ``FULL`` (float32), ``HALF`` (float16), ``BFLOAT16``.

OOM Fallback
------------

The ``@gpu_safe`` decorator catches CUDA out-of-memory errors and retries on CPU:

.. code-block:: python

   from medical_image import gpu_safe

   @gpu_safe
   def process(image, output, device=None):
       # If CUDA OOM occurs, automatically retried on CPU
       Filters.gaussian_filter(image, output, sigma=2.0, device=device)

Batch Processing
----------------

Process multiple images efficiently:

.. code-block:: python

   # Per-algorithm batching
   algo.apply_batch(images, outputs)

   # Batched filtering
   batch = torch.randn(8, 1, 256, 256, device="cuda")
   result = Filters.gaussian_filter_batch(batch, sigma=2.0)

Async GPU Pipeline
------------------

Overlap disk I/O with GPU compute using CUDA streams:

.. code-block:: python

   from medical_image import AsyncGPUPipeline

   pipeline = AsyncGPUPipeline()
   results = pipeline.process_images(images, algorithm)

Multi-GPU
---------

Distribute work across multiple GPUs:

.. code-block:: python

   from medical_image import MultiGPUAlgorithm

   multi = MultiGPUAlgorithm(
       algorithm_class=FebdsAlgorithm,
       gpu_ids=[0, 1, 2],
       method="dog",
   )
   multi.apply_batch(images, outputs)

Memory Management Tips
----------------------

- Use ``DeviceContext`` to ensure cache is cleared after processing blocks.
- Use ``image.pin_memory()`` before ``image.to("cuda")`` for faster transfers.
- Use ``Precision.HALF`` for large images when float16 accuracy is acceptable.
- Use ``@gpu_safe`` to handle unpredictable OOM without crashing.