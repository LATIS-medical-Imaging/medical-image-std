Utilities
=========

Device management, precision control, export, and helper functions.

Device Management
-----------------

.. autofunction:: medical_image.utils.device.resolve_device

.. autoclass:: medical_image.utils.device.Precision
   :members:
   :undoc-members:

.. autofunction:: medical_image.utils.device.set_default_precision

.. autofunction:: medical_image.utils.device.get_default_precision

.. autofunction:: medical_image.utils.device.get_dtype

.. autofunction:: medical_image.utils.device.estimate_image_bytes

.. autofunction:: medical_image.utils.device.check_gpu_budget

.. autoclass:: medical_image.utils.device.DeviceContext
   :members:

.. autofunction:: medical_image.utils.device.gpu_safe

.. autoclass:: medical_image.utils.device.AsyncGPUPipeline
   :members:

.. autoclass:: medical_image.utils.device.MultiGPUAlgorithm
   :members:

Image Utilities
---------------

.. autoclass:: medical_image.utils.image_utils.TensorConverter
   :members:

.. autoclass:: medical_image.utils.image_utils.ImageExporter
   :members:

.. autoclass:: medical_image.utils.image_utils.ImageVisualizer
   :members:

.. autoclass:: medical_image.utils.image_utils.MathematicalOperations
   :members:

Logging
-------

.. autofunction:: medical_image.utils.logging.configure_logging