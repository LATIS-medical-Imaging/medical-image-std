Working with Images
===================

The :class:`~medical_image.data.image.Image` abstract class provides a unified interface for all image formats. Concrete implementations handle format-specific I/O.

Image Types
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Class
     - Format
     - Use Case
   * - :class:`~medical_image.data.dicom_image.DicomImage`
     - DICOM (``.dcm``)
     - Clinical mammography, radiology
   * - :class:`~medical_image.data.png_image.PNGImage`
     - PNG (``.png``)
     - Exported images, masks
   * - :class:`~medical_image.data.in_memory_image.InMemoryImage`
     - NumPy / Tensor
     - Intermediate results, testing

Loading Images
--------------

All images follow the **lazy loading** pattern:

.. code-block:: python

   from medical_image import DicomImage

   # Step 1: Create (no I/O)
   image = DicomImage("mammogram.dcm")

   # Step 2: Load pixel data
   image.load()

   # Now pixel_data is available
   print(image.pixel_data.shape)   # e.g., torch.Size([3328, 2560])
   print(image.width, image.height)

Factory Methods
~~~~~~~~~~~~~~~

The :class:`~medical_image.data.image.Image` class provides factory constructors:

.. code-block:: python

   from medical_image import Image, DicomImage

   # From file path (auto-detects format)
   img = DicomImage.from_file("scan.dcm")

   # From numpy array
   import numpy as np
   arr = np.random.rand(256, 256).astype(np.float32)
   img = DicomImage.from_array(arr)

   # Empty image (zeros)
   blank = Image.empty(512, 512)

   # Clone an existing image
   copy = img.clone()

Cloning
-------

``clone()`` creates a lightweight copy: it clones the pixel data tensor but **not** heavy backing objects (like the pydicom Dataset). This makes it efficient for creating output images:

.. code-block:: python

   output = image.clone()
   # output.pixel_data is a new tensor
   # image.pixel_data is unchanged

Device Management
-----------------

Move images between CPU and GPU:

.. code-block:: python

   image.to("cuda")          # move to GPU
   image.to("cpu")           # move back
   image.pin_memory()        # page-lock for async GPU transfer

   print(image.device)       # torch.device('cuda:0')

Annotations
-----------

Images can carry geometric annotations:

.. code-block:: python

   from medical_image import Annotation, GeometryType

   ann = Annotation(
       shape=GeometryType.POLYGON,
       coordinates=[(10, 20), (30, 20), (30, 40), (10, 40)],
       label="mass",
       metadata={"confidence": 0.92}
   )

   image.add_annotation(ann)
   image.remove_annotation(0)

Serialization
~~~~~~~~~~~~~

Images can be serialized to JSON (metadata + annotations, not pixel data):

.. code-block:: python

   image.to_json("image_meta.json")

   # Restore
   restored = Image.from_json("image_meta.json")

Displaying Image Info
---------------------

.. code-block:: python

   image.display_info()
   # Logs: path, dimensions, device, pixel range, annotation count