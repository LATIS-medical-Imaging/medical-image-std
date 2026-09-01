Data
====

Image abstractions, patches, annotations, and regions of interest.

Image (Abstract Base)
---------------------

.. autoclass:: medical_image.data.image.Image
   :members:
   :exclude-members: _post_load, _cached_device

.. autofunction:: medical_image.data.image.requires_loaded

DicomImage
----------

.. autoclass:: medical_image.data.dicom_image.DicomImage
   :members:
   :show-inheritance:

PNGImage
--------

.. autoclass:: medical_image.data.png_image.PNGImage
   :members:
   :show-inheritance:

InMemoryImage
-------------

.. autoclass:: medical_image.data.in_memory_image.InMemoryImage
   :members:
   :show-inheritance:

PatchGrid
---------

.. autoclass:: medical_image.data.patch.PatchGrid
   :members:

Patch
-----

.. autoclass:: medical_image.data.patch.Patch
   :members:

RegionOfInterest
----------------

.. autoclass:: medical_image.data.region_of_interest.RegionOfInterest
   :members:

Annotation
----------

.. autoclass:: medical_image.data.annotation.Annotation
   :members:

GeometryType
~~~~~~~~~~~~~

.. autoclass:: medical_image.data.annotation.GeometryType
   :members:
   :undoc-members: