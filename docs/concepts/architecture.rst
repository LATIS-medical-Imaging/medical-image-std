Architecture
============

Medical Image Standard is designed around five core principles:

1. **Abstraction-first** --- Unified interfaces hide format-specific complexity.
2. **Lazy loading** --- No I/O until explicitly requested.
3. **Stateless processing** --- Processing operations are pure functions on tensors.
4. **Algorithm composition** --- Pipelines built by composing simple operations.
5. **Automatic device inference** --- GPU usage requires no special handling.

Package Structure
-----------------

.. code-block:: text

   medical_image/
   ├── data/           # Image abstractions, patches, annotations
   ├── process/        # Stateless operations (filters, threshold, morphology)
   ├── algorithms/     # Multi-step pipelines (FEBDS, FCM, DeepSeg, ...)
   ├── datasets/       # PyTorch Dataset subclasses (INbreast, CBIS-DDSM)
   └── utils/          # Device management, export, logging, errors

Data Flow
---------

A typical workflow proceeds through these layers:

.. code-block:: text

   Input                      Processing                    Output
   ─────                      ──────────                    ──────
   DicomImage.load()    ──>   Filters / Threshold     ──>   output.pixel_data
        |                          |                             |
        v                          v                             v
   image.pixel_data         Algorithm.apply()           output.annotations
   (torch.Tensor)           (composed pipeline)         (List[Annotation])

Design Patterns
---------------

Strategy Pattern (Algorithms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~medical_image.algorithms.algorithm.Algorithm` base class defines the interface. Concrete algorithms are interchangeable strategies:

.. code-block:: python

   # Any Algorithm can be used interchangeably
   algo = FebdsAlgorithm(method="dog")
   algo = KMeansAlgorithm(k=3)
   algo = DeepSegmentationAlgorithm.from_pretrained("unetpp_bce_dice_32_inbreast")

   # Same calling convention
   algo(image=image, output=output)

Template Method (Algorithm.__call__)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class ``__call__`` wraps ``apply()`` with optional mixed-precision autocast. Subclasses only implement ``apply()``:

.. code-block:: python

   class Algorithm(ABC):
       def __call__(self, image, output):
           if self.precision != Precision.FULL and self.device != "cpu":
               with torch.cuda.amp.autocast(dtype=self.precision.value):
                   self.apply(image, output)
           else:
               self.apply(image, output)
           return output

Lambda Composition (FEBDS)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex algorithms define processing steps as lambda functions in ``__init__``, executed sequentially in ``apply()``. This allows swapping individual steps without subclassing:

.. code-block:: python

   class FebdsAlgorithm(Algorithm):
       def __init__(self, method="dog"):
           self.dog = lambda img, out: Filters.difference_of_gaussian(
               image=img, output=out, low_sigma=1.7, high_sigma=2.0
           )
           self.otsu = lambda img, out: Threshold.otsu_threshold(
               image=img, output=out
           )

Adapter Pattern (Image Subclasses)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each image format (DICOM, PNG, in-memory) adapts its native I/O library to the common :class:`~medical_image.data.image.Image` interface. Users interact with the same API regardless of the underlying format.

Factory Methods
~~~~~~~~~~~~~~~

The ``Image`` class provides factory constructors: ``from_file()``, ``from_array()``, ``from_image()``, ``empty()``. These select the right subclass or create appropriate instances without exposing construction details.

Error Handling
--------------

The framework defines a custom exception hierarchy rooted at ``AppError``:

- ``FileNotFoundAppError`` --- file path does not exist
- ``InvalidPixelDataError`` --- pixel data is None or invalid
- ``UnsupportedFileTypeError`` --- wrong file extension for image type
- ``DicomDataNotLoadedError`` --- operation on unloaded DICOM
- ``EmptyDatasetError`` --- dataset has no samples

The ``@requires_loaded`` decorator on processing methods raises ``DicomDataNotLoadedError`` automatically when ``pixel_data`` is ``None``.