Processing Operations
=====================

The ``medical_image.process`` module provides stateless image processing operations. All methods are **static** and decorated with ``@requires_loaded`` --- they raise an error if the input image has not been loaded.

General Pattern
---------------

Every processing method follows the same signature:

.. code-block:: python

   SomeProcessor.operation(image, output, <params>, device=None)

- ``image`` --- input :class:`~medical_image.data.image.Image` (read-only)
- ``output`` --- output :class:`~medical_image.data.image.Image` (result written here)
- ``device`` --- optional; inferred from ``image`` if omitted

Filters
-------

Spatial and frequency domain filters:

.. code-block:: python

   from medical_image import Filters

   output = image.clone()

   # Gaussian blur
   Filters.gaussian_filter(image, output, sigma=2.0)

   # Median filter (denoising)
   Filters.median_filter(image, output, size=5)

   # Difference of Gaussians (band-pass)
   Filters.difference_of_gaussian(image, output, low_sigma=1.7, high_sigma=2.0)

   # Laplacian of Gaussian (edge detection)
   Filters.laplacian_of_gaussian(image, output, sigma=1.5)

   # Gamma correction
   Filters.gamma_correction(image, output, gamma=0.5)

   # Contrast adjustment
   Filters.contrast_adjust(image, output, contrast=1.5, brightness=0.1)

Thresholding
------------

Global and adaptive thresholding:

.. code-block:: python

   from medical_image import Threshold

   output = image.clone()

   # Otsu (global, automatic threshold)
   Threshold.otsu_threshold(image, output)

   # Sauvola (adaptive local threshold)
   Threshold.sauvola_threshold(image, output, window_size=15, k=0.2)

Morphological Operations
-------------------------

Binary and grayscale morphology:

.. code-block:: python

   from medical_image import MorphologyOperations

   output = image.clone()

   # Closing (fills small gaps)
   MorphologyOperations.morphology_closing(image, output, kernel_size=7)

   # Erosion / Dilation (disk structuring element)
   MorphologyOperations.erosion(image, output, radius=2)
   MorphologyOperations.dilation(image, output, radius=3)

   # White top-hat (isolates bright structures)
   MorphologyOperations.white_top_hat(image, output, radius=4)

   # Region fill (fill holes in binary mask)
   MorphologyOperations.region_fill(image, output)

Frequency Domain
----------------

FFT-based operations:

.. code-block:: python

   from medical_image import FrequencyOperations

   output = image.clone()
   FrequencyOperations.fft(image, output)
   # output.pixel_data now holds the magnitude spectrum

   FrequencyOperations.inverse_fft(output, reconstructed)

Metrics
-------

Information-theoretic metrics:

.. code-block:: python

   from medical_image import Metrics

   # Shannon entropy
   h = Metrics.entropy(image)

   # Joint entropy of two images
   h_joint = Metrics.joint_entropy(image1, image2)

   # Mutual information
   mi = Metrics.mutual_information(image1, image2)

   # Variance (global or local)
   var = Metrics.variance(image)

Mammography-Specific
--------------------

Specialized operations for mammogram preprocessing:

.. code-block:: python

   from medical_image import MammographyPreprocessing

   # Extract breast region mask
   mask = MammographyPreprocessing.breast_mask(image)

   # Apply DICOM windowing
   windowed = MammographyPreprocessing.dicom_window(
       image, window_center=2000, window_width=3000
   )

   # Normalize bit depth (12-bit DICOM -> [0, 255])
   normalized = MammographyPreprocessing.normalize_bit_depth(
       image, bits_stored=12
   )