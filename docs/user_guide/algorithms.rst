Algorithms
==========

Algorithms are multi-step processing pipelines that inherit from :class:`~medical_image.algorithms.algorithm.Algorithm`. They follow the **Template Method** pattern: the base class handles mixed-precision wrapping, and subclasses implement ``apply()``.

Usage Pattern
-------------

.. code-block:: python

   algo = SomeAlgorithm(device="cuda")
   output = image.clone()
   algo(image=image, output=output)

   mask = output.pixel_data          # result tensor
   annotations = output.annotations  # per-lesion annotations (if applicable)

Available Algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Algorithm
     - Description
   * - :class:`~medical_image.algorithms.FEBDS.FebdsAlgorithm`
     - Fourier Enhancement + Band-pass Detection and Segmentation for microcalcifications
   * - :class:`~medical_image.algorithms.top_hat.TopHatAlgorithm`
     - White top-hat morphological filtering
   * - :class:`~medical_image.algorithms.kmeans.KMeansAlgorithm`
     - Hard K-Means clustering segmentation
   * - :class:`~medical_image.algorithms.fcm.FCMAlgorithm`
     - Fuzzy C-Means soft clustering
   * - :class:`~medical_image.algorithms.pfcm.PFCMAlgorithm`
     - Possibilistic Fuzzy C-Means (typicality-based)
   * - :class:`~medical_image.algorithms.breast_mask.BreastMaskAlgorithm`
     - Breast region extraction via Otsu + largest connected component
   * - :class:`~medical_image.algorithms.dicom_window.DicomWindowAlgorithm`
     - Linear DICOM window center / width mapping
   * - :class:`~medical_image.algorithms.dicom_window.GrailWindowAlgorithm`
     - GRAIL perceptual windowing with Gabor-filtered MI
   * - :class:`~medical_image.algorithms.bit_depth_norm.BitDepthNormAlgorithm`
     - Automatic bit-depth detection and normalization
   * - :class:`~medical_image.algorithms.sbrg.SbrgAlgorithm`
     - Seed-Based Region Growing segmentation
   * - :class:`~medical_image.algorithms.deep_segmentation.DeepSegmentationAlgorithm`
     - Deep learning segmentation with remote model support

FEBDS (Microcalcification Detection)
-------------------------------------

The FEBDS algorithm combines enhancement, filtering, thresholding, and morphology into a single pipeline:

.. code-block:: python

   from medical_image import FebdsAlgorithm

   algo = FebdsAlgorithm(method="dog", device="cpu")
   output = image.clone()
   algo(image=image, output=output)

   mask = output.pixel_data.numpy()

Methods: ``"dog"`` (Difference of Gaussians), ``"log"`` (Laplacian of Gaussian), ``"fft"`` (Fourier band-pass).

Clustering Algorithms
---------------------

K-Means, FCM, and PFCM segment images by grouping pixels into clusters:

.. code-block:: python

   from medical_image import KMeansAlgorithm, PatchGrid, RegionOfInterest

   # Normalize and extract a patch for clustering
   normalized = RegionOfInterest.normalize(image.clone(), divisor=4095.0)
   grid = PatchGrid(normalized, (64, 64))
   patch_img = grid.patches[0].to_image()

   out = patch_img.clone()
   km = KMeansAlgorithm(k=3, device="cpu", random_state=42)
   km(patch_img, out)

   # Access clustering results
   print(km.centroids.shape)  # (3, 1)
   print(km.converged)        # True/False
   for s in km.stats:
       print(f"Cluster: {s['pixels']} pixels, MC={s['is_mc']}")

Deep Segmentation
-----------------

Run pretrained deep learning models with automatic download and caching:

.. code-block:: python

   from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm

   # List available pretrained models
   models = DeepSegmentationAlgorithm.list_available_models()
   for m in models:
       print(f"{m['name']} - {m['architecture']}, patch={m['patch_size']}")

   # Download and load
   algo = DeepSegmentationAlgorithm.from_pretrained(
       "unetpp_bce_dice_32_inbreast", device="cuda"
   )
   print(algo.model_info)  # {'architecture': 'unetpp', 'loss': 'bce_dice', ...}

   # Inference
   output = image.clone()
   algo(image=image, output=output)

   # Results
   mask_np = output.pixel_data.cpu().numpy()       # binary mask
   prob_np = algo.probability_map.cpu().numpy()     # probability map [0, 1]
   print(f"Found {algo.lesion_count} lesions")

   for ann in output.annotations:
       print(f"  {ann.label}: confidence={ann.metadata['confidence']:.2f}, "
             f"area={ann.metadata['area']}px")

Models are cached in ``~/.cache/medical-std/models/``.

Loading from Local Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   algo = DeepSegmentationAlgorithm(
       checkpoint_path="results/unet_bce_dice_128_inbreast/best_model.pt",
       device="cpu"
   )
   # Config (patch_size, clahe, threshold) auto-read from checkpoint

Breast Mask Extraction
----------------------

.. code-block:: python

   from medical_image import BreastMaskAlgorithm

   # Binary mask only
   algo = BreastMaskAlgorithm(mask_only=True)
   output = image.clone()
   algo(image, output)

   # Masked mammogram (background removed)
   algo = BreastMaskAlgorithm(mask_only=False)
   output = image.clone()
   algo(image, output)

Composing Algorithms
--------------------

Build custom pipelines by combining existing operations:

.. code-block:: python

   from medical_image.algorithms.algorithm import Algorithm
   from medical_image import Filters, Threshold

   class MyPipeline(Algorithm):
       def apply(self, image, output):
           # Step 1: Gaussian blur
           Filters.gaussian_filter(image, output, sigma=2.0, device=self.device)

           # Step 2: Otsu threshold
           temp = output.clone()
           Threshold.otsu_threshold(output, temp, device=self.device)

           output.pixel_data = temp.pixel_data
           return output