Extending the Framework
=======================

The framework is designed for extensibility at three levels: image formats, algorithms, and datasets.

Adding an Image Format
----------------------

Subclass :class:`~medical_image.data.image.Image` and implement ``load()`` and ``save()``:

.. code-block:: python

   from medical_image.data.image import Image

   class NiftiImage(Image):
       def load(self):
           import nibabel as nib
           nii = nib.load(self.file_path)
           data = nii.get_fdata().astype("float32")
           self.pixel_data = torch.from_numpy(data)
           self._width = data.shape[1]
           self._height = data.shape[0]
           self._post_load()  # applies deferred device migration
           return self

       def save(self):
           # Write modified pixel_data back to disk
           ...

Key points:

- Call ``self._post_load()`` at the end of ``load()`` to apply deferred ``.to()`` calls.
- Set ``self._width`` and ``self._height`` from the loaded data.

Adding an Algorithm
-------------------

Subclass :class:`~medical_image.algorithms.algorithm.Algorithm` and implement ``apply()``:

.. code-block:: python

   from medical_image.algorithms.algorithm import Algorithm
   from medical_image.process.filters import Filters
   from medical_image.process.threshold import Threshold

   class AdaptiveEnhancer(Algorithm):
       def __init__(self, sigma=2.0, k=0.3, device=None):
           super().__init__(device=device)
           self.sigma = sigma
           self.k = k

       def apply(self, image, output):
           # Step 1: enhance
           Filters.gaussian_filter(
               image, output, sigma=self.sigma, device=self.device
           )
           # Step 2: threshold
           temp = output.clone()
           Threshold.sauvola_threshold(
               output, temp, window_size=15, k=self.k, device=self.device
           )
           output.pixel_data = temp.pixel_data
           return output

The base class ``__call__`` automatically handles mixed-precision wrapping via ``self.precision``.

Adding a Dataset
----------------

Subclass :class:`~medical_image.datasets.base_dataset.BaseDataset` and implement two methods:

.. code-block:: python

   from medical_image.datasets.base_dataset import BaseDataset

   class MyDataset(BaseDataset):
       def _build_sample_list(self):
           """Scan root_dir and populate self._samples with metadata."""
           for dcm_path in sorted(self.root_dir.glob("**/*.dcm")):
               mask_path = dcm_path.with_suffix(".png")
               self._samples.append({
                   "image_path": dcm_path,
                   "mask_path": mask_path if mask_path.exists() else None,
               })

       def _load_sample(self, idx):
           """Load a single sample and return the standard dict."""
           info = self._samples[idx]
           image = DicomImage(str(info["image_path"]))
           image.load()

           if info["mask_path"]:
               mask = PNGImage(str(info["mask_path"]))
               mask.load()
               mask_tensor = mask.pixel_data
           else:
               mask_tensor = torch.zeros_like(image.pixel_data)

           return {
               "image": image.pixel_data.unsqueeze(0),
               "mask": mask_tensor.unsqueeze(0),
               "metadata": {"path": str(info["image_path"])},
           }

The base class provides ``__len__``, ``__getitem__`` (with transforms), and ``collate_fn``.

Adding a Processing Operation
------------------------------

Add static methods to existing process classes, or create a new one:

.. code-block:: python

   from medical_image.data.image import requires_loaded

   class MyOperations:
       @staticmethod
       @requires_loaded
       def custom_filter(image, output, param=1.0, device=None):
           from medical_image.utils.device import resolve_device
           device = resolve_device(image, explicit=device)
           img = image.pixel_data.to(device).float()
           # ... your processing logic ...
           output.pixel_data = result