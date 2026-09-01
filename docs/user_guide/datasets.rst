Datasets
========

The framework provides PyTorch-compatible dataset classes for common mammography datasets. All datasets follow the **lazy loading** pattern --- only metadata is scanned at initialization; actual image loading happens in ``__getitem__``.

Available Datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Class
     - Dataset
     - Features
   * - :class:`~medical_image.datasets.inbreast.INbreastDataset`
     - INbreast
     - DICOM + XML/COCO annotations, on-the-fly mask generation
   * - :class:`~medical_image.datasets.custom_inbreast.CustomINbreastDataset`
     - Custom INbreast
     - Extends INbreast with TIF mask support
   * - :class:`~medical_image.datasets.cbis_ddsm.CBISDDSMDataset`
     - CBIS-DDSM
     - Full-image and patch modes, automatic DICOM pairing

Output Format
-------------

All datasets return a standardized dictionary:

.. code-block:: python

   sample = dataset[0]
   # {
   #     "image":    torch.Tensor  [1, H, W],
   #     "mask":     torch.Tensor  [1, H, W],
   #     "metadata": {"case_id": "...", ...}
   # }

INbreast Dataset
----------------

.. code-block:: python

   from medical_image.datasets.inbreast import INbreastDataset

   dataset = INbreastDataset(
       root_dir="/path/to/INbreast Release 1.0",
       target_size=(512, 512),  # optional resize
   )

   sample = dataset[0]
   image = sample["image"]     # [1, 512, 512]
   mask = sample["mask"]       # [1, 512, 512] binary
   meta = sample["metadata"]   # case_id, file info

Supports two directory layouts:

1. **COCO JSON** --- ``annotations.json`` + ``images/`` directory
2. **XML (legacy)** --- ``AllDICOMs/``, ``AllXML/``, optional ``AllROI/``

CBIS-DDSM Dataset
-----------------

.. code-block:: python

   from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

   # Full-image mode
   dataset = CBISDDSMDataset(
       root_dir="/path/to/CBIS-DDSM",
       percentage=0.5,   # use 50% of data
   )

   # Patch-based mode (sliding window)
   dataset = CBISDDSMDataset(
       root_dir="/path/to/CBIS-DDSM",
       mode="patch",
       patch_size=256,
       stride=128,
   )

PyTorch DataLoader Integration
------------------------------

.. code-block:: python

   from torch.utils.data import DataLoader

   dataset = INbreastDataset("/path/to/data", target_size=(256, 256))
   loader = DataLoader(
       dataset,
       batch_size=8,
       shuffle=True,
       collate_fn=dataset.collate_fn,
       num_workers=4,
   )

   for batch in loader:
       images = batch["image"]   # [B, 1, H, W]
       masks = batch["mask"]     # [B, 1, H, W]

Creating a Custom Dataset
-------------------------

Subclass :class:`~medical_image.datasets.base_dataset.BaseDataset` and implement two methods:

.. code-block:: python

   from medical_image.datasets.base_dataset import BaseDataset

   class MyDataset(BaseDataset):
       def _build_sample_list(self):
           # Scan root_dir and populate self._samples
           for path in self.root_dir.glob("*.dcm"):
               self._samples.append({"path": path})

       def _load_sample(self, idx):
           info = self._samples[idx]
           image = DicomImage(str(info["path"]))
           image.load()
           return {
               "image": image.pixel_data.unsqueeze(0),
               "mask": torch.zeros_like(image.pixel_data).unsqueeze(0),
               "metadata": {"path": str(info["path"])},
           }