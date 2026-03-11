"""
Datasets package for production-grade medical image dataloaders.

Provides PyTorch-compatible Dataset classes for:
- INbreast mammography
- Custom INbreast (with segmentation masks)
- CBIS-DDSM (large-scale, with patch-based loading)
"""

from medical_image.datasets.base_dataset import BaseDataset
from medical_image.datasets.inbreast import INbreastDataset
from medical_image.datasets.custom_inbreast import CustomINbreastDataset
from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

__all__ = [
    "BaseDataset",
    "INbreastDataset",
    "CustomINbreastDataset",
    "CBISDDSMDataset",
]
