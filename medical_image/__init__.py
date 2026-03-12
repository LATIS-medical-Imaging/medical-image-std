"""
Medical Image Standard — A framework for medical image processing.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("medical-image-std")
except PackageNotFoundError:
    __version__ = "0.2.0.dev0"

# Data layer
from medical_image.data.image import Image, requires_loaded
from medical_image.data.dicom_image import DicomImage
from medical_image.data.png_image import PNGImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.data.patch import PatchGrid, Patch
from medical_image.data.region_of_interest import RegionOfInterest

# Processing layer
from medical_image.process.filters import Filters
from medical_image.process.morphology import MorphologyOperations
from medical_image.process.threshold import Threshold
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.metrics import Metrics

# Algorithm layer
from medical_image.algorithms.algorithm import Algorithm
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.algorithms.pfcm import PFCMAlgorithm
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm

# Utilities
from medical_image.utils.image_utils import (
    TensorConverter,
    ImageExporter,
    ImageVisualizer,
    MathematicalOperations,
)
from medical_image.utils.annotation import Annotation, GeometryType

__all__ = [
    # Data
    "Image",
    "DicomImage",
    "PNGImage",
    "InMemoryImage",
    "PatchGrid",
    "Patch",
    "RegionOfInterest",
    # Processing
    "Filters",
    "MorphologyOperations",
    "Threshold",
    "FrequencyOperations",
    "Metrics",
    # Algorithms
    "Algorithm",
    "FebdsAlgorithm",
    "FCMAlgorithm",
    "PFCMAlgorithm",
    "TopHatAlgorithm",
    "KMeansAlgorithm",
    # Utilities
    "TensorConverter",
    "ImageExporter",
    "ImageVisualizer",
    "MathematicalOperations",
    "Annotation",
    "GeometryType",
    "requires_loaded",
]
