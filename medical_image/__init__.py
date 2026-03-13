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
from medical_image.process.mammography import MammographyPreprocessing

# Algorithm layer
from medical_image.algorithms.algorithm import Algorithm
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.algorithms.pfcm import PFCMAlgorithm
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.algorithms.breast_mask import BreastMaskAlgorithm
from medical_image.algorithms.dicom_window import DicomWindowAlgorithm, GrailWindowAlgorithm
from medical_image.algorithms.bit_depth_norm import BitDepthNormAlgorithm

# Utilities
from medical_image.utils.image_utils import (
    TensorConverter,
    ImageExporter,
    ImageVisualizer,
    MathematicalOperations,
)
from medical_image.utils.annotation import Annotation, GeometryType
from medical_image.utils.device import (
    resolve_device,
    Precision,
    set_default_precision,
    get_default_precision,
    get_dtype,
    DeviceContext,
    gpu_safe,
    AsyncGPUPipeline,
    MultiGPUAlgorithm,
)

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
    "MammographyPreprocessing",
    # Algorithms
    "Algorithm",
    "FebdsAlgorithm",
    "FCMAlgorithm",
    "PFCMAlgorithm",
    "TopHatAlgorithm",
    "KMeansAlgorithm",
    "BreastMaskAlgorithm",
    "DicomWindowAlgorithm",
    "GrailWindowAlgorithm",
    "BitDepthNormAlgorithm",
    # Utilities
    "TensorConverter",
    "ImageExporter",
    "ImageVisualizer",
    "MathematicalOperations",
    "Annotation",
    "GeometryType",
    "requires_loaded",
    # GPU utilities
    "resolve_device",
    "Precision",
    "set_default_precision",
    "get_default_precision",
    "get_dtype",
    "DeviceContext",
    "gpu_safe",
    "AsyncGPUPipeline",
    "MultiGPUAlgorithm",
]
