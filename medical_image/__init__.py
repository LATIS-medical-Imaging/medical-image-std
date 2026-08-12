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
from medical_image.data.mammogram_feature import GlobalMammogramFeatures
from medical_image.data.region_of_interest import RegionOfInterest
from medical_image.data.physical_features import CandidatePhysicalFeatures, Particle, SpatialParticleFeatures

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
from medical_image.algorithms.global_mammogram_reasoning_algorithm import GlobalMammogramReasoningAlgorithm
from medical_image.algorithms.native_resolution_segmentation import NativeResolutionSegmentationAlgorithm
from medical_image.algorithms.particle_builder import ParticleBuilder
from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.algorithms.pfcm import PFCMAlgorithm
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.algorithms.scale_profile_analyzer import ScaleProfileAnalyzer
from medical_image.algorithms.candidate_scale_signature import CandidateScaleSignature
from medical_image.algorithms.multiscale_top_hat import MultiScaleTopHatAlgorithm
from medical_image.algorithms.differential_blob_algorithm import DifferentialBlobAlgorithm
from medical_image.algorithms.candidate_evidence_algorithm import CandidateEvidenceAlgorithm
from medical_image.algorithms.candidate_generation_algorithm import CandidateGenerationAlgorithm
from medical_image.algorithms.local_physical_analysis_algorithm import LocalPhysicalAnalysisAlgorithm
from medical_image.algorithms.spatial_reasoning_algorithm import SpatialReasoningAlgorithm
from medical_image.algorithms.simple_approach import SimpleApproach
from medical_image.algorithms.dog import DoG
from medical_image.algorithms.gabor_orientation_algorithm import GaborOrientationAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.algorithms.breast_mask import BreastMaskAlgorithm
from medical_image.algorithms.dicom_window import (
    DicomWindowAlgorithm,
    GrailWindowAlgorithm,
)
from medical_image.algorithms.bit_depth_norm import BitDepthNormAlgorithm
from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm

# Utilities
from medical_image.utils.image_utils import (
    TensorConverter,
    ImageExporter,
    ImageVisualizer,
    MathematicalOperations,
)
from medical_image.data.annotation import Annotation, GeometryType
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
    check_gpu_budget,
    estimate_image_bytes,
)

__all__ = [
    # Data
    "Image",
    "DicomImage",
    "CandidatePhysicalFeatures",
    "GlobalMammogramFeatures",
    "GlobalMammogramReasoningAlgorithm",
    "NativeResolutionSegmentationAlgorithm",
    "Particle",
    "SpatialParticleFeatures",
    "SpatialReasoningAlgorithm",
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
    "GaborOrientationAlgorithm",
    "DifferentialBlobAlgorithm",
    "CandidateScaleSignature",
    "ScaleProfileAnalyzer",
    "ParticleBuilder",
    "MultiScaleTopHatAlgorithm",
    "LocalPhysicalAnalysisAlgorithm",
    "CandidateEvidenceAlgorithm",
    "FebdsAlgorithm",
    "SimpleApproach",
    "FCMAlgorithm",
    "PFCMAlgorithm",
    "TopHatAlgorithm",
    "DoG",
    "KMeansAlgorithm",
    "BreastMaskAlgorithm",
    "DicomWindowAlgorithm",
    "GrailWindowAlgorithm",
    "CandidateGenerationAlgorithm",
    "BitDepthNormAlgorithm",
    "DeepSegmentationAlgorithm",
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
    "check_gpu_budget",
    "estimate_image_bytes",
]
