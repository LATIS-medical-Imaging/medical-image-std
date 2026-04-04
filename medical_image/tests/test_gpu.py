"""
GPU-aware tests for device management, resolve_device, DeviceContext,
@gpu_safe, mixed precision, batch processing, and pin_memory.

Tests run on both CPU and CUDA (when available).
"""

import numpy as np
import pytest
import torch

from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.filters import Filters
from medical_image.process.morphology import MorphologyOperations
from medical_image.process.threshold import Threshold
from medical_image.process.metrics import Metrics
from medical_image.process.frequency import FrequencyOperations
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.utils.device import (
    resolve_device,
    DeviceContext,
    gpu_safe,
    Precision,
    set_default_precision,
    get_dtype,
    get_default_precision,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def sample_image():
    """8x8 float32 image."""
    arr = np.random.rand(8, 8).astype(np.float32)
    return DicomImage.from_array(arr)


@pytest.fixture
def sample_16x16():
    """16x16 float32 image with a bright spot for algorithm tests."""
    arr = np.random.rand(16, 16).astype(np.float32) * 100
    arr[6:10, 6:10] = 4000  # bright MC-like cluster
    return DicomImage.from_array(arr)


@pytest.fixture
def binary_image():
    """16x16 binary image."""
    arr = (np.random.rand(16, 16) > 0.5).astype(np.float32)
    return DicomImage.from_array(arr)


# ---------------------------------------------------------------------------
# resolve_device tests
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_explicit_overrides(self, sample_image):
        device = resolve_device(sample_image, explicit="cpu")
        assert device == torch.device("cpu")

    def test_infer_from_image(self, sample_image):
        device = resolve_device(sample_image)
        assert device == torch.device("cpu")

    def test_fallback_to_cpu(self):
        device = resolve_device()
        assert device == torch.device("cpu")

    @gpu
    def test_infer_cuda(self, sample_image):
        sample_image.to("cuda")
        device = resolve_device(sample_image)
        assert device.type == "cuda"

    @gpu
    def test_explicit_cuda(self, sample_image):
        device = resolve_device(sample_image, explicit="cuda")
        assert device.type == "cuda"


# ---------------------------------------------------------------------------
# DeviceContext tests
# ---------------------------------------------------------------------------


class TestDeviceContext:
    def test_cpu_context(self):
        with DeviceContext("cpu") as ctx:
            assert ctx.device == torch.device("cpu")
            stats = ctx.memory_stats()
            assert stats["device"] == "cpu"

    def test_cuda_fallback_when_unavailable(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test fallback")
        with DeviceContext("cuda") as ctx:
            assert ctx.device == torch.device("cpu")

    @gpu
    def test_cuda_context(self):
        with DeviceContext("cuda", verbose=True) as ctx:
            assert ctx.device.type == "cuda"
            stats = ctx.memory_stats()
            assert "allocated_gb" in stats
            assert "free_gb" in stats


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_default_is_full(self):
        assert get_default_precision() == Precision.FULL
        assert get_dtype() == torch.float32

    def test_set_half(self):
        original = get_default_precision()
        try:
            set_default_precision(Precision.HALF)
            assert get_dtype() == torch.float16
        finally:
            set_default_precision(original)

    def test_set_bfloat16(self):
        original = get_default_precision()
        try:
            set_default_precision(Precision.BFLOAT16)
            assert get_dtype() == torch.bfloat16
        finally:
            set_default_precision(original)


# ---------------------------------------------------------------------------
# Image.pin_memory tests
# ---------------------------------------------------------------------------


class TestPinMemory:
    def test_pin_memory_cpu(self, sample_image):
        sample_image.pin_memory()
        assert sample_image.pixel_data.is_pinned()

    def test_pin_memory_returns_self(self, sample_image):
        result = sample_image.pin_memory()
        assert result is sample_image


# ---------------------------------------------------------------------------
# Filters with device=None (auto-infer)
# ---------------------------------------------------------------------------


class TestFiltersDeviceInference:
    @pytest.mark.parametrize("device", DEVICES)
    def test_gaussian_filter(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = Filters.gaussian_filter(sample_image, output, sigma=1.5)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_median_filter(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = Filters.median_filter(sample_image, output, size=3)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_difference_of_gaussian(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = Filters.difference_of_gaussian(sample_image, output, low_sigma=1.0)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_laplacian_of_gaussian(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = Filters.laplacian_of_gaussian(sample_image, output, sigma=1.5)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_gamma_correction(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = Filters.gamma_correction(sample_image, output, gamma=1.25)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_convolution(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        kernel = torch.ones(3, 3) / 9.0
        result = Filters.convolution(sample_image, output, kernel)
        assert result.pixel_data.device.type == device


# ---------------------------------------------------------------------------
# Morphology with device=None
# ---------------------------------------------------------------------------


class TestMorphologyDeviceInference:
    @pytest.mark.parametrize("device", DEVICES)
    def test_erosion(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = MorphologyOperations.erosion(sample_image, output, radius=2)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_dilation(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = MorphologyOperations.dilation(sample_image, output, radius=2)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_white_top_hat(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = MorphologyOperations.white_top_hat(sample_image, output, radius=2)
        assert result.pixel_data.device.type == device


# ---------------------------------------------------------------------------
# Threshold with device=None
# ---------------------------------------------------------------------------


class TestThresholdDeviceInference:
    @pytest.mark.parametrize("device", DEVICES)
    def test_otsu(self, sample_16x16, device):
        sample_16x16.to(device)
        output = sample_16x16.clone()
        result = Threshold.otsu_threshold(sample_16x16, output)
        assert result.pixel_data.device.type == device

    @pytest.mark.parametrize("device", DEVICES)
    def test_sauvola(self, sample_16x16, device):
        sample_16x16.to(device)
        output = sample_16x16.clone()
        result = Threshold.sauvola_threshold(sample_16x16, output, window_size=5)
        assert result.pixel_data.device.type == device


# ---------------------------------------------------------------------------
# Metrics with device=None
# ---------------------------------------------------------------------------


class TestMetricsDeviceInference:
    @pytest.mark.parametrize("device", DEVICES)
    def test_entropy(self, sample_image, device):
        sample_image.to(device)
        val = Metrics.entropy(sample_image)
        assert isinstance(val, float)

    @pytest.mark.parametrize("device", DEVICES)
    def test_variance(self, sample_image, device):
        sample_image.to(device)
        output = InMemoryImage(array=torch.empty(1))
        result = Metrics.variance(sample_image, output)
        assert result.pixel_data is not None


# ---------------------------------------------------------------------------
# Frequency with device=None
# ---------------------------------------------------------------------------


class TestFrequencyDeviceInference:
    @pytest.mark.parametrize("device", DEVICES)
    def test_fft(self, sample_image, device):
        sample_image.to(device)
        output = sample_image.clone()
        result = FrequencyOperations.fft(sample_image, output)
        assert result.pixel_data.device.type == device


# ---------------------------------------------------------------------------
# Algorithm.apply_batch
# ---------------------------------------------------------------------------


class TestAlgorithmBatch:
    def test_apply_batch_tophat(self):
        images = [
            DicomImage.from_array(np.random.rand(16, 16).astype(np.float32))
            for _ in range(3)
        ]
        outputs = [img.clone() for img in images]
        algo = TopHatAlgorithm(radius=2, device="cpu")
        results = algo.apply_batch(images, outputs)
        assert len(results) == 3
        for r in results:
            assert r.pixel_data is not None

    def test_apply_batch_kmeans(self):
        images = [
            DicomImage.from_array(np.random.rand(16, 16).astype(np.float32) * 4095)
            for _ in range(2)
        ]
        outputs = [img.clone() for img in images]
        algo = KMeansAlgorithm(k=2, device="cpu")
        results = algo.apply_batch(images, outputs)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Batch filter
# ---------------------------------------------------------------------------


class TestBatchFilter:
    @pytest.mark.parametrize("device", DEVICES)
    def test_gaussian_filter_batch(self, device):
        batch = torch.randn(4, 1, 16, 16, device=device)
        result = Filters.gaussian_filter_batch(batch, sigma=1.5, device=device)
        assert result.shape == (4, 1, 16, 16)
        assert result.device.type == device


# ---------------------------------------------------------------------------
# Algorithm repr with precision
# ---------------------------------------------------------------------------


class TestAlgorithmRepr:
    def test_repr_includes_precision(self):
        algo = TopHatAlgorithm(radius=4, device="cpu")
        r = repr(algo)
        assert "FULL" in r
        assert "cpu" in r

    def test_repr_half_precision(self):
        algo = KMeansAlgorithm(k=3, device="cpu")
        algo.precision = Precision.HALF
        r = repr(algo)
        assert "HALF" in r
