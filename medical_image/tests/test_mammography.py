"""
Unit tests for mammography preprocessing algorithms and utilities.

Tests cover:
- BreastMaskAlgorithm (Otsu + largest connected component)
- DicomWindowAlgorithm (simple WC/WW)
- GrailWindowAlgorithm (GRAIL perceptual metric)
- BitDepthNormAlgorithm (auto-detect BitsStored)
- MammographyPreprocessing static utilities
- GPU support (when CUDA is available)
"""

import pytest
import torch

from medical_image.data.in_memory_image import InMemoryImage
from medical_image.algorithms.breast_mask import BreastMaskAlgorithm
from medical_image.algorithms.dicom_window import (
    DicomWindowAlgorithm,
    GrailWindowAlgorithm,
)
from medical_image.algorithms.bit_depth_norm import BitDepthNormAlgorithm
from medical_image.process.mammography import MammographyPreprocessing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def _make_mammogram(device="cpu"):
    """Create a synthetic mammogram-like image (12-bit range, breast on left)."""
    img = torch.zeros(256, 256, dtype=torch.float32, device=device)
    # Simulate breast tissue on the left half with varying intensity
    img[:, :140] = torch.rand(256, 140, device=device) * 3000 + 500
    # Small bright spot (simulated microcalcification)
    img[100:110, 60:70] = 3800
    return InMemoryImage(array=img)


def _make_simple_binary_scene(device="cpu"):
    """Create an image with two distinct blobs for connected component testing."""
    img = torch.zeros(100, 100, dtype=torch.float32, device=device)
    # Large blob (should be selected as largest CC)
    img[10:60, 10:60] = 2000.0
    # Small blob
    img[70:80, 70:80] = 2000.0
    return InMemoryImage(array=img)


# ---------------------------------------------------------------------------
# BreastMaskAlgorithm
# ---------------------------------------------------------------------------


class TestBreastMaskAlgorithm:
    """Tests for BreastMaskAlgorithm (Algorithm subclass)."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_mask_only_returns_binary(self, device):
        image = _make_mammogram(device)
        algo = BreastMaskAlgorithm(mask_only=True, device=device)
        output = image.clone()
        algo(image, output)
        unique = torch.unique(output.pixel_data)
        assert all(v in (0, 1) for v in unique.tolist())

    @pytest.mark.parametrize("device", DEVICES)
    def test_mask_shape_preserved(self, device):
        image = _make_mammogram(device)
        algo = BreastMaskAlgorithm(mask_only=True, device=device)
        output = image.clone()
        algo(image, output)
        assert output.pixel_data.shape == image.pixel_data.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_selects_largest_component(self, device):
        image = _make_simple_binary_scene(device)
        algo = BreastMaskAlgorithm(mask_only=True, device=device)
        output = image.clone()
        algo(image, output)
        assert output.pixel_data[30, 30] == 1  # centre of large blob
        assert output.pixel_data[75, 75] == 0  # centre of small blob

    @pytest.mark.parametrize("device", DEVICES)
    def test_apply_mask_zeros_background(self, device):
        image = _make_mammogram(device)
        algo = BreastMaskAlgorithm(mask_only=False, device=device)
        output = image.clone()
        algo(image, output)

        mask_algo = BreastMaskAlgorithm(mask_only=True, device=device)
        mask_out = image.clone()
        mask_algo(image, mask_out)
        bg = mask_out.pixel_data == 0
        assert (output.pixel_data[bg] == 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_apply_mask_preserves_foreground(self, device):
        image = _make_mammogram(device)
        algo = BreastMaskAlgorithm(mask_only=False, device=device)
        output = image.clone()
        algo(image, output)

        mask_algo = BreastMaskAlgorithm(mask_only=True, device=device)
        mask_out = image.clone()
        mask_algo(image, mask_out)
        fg = mask_out.pixel_data == 1
        orig = image.pixel_data.to(device).float()
        assert torch.allclose(output.pixel_data[fg], orig[fg])

    def test_call_returns_output(self):
        image = _make_mammogram()
        algo = BreastMaskAlgorithm(mask_only=True)
        output = image.clone()
        ret = algo(image, output)
        assert ret is output

    def test_all_black_image(self):
        img = InMemoryImage(array=torch.zeros(64, 64))
        algo = BreastMaskAlgorithm(mask_only=True, device="cpu")
        output = img.clone()
        algo(img, output)
        assert output.pixel_data.sum() == 0

    def test_repr(self):
        algo = BreastMaskAlgorithm(device="cpu")
        r = repr(algo)
        assert "BreastMaskAlgorithm" in r

    def test_apply_batch(self):
        images = [_make_mammogram() for _ in range(3)]
        outputs = [img.clone() for img in images]
        algo = BreastMaskAlgorithm(mask_only=True, device="cpu")
        results = algo.apply_batch(images, outputs)
        assert len(results) == 3
        for out in results:
            assert out.pixel_data is not None


# ---------------------------------------------------------------------------
# DicomWindowAlgorithm
# ---------------------------------------------------------------------------


class TestDicomWindowAlgorithm:
    """Tests for DicomWindowAlgorithm (Algorithm subclass)."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_output_range_0_255(self, device):
        image = _make_mammogram(device)
        algo = DicomWindowAlgorithm(
            window_center=2000, window_width=3000, device=device
        )
        output = image.clone()
        algo(image, output)
        assert output.pixel_data.min() >= 0
        assert output.pixel_data.max() <= 255

    @pytest.mark.parametrize("device", DEVICES)
    def test_explicit_wc_ww(self, device):
        img = InMemoryImage(array=torch.full((64, 64), 1000.0, device=device))
        algo = DicomWindowAlgorithm(window_center=1000, window_width=100, device=device)
        output = img.clone()
        algo(img, output)
        assert torch.allclose(
            output.pixel_data, torch.full_like(output.pixel_data, 127.5)
        )

    def test_full_range_fallback(self):
        img = InMemoryImage(array=torch.linspace(0, 4095, 256 * 256).reshape(256, 256))
        algo = DicomWindowAlgorithm(device="cpu")
        output = img.clone()
        algo(img, output)
        assert output.pixel_data.min().item() == pytest.approx(0.0, abs=0.1)
        assert output.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    def test_call_returns_output(self):
        image = _make_mammogram()
        algo = DicomWindowAlgorithm(window_center=2000, window_width=3000)
        output = image.clone()
        ret = algo(image, output)
        assert ret is output

    @pytest.mark.parametrize("device", DEVICES)
    def test_clamp_below_window(self, device):
        img = InMemoryImage(array=torch.full((32, 32), 100.0, device=device))
        algo = DicomWindowAlgorithm(window_center=2000, window_width=100, device=device)
        output = img.clone()
        algo(img, output)
        assert (output.pixel_data == 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_clamp_above_window(self, device):
        img = InMemoryImage(array=torch.full((32, 32), 4000.0, device=device))
        algo = DicomWindowAlgorithm(window_center=2000, window_width=100, device=device)
        output = img.clone()
        algo(img, output)
        assert (output.pixel_data == 255).all()

    def test_repr(self):
        algo = DicomWindowAlgorithm(window_center=2000, window_width=3000)
        assert "DicomWindowAlgorithm" in repr(algo)


# ---------------------------------------------------------------------------
# GrailWindowAlgorithm
# ---------------------------------------------------------------------------


class TestGrailWindowAlgorithm:
    """Tests for GrailWindowAlgorithm (Algorithm subclass)."""

    def test_output_range(self):
        image = _make_mammogram()
        algo = GrailWindowAlgorithm(n_scales=2, n_orientations=4, delta=300, k_max=1)
        output = image.clone()
        algo(image, output)
        assert output.pixel_data.min() >= 0
        assert output.pixel_data.max() <= 255

    def test_stores_a_b_on_algo(self):
        image = _make_mammogram()
        algo = GrailWindowAlgorithm(n_scales=2, n_orientations=4, delta=300, k_max=1)
        output = image.clone()
        algo(image, output)
        assert algo.grail_a is not None
        assert algo.grail_b is not None
        assert algo.grail_a < algo.grail_b

    def test_shape_preserved(self):
        image = _make_mammogram()
        algo = GrailWindowAlgorithm(n_scales=2, n_orientations=4, delta=300, k_max=1)
        output = image.clone()
        algo(image, output)
        assert output.pixel_data.shape == image.pixel_data.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_device(self, device):
        image = _make_mammogram(device)
        algo = GrailWindowAlgorithm(
            n_scales=2, n_orientations=4, delta=300, k_max=1, device=device
        )
        output = image.clone()
        algo(image, output)
        assert output.pixel_data.device.type == device

    def test_call_returns_output(self):
        image = _make_mammogram()
        algo = GrailWindowAlgorithm(n_scales=2, n_orientations=4, delta=300, k_max=1)
        output = image.clone()
        ret = algo(image, output)
        assert ret is output

    def test_repr(self):
        algo = GrailWindowAlgorithm()
        assert "GrailWindowAlgorithm" in repr(algo)


# ---------------------------------------------------------------------------
# BitDepthNormAlgorithm
# ---------------------------------------------------------------------------


class TestBitDepthNormAlgorithm:
    """Tests for BitDepthNormAlgorithm (Algorithm subclass)."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_12bit_to_255(self, device):
        img = InMemoryImage(
            array=torch.linspace(0, 4095, 100, device=device).reshape(10, 10)
        )
        algo = BitDepthNormAlgorithm(bits_stored=12, device=device)
        output = img.clone()
        algo(img, output)
        assert output.pixel_data.min().item() == pytest.approx(0.0, abs=0.1)
        assert output.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    @pytest.mark.parametrize("device", DEVICES)
    def test_8bit_to_255(self, device):
        img = InMemoryImage(
            array=torch.linspace(0, 255, 100, device=device).reshape(10, 10)
        )
        algo = BitDepthNormAlgorithm(bits_stored=8, device=device)
        output = img.clone()
        algo(img, output)
        assert output.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    @pytest.mark.parametrize("device", DEVICES)
    def test_16bit_to_255(self, device):
        img = InMemoryImage(
            array=torch.linspace(0, 65535, 100, device=device).reshape(10, 10)
        )
        algo = BitDepthNormAlgorithm(bits_stored=16, device=device)
        output = img.clone()
        algo(img, output)
        assert output.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    def test_auto_detect_12bit(self):
        img = InMemoryImage(array=torch.tensor([[0.0, 4095.0]]))
        algo = BitDepthNormAlgorithm(device="cpu")
        output = img.clone()
        algo(img, output)
        assert output.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    def test_custom_target_max(self):
        img = InMemoryImage(array=torch.tensor([[0.0, 4095.0]]))
        algo = BitDepthNormAlgorithm(bits_stored=12, target_max=1.0, device="cpu")
        output = img.clone()
        algo(img, output)
        assert output.pixel_data.max().item() == pytest.approx(1.0, abs=0.01)

    def test_call_returns_output(self):
        img = InMemoryImage(array=torch.tensor([[0.0, 4095.0]]))
        algo = BitDepthNormAlgorithm(bits_stored=12)
        output = img.clone()
        ret = algo(img, output)
        assert ret is output

    @pytest.mark.parametrize("device", DEVICES)
    def test_shape_preserved(self, device):
        image = _make_mammogram(device)
        algo = BitDepthNormAlgorithm(bits_stored=12, device=device)
        output = image.clone()
        algo(image, output)
        assert output.pixel_data.shape == image.pixel_data.shape

    def test_repr(self):
        algo = BitDepthNormAlgorithm(bits_stored=12)
        assert "BitDepthNormAlgorithm" in repr(algo)

    def test_apply_batch(self):
        images = [
            InMemoryImage(array=torch.linspace(0, 4095, 100).reshape(10, 10))
            for _ in range(3)
        ]
        outputs = [img.clone() for img in images]
        algo = BitDepthNormAlgorithm(bits_stored=12, device="cpu")
        results = algo.apply_batch(images, outputs)
        assert len(results) == 3
        for out in results:
            assert out.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)


# ---------------------------------------------------------------------------
# MammographyPreprocessing static utilities (low-level)
# ---------------------------------------------------------------------------


class TestStaticUtilities:
    """Tests for MammographyPreprocessing static helper methods."""

    def test_otsu_binary_returns_0_1(self):
        img = torch.cat([torch.zeros(50), torch.ones(50) * 3000]).reshape(10, 10)
        result = MammographyPreprocessing._otsu_binary(img, "cpu")
        unique = torch.unique(result)
        assert all(v in (0, 1) for v in unique.tolist())

    def test_intensity_window_basic(self):
        img = torch.tensor([0.0, 500.0, 1000.0])
        result = MammographyPreprocessing._intensity_window(img, 0.0, 1000.0)
        assert result[0].item() == pytest.approx(0.0)
        assert result[1].item() == pytest.approx(127.5)
        assert result[2].item() == pytest.approx(255.0)

    def test_intensity_window_clamp(self):
        img = torch.tensor([-100.0, 2000.0])
        result = MammographyPreprocessing._intensity_window(img, 0.0, 1000.0)
        assert result[0].item() == 0.0
        assert result[1].item() == 255.0

    def test_detect_bits_stored_fallback(self):
        img_12 = InMemoryImage(array=torch.tensor([[4000.0]]))
        assert MammographyPreprocessing._detect_bits_stored(img_12) == 12

        img_8 = InMemoryImage(array=torch.tensor([[200.0]]))
        assert MammographyPreprocessing._detect_bits_stored(img_8) == 8

    def test_gabor_bank_count(self):
        kernels = MammographyPreprocessing._build_gabor_bank(3, 6, "cpu")
        assert len(kernels) == 18

    def test_gabor_bank_2d(self):
        kernels = MammographyPreprocessing._build_gabor_bank(2, 4, "cpu")
        for k in kernels:
            assert k.ndim == 2

    def test_largest_cc_single_blob(self):
        mask = torch.zeros(50, 50, dtype=torch.uint8)
        mask[10:30, 10:30] = 1
        result = MammographyPreprocessing._largest_connected_component(mask, "cpu")
        assert result.sum() == mask.sum()

    def test_largest_cc_picks_bigger(self):
        mask = torch.zeros(100, 100, dtype=torch.uint8)
        mask[0:40, 0:40] = 1
        mask[60:70, 60:70] = 1
        result = MammographyPreprocessing._largest_connected_component(mask, "cpu")
        assert result[20, 20] == 1
        assert result[65, 65] == 0

    def test_static_breast_mask_directly(self):
        """Ensure static method still works standalone."""
        image = _make_mammogram()
        result = MammographyPreprocessing.breast_mask(image)
        assert result.pixel_data is not None
        unique = torch.unique(result.pixel_data)
        assert all(v in (0, 1) for v in unique.tolist())

    def test_static_dicom_window_directly(self):
        image = _make_mammogram()
        result = MammographyPreprocessing.dicom_window(
            image, window_center=2000, window_width=3000
        )
        assert result.pixel_data.min() >= 0
        assert result.pixel_data.max() <= 255

    def test_static_normalize_bit_depth_directly(self):
        img = InMemoryImage(array=torch.tensor([[0.0, 4095.0]]))
        result = MammographyPreprocessing.normalize_bit_depth(img, bits_stored=12)
        assert result.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)
