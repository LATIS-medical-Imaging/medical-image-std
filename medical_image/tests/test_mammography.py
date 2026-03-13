"""
Unit tests for mammography preprocessing module.

Tests cover:
- Breast region masking (Otsu + largest connected component)
- DICOM windowing (simple WC/WW)
- GRAIL algorithm
- Bit depth normalization
- GPU support (when CUDA is available)
"""

import pytest
import torch

from medical_image.data.in_memory_image import InMemoryImage
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
# Breast Region Masking
# ---------------------------------------------------------------------------

class TestBreastMask:
    """Tests for breast_mask and apply_breast_mask."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_breast_mask_returns_binary(self, device):
        image = _make_mammogram(device)
        result = MammographyPreprocessing.breast_mask(image, device=device)
        unique = torch.unique(result.pixel_data)
        assert all(v in (0, 1) for v in unique.tolist())

    @pytest.mark.parametrize("device", DEVICES)
    def test_breast_mask_shape_preserved(self, device):
        image = _make_mammogram(device)
        result = MammographyPreprocessing.breast_mask(image, device=device)
        assert result.pixel_data.shape == image.pixel_data.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_breast_mask_selects_largest_component(self, device):
        image = _make_simple_binary_scene(device)
        result = MammographyPreprocessing.breast_mask(image, device=device)
        mask = result.pixel_data
        # The large blob (50x50=2500 pixels) should be selected, not the small (10x10=100)
        assert mask[30, 30] == 1  # centre of large blob
        assert mask[75, 75] == 0  # centre of small blob

    @pytest.mark.parametrize("device", DEVICES)
    def test_apply_breast_mask_zeros_background(self, device):
        image = _make_mammogram(device)
        result = MammographyPreprocessing.apply_breast_mask(image, device=device)
        mask = MammographyPreprocessing.breast_mask(image, device=device)
        bg = mask.pixel_data == 0
        assert (result.pixel_data[bg] == 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_apply_breast_mask_preserves_foreground(self, device):
        image = _make_mammogram(device)
        result = MammographyPreprocessing.apply_breast_mask(image, device=device)
        mask = MammographyPreprocessing.breast_mask(image, device=device)
        fg = mask.pixel_data == 1
        # Foreground values should match original
        orig = image.pixel_data.to(device).float()
        assert torch.allclose(result.pixel_data[fg], orig[fg])

    def test_breast_mask_output_param(self):
        image = _make_mammogram()
        output = image.clone()
        ret = MammographyPreprocessing.breast_mask(image, output=output)
        assert ret is output
        assert output.pixel_data is not None

    def test_breast_mask_all_black_image(self):
        img = InMemoryImage(array=torch.zeros(64, 64))
        result = MammographyPreprocessing.breast_mask(img)
        assert result.pixel_data.sum() == 0


# ---------------------------------------------------------------------------
# DICOM Windowing
# ---------------------------------------------------------------------------

class TestDicomWindow:
    """Tests for dicom_window."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_output_range_0_255(self, device):
        image = _make_mammogram(device)
        result = MammographyPreprocessing.dicom_window(
            image, window_center=2000, window_width=3000, device=device
        )
        assert result.pixel_data.min() >= 0
        assert result.pixel_data.max() <= 255

    @pytest.mark.parametrize("device", DEVICES)
    def test_explicit_wc_ww(self, device):
        # Uniform image at intensity 1000; WC=1000, WW=100 → maps to 127.5
        img = InMemoryImage(array=torch.full((64, 64), 1000.0, device=device))
        result = MammographyPreprocessing.dicom_window(
            img, window_center=1000, window_width=100, device=device
        )
        expected = 127.5
        assert torch.allclose(result.pixel_data, torch.full_like(result.pixel_data, expected))

    def test_full_range_fallback(self):
        """Without WC/WW, the full dynamic range should be used."""
        img = InMemoryImage(array=torch.linspace(0, 4095, 256 * 256).reshape(256, 256))
        result = MammographyPreprocessing.dicom_window(img)
        assert result.pixel_data.min().item() == pytest.approx(0.0, abs=0.1)
        assert result.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    def test_dicom_window_output_param(self):
        image = _make_mammogram()
        output = image.clone()
        ret = MammographyPreprocessing.dicom_window(
            image, output=output, window_center=2000, window_width=3000
        )
        assert ret is output

    @pytest.mark.parametrize("device", DEVICES)
    def test_clamp_below_window(self, device):
        """Pixels below the window should be clamped to 0."""
        img = InMemoryImage(array=torch.full((32, 32), 100.0, device=device))
        result = MammographyPreprocessing.dicom_window(
            img, window_center=2000, window_width=100, device=device
        )
        assert (result.pixel_data == 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_clamp_above_window(self, device):
        """Pixels above the window should be clamped to 255."""
        img = InMemoryImage(array=torch.full((32, 32), 4000.0, device=device))
        result = MammographyPreprocessing.dicom_window(
            img, window_center=2000, window_width=100, device=device
        )
        assert (result.pixel_data == 255).all()


# ---------------------------------------------------------------------------
# GRAIL Windowing
# ---------------------------------------------------------------------------

class TestGrailWindow:
    """Tests for grail_window."""

    def test_grail_output_range(self):
        image = _make_mammogram()
        result = MammographyPreprocessing.grail_window(
            image, n_scales=2, n_orientations=4, delta=300, k_max=1
        )
        assert result.pixel_data.min() >= 0
        assert result.pixel_data.max() <= 255

    def test_grail_stores_a_b(self):
        image = _make_mammogram()
        result = MammographyPreprocessing.grail_window(
            image, n_scales=2, n_orientations=4, delta=300, k_max=1
        )
        assert hasattr(result, "grail_a")
        assert hasattr(result, "grail_b")
        assert result.grail_a < result.grail_b

    def test_grail_shape_preserved(self):
        image = _make_mammogram()
        result = MammographyPreprocessing.grail_window(
            image, n_scales=2, n_orientations=4, delta=300, k_max=1
        )
        assert result.pixel_data.shape == image.pixel_data.shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_grail_device(self, device):
        image = _make_mammogram(device)
        result = MammographyPreprocessing.grail_window(
            image, n_scales=2, n_orientations=4, delta=300, k_max=1,
            device=device,
        )
        assert result.pixel_data.device.type == device

    def test_grail_output_param(self):
        image = _make_mammogram()
        output = image.clone()
        ret = MammographyPreprocessing.grail_window(
            image, output=output, n_scales=2, n_orientations=4, delta=300, k_max=1
        )
        assert ret is output


# ---------------------------------------------------------------------------
# Bit Depth Normalization
# ---------------------------------------------------------------------------

class TestNormalizeBitDepth:
    """Tests for normalize_bit_depth."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_12bit_to_255(self, device):
        img = InMemoryImage(
            array=torch.linspace(0, 4095, 100, device=device).reshape(10, 10)
        )
        result = MammographyPreprocessing.normalize_bit_depth(
            img, bits_stored=12, device=device
        )
        assert result.pixel_data.min().item() == pytest.approx(0.0, abs=0.1)
        assert result.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    @pytest.mark.parametrize("device", DEVICES)
    def test_8bit_to_255(self, device):
        img = InMemoryImage(
            array=torch.linspace(0, 255, 100, device=device).reshape(10, 10)
        )
        result = MammographyPreprocessing.normalize_bit_depth(
            img, bits_stored=8, device=device
        )
        assert result.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    @pytest.mark.parametrize("device", DEVICES)
    def test_16bit_to_255(self, device):
        img = InMemoryImage(
            array=torch.linspace(0, 65535, 100, device=device).reshape(10, 10)
        )
        result = MammographyPreprocessing.normalize_bit_depth(
            img, bits_stored=16, device=device
        )
        assert result.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    def test_auto_detect_12bit(self):
        """Without explicit bits_stored, should infer 12-bit from max value."""
        img = InMemoryImage(array=torch.tensor([[0.0, 4095.0]]))
        result = MammographyPreprocessing.normalize_bit_depth(img)
        assert result.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    def test_auto_detect_8bit(self):
        img = InMemoryImage(array=torch.tensor([[0.0, 255.0]]))
        result = MammographyPreprocessing.normalize_bit_depth(img)
        assert result.pixel_data.max().item() == pytest.approx(255.0, abs=0.1)

    def test_custom_target_max(self):
        img = InMemoryImage(array=torch.tensor([[0.0, 4095.0]]))
        result = MammographyPreprocessing.normalize_bit_depth(img, bits_stored=12, target_max=1.0)
        assert result.pixel_data.max().item() == pytest.approx(1.0, abs=0.01)

    def test_output_param(self):
        img = InMemoryImage(array=torch.tensor([[0.0, 4095.0]]))
        output = img.clone()
        ret = MammographyPreprocessing.normalize_bit_depth(img, output=output, bits_stored=12)
        assert ret is output

    @pytest.mark.parametrize("device", DEVICES)
    def test_shape_preserved(self, device):
        image = _make_mammogram(device)
        result = MammographyPreprocessing.normalize_bit_depth(image, bits_stored=12, device=device)
        assert result.pixel_data.shape == image.pixel_data.shape


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for private helper methods."""

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
        assert len(kernels) == 18  # 3 scales * 6 orientations

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
        mask[0:40, 0:40] = 1   # 1600 pixels
        mask[60:70, 60:70] = 1  # 100 pixels
        result = MammographyPreprocessing._largest_connected_component(mask, "cpu")
        assert result[20, 20] == 1
        assert result[65, 65] == 0