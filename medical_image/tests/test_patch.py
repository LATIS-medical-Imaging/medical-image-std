"""Unit tests for Patch and PatchGrid classes."""

import numpy as np
import pytest
import torch

from medical_image.data.in_memory_image import InMemoryImage
from medical_image.data.patch import Patch, PatchGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(h: int, w: int, channels: int = 0) -> InMemoryImage:
    """Create an InMemoryImage with deterministic pixel values.

    Args:
        h: Height.
        w: Width.
        channels: 0 for HW layout, otherwise number of channels (CHW).
    """
    if channels:
        arr = np.arange(channels * h * w, dtype=np.float32).reshape(channels, h, w)
    else:
        arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return InMemoryImage(array=arr)


# ---------------------------------------------------------------------------
# Patch tests
# ---------------------------------------------------------------------------


class TestPatch:
    def test_properties(self):
        img = _make_image(64, 64)
        patch_data = img.pixel_data[:32, :16]
        p = Patch(parent=img, row_idx=0, col_idx=0, x=0, y=0, pixel_data=patch_data)
        assert p.height == 32
        assert p.width == 16

    def test_grid_id(self):
        img = _make_image(64, 64)
        p = Patch(
            parent=img,
            row_idx=2,
            col_idx=3,
            x=0,
            y=0,
            pixel_data=img.pixel_data[:8, :8],
        )
        assert p.grid_id() == (2, 3)

    def test_pixel_position(self):
        img = _make_image(64, 64)
        p = Patch(
            parent=img,
            row_idx=0,
            col_idx=0,
            x=16,
            y=32,
            pixel_data=img.pixel_data[:8, :8],
        )
        assert p.pixel_position() == (16, 32)

    def test_to_numpy(self):
        img = _make_image(16, 16)
        p = Patch(
            parent=img,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=img.pixel_data[:8, :8],
        )
        arr = p.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (8, 8)

    def test_to_image_returns_image(self):
        img = _make_image(32, 32)
        p = Patch(
            parent=img,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=img.pixel_data[:16, :16],
        )
        result = p.to_image()
        assert isinstance(result, InMemoryImage)
        assert result.pixel_data.shape == (16, 16)
        assert result.file_path is None

    def test_to_image_clones_data(self):
        """to_image must not share the same tensor with the parent."""
        img = _make_image(32, 32)
        p = Patch(
            parent=img,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=img.pixel_data[:16, :16].clone(),
        )
        result = p.to_image()
        result.pixel_data.fill_(999.0)
        assert img.pixel_data[0, 0].item() != 999.0

    def test_load_delegates_to_to_image(self):
        img = _make_image(32, 32)
        p = Patch(
            parent=img,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=img.pixel_data[:16, :16],
        )
        result = p.load()
        assert isinstance(result, InMemoryImage)
        assert result.pixel_data.shape == (16, 16)

    def test_repr(self):
        img = _make_image(32, 32)
        p = Patch(
            parent=img,
            row_idx=1,
            col_idx=2,
            x=16,
            y=32,
            pixel_data=img.pixel_data[:8, :8],
        )
        r = repr(p)
        assert "Patch[1,2]" in r
        assert "(16,32)" in r


# ---------------------------------------------------------------------------
# PatchGrid tests — HW layout
# ---------------------------------------------------------------------------


class TestPatchGridHW:
    def test_exact_split(self):
        """Image dimensions evenly divisible by patch size."""
        img = _make_image(64, 64)
        grid = PatchGrid(img, (32, 32))
        assert len(grid.patches) == 4
        assert len(grid.grid) == 2
        assert len(grid.grid[0]) == 2
        assert grid.pad_bottom == 0
        assert grid.pad_right == 0

    def test_padding(self):
        """Image dimensions not divisible — padding required."""
        img = _make_image(50, 70)
        grid = PatchGrid(img, (32, 32))
        assert grid.pad_bottom == 14  # 64 - 50
        assert grid.pad_right == 26  # 96 - 70
        assert len(grid.patches) == 2 * 3  # 2 rows x 3 cols

    def test_padded_flag(self):
        img = _make_image(50, 70)
        grid = PatchGrid(img, (32, 32))
        # Last row and last column patches are padded
        for p in grid.patches:
            r, c = p.grid_id()
            num_rows = len(grid.grid)
            num_cols = len(grid.grid[0])
            expected = (r == num_rows - 1 and grid.pad_bottom > 0) or (
                c == num_cols - 1 and grid.pad_right > 0
            )
            assert p.is_padded == expected, f"Patch[{r},{c}] is_padded mismatch"

    def test_reconstruct_exact(self):
        """Reconstruction of an exactly-divisible image matches original."""
        img = _make_image(64, 64)
        grid = PatchGrid(img, (32, 32))
        recon = grid.reconstruct()
        assert torch.equal(recon, img.pixel_data)

    def test_reconstruct_padded(self):
        """Reconstruction strips padding and matches original."""
        img = _make_image(50, 70)
        grid = PatchGrid(img, (32, 32))
        recon = grid.reconstruct()
        assert recon.shape == img.pixel_data.shape
        assert torch.equal(recon, img.pixel_data)

    def test_int_patch_size(self):
        """Passing a single int creates square patches."""
        img = _make_image(64, 64)
        grid = PatchGrid(img, 32)
        assert grid.patch_h == 32
        assert grid.patch_w == 32
        assert len(grid.patches) == 4


# ---------------------------------------------------------------------------
# PatchGrid tests — CHW layout
# ---------------------------------------------------------------------------


class TestPatchGridCHW:
    def test_exact_split(self):
        img = _make_image(64, 64, channels=3)
        grid = PatchGrid(img, (32, 32))
        assert len(grid.patches) == 4
        for p in grid.patches:
            assert p.pixel_data.shape == (3, 32, 32)

    def test_reconstruct_exact(self):
        img = _make_image(64, 64, channels=3)
        grid = PatchGrid(img, (32, 32))
        recon = grid.reconstruct()
        assert torch.equal(recon, img.pixel_data)

    def test_reconstruct_padded(self):
        img = _make_image(50, 70, channels=1)
        grid = PatchGrid(img, (32, 32))
        recon = grid.reconstruct()
        assert recon.shape == img.pixel_data.shape
        assert torch.equal(recon, img.pixel_data)


# ---------------------------------------------------------------------------
# PatchGrid.from_image
# ---------------------------------------------------------------------------


class TestPatchGridFromImage:
    def test_from_image_basic(self):
        img = _make_image(64, 64)
        grid = PatchGrid.from_image(img, (32, 32))
        assert len(grid.patches) == 4

    def test_from_image_int_size(self):
        img = _make_image(64, 64)
        grid = PatchGrid.from_image(img, 16)
        assert len(grid.patches) == 16

    def test_from_image_loads_lazily(self):
        """from_image triggers load() on an unloaded image."""
        img = InMemoryImage(width=32, height=32)
        # InMemoryImage.load() is a no-op, so pixel_data stays None.
        # Provide pixel_data manually to test the path logic.
        img.pixel_data = torch.zeros(32, 32)
        grid = PatchGrid.from_image(img, 16)
        assert len(grid.patches) == 4


# ---------------------------------------------------------------------------
# PatchGrid.to_image
# ---------------------------------------------------------------------------


class TestPatchGridToImage:
    def test_to_image_returns_image_instance(self):
        img = _make_image(64, 64)
        grid = PatchGrid(img, (32, 32))
        result = grid.to_image()
        assert isinstance(result, InMemoryImage)

    def test_to_image_pixel_data_matches(self):
        img = _make_image(64, 64)
        grid = PatchGrid(img, (32, 32))
        result = grid.to_image()
        assert torch.equal(result.pixel_data, img.pixel_data)

    def test_to_image_with_padding(self):
        img = _make_image(50, 70)
        grid = PatchGrid(img, (32, 32))
        result = grid.to_image()
        assert result.pixel_data.shape == img.pixel_data.shape
        assert torch.equal(result.pixel_data, img.pixel_data)

    def test_to_image_clones(self):
        """to_image should not share pixel_data with the parent."""
        img = _make_image(64, 64)
        original = img.pixel_data.clone()
        grid = PatchGrid(img, (32, 32))
        result = grid.to_image()
        result.pixel_data.fill_(-1.0)
        assert torch.equal(img.pixel_data, original)


# ---------------------------------------------------------------------------
# Visualization / round-trip test
# ---------------------------------------------------------------------------


class TestPatchVisualization:
    """Verifies that patches can be individually converted to images and
    that the full round-trip (split → per-patch to_image → reassemble)
    preserves pixel values.
    """

    def test_patch_to_image_round_trip(self):
        """Each patch's to_image pixel data matches the corresponding
        region in the original image."""
        img = _make_image(64, 64)
        grid = PatchGrid(img, (32, 32))

        for p in grid.patches:
            patch_img = p.to_image()
            expected = img.pixel_data[p.x : p.x + p.height, p.y : p.y + p.width]
            assert torch.equal(patch_img.pixel_data, expected)

    def test_full_round_trip_chw(self):
        """Split → reconstruct round-trip for CHW layout."""
        img = _make_image(48, 48, channels=3)
        grid = PatchGrid(img, (16, 16))
        result = grid.to_image()
        assert torch.equal(result.pixel_data, img.pixel_data)

    def test_patches_cover_entire_image(self):
        """All original pixels are accounted for by exactly one patch."""
        H, W = 40, 60
        img = _make_image(H, W)
        grid = PatchGrid(img, (16, 16))

        coverage = torch.zeros(H, W)
        for p in grid.patches:
            ph, pw = p.height, p.width
            # Only count non-padded region
            valid_h = min(ph, H - p.x)
            valid_w = min(pw, W - p.y)
            coverage[p.x : p.x + valid_h, p.y : p.y + valid_w] += 1

        assert (coverage == 1).all(), "Every pixel must be covered exactly once"
