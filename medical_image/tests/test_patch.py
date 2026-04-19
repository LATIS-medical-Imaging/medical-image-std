"""Unit tests for Patch and PatchGrid classes.

Uses real DICOM images via ``mock_dicom_image`` for consistency with the
rest of the test suite, and includes algorithm-on-patch integration tests.
"""

import numpy as np
import pytest
import torch

from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.data.patch import Patch, PatchGrid
from medical_image.data.region_of_interest import RegionOfInterest
from medical_image.tests.mock_sample import mock_dicom_image

# ---------------------------------------------------------------------------
# Patch tests
# ---------------------------------------------------------------------------


class TestPatch:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_properties(self, dicom_image):
        patch_data = dicom_image.pixel_data[:32, :16]
        p = Patch(
            parent=dicom_image,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=patch_data,
        )
        assert p.height == 32
        assert p.width == 16

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_grid_id(self, dicom_image):
        p = Patch(
            parent=dicom_image,
            row_idx=2,
            col_idx=3,
            x=0,
            y=0,
            pixel_data=dicom_image.pixel_data[:8, :8],
        )
        assert p.grid_id() == (2, 3)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_pixel_position(self, dicom_image):
        p = Patch(
            parent=dicom_image,
            row_idx=0,
            col_idx=0,
            x=16,
            y=32,
            pixel_data=dicom_image.pixel_data[:8, :8],
        )
        assert p.pixel_position() == (16, 32)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_to_numpy(self, dicom_image):
        p = Patch(
            parent=dicom_image,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=dicom_image.pixel_data[:8, :8],
        )
        arr = p.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (8, 8)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_to_image_returns_image(self, dicom_image):
        p = Patch(
            parent=dicom_image,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=dicom_image.pixel_data[:16, :16],
        )
        result = p.to_image()
        assert result.pixel_data is not None
        assert result.pixel_data.shape == (16, 16)
        assert result.file_path is None

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_to_image_clones_data(self, dicom_image):
        """to_image must not share the same tensor with the parent."""
        original = dicom_image.pixel_data.clone()
        p = Patch(
            parent=dicom_image,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=dicom_image.pixel_data[:16, :16].clone(),
        )
        result = p.to_image()
        result.pixel_data.fill_(999.0)
        assert torch.equal(dicom_image.pixel_data, original)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_load_delegates_to_to_image(self, dicom_image):
        p = Patch(
            parent=dicom_image,
            row_idx=0,
            col_idx=0,
            x=0,
            y=0,
            pixel_data=dicom_image.pixel_data[:16, :16],
        )
        result = p.load()
        assert result.pixel_data is not None
        assert result.pixel_data.shape == (16, 16)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_repr(self, dicom_image):
        p = Patch(
            parent=dicom_image,
            row_idx=1,
            col_idx=2,
            x=16,
            y=32,
            pixel_data=dicom_image.pixel_data[:8, :8],
        )
        r = repr(p)
        assert "Patch[1,2]" in r
        assert "(16,32)" in r


# ---------------------------------------------------------------------------
# PatchGrid tests
# ---------------------------------------------------------------------------


class TestPatchGrid:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_split_creates_patches(self, dicom_image):
        grid = PatchGrid(dicom_image, (64, 64))
        assert len(grid.patches) > 0
        assert len(grid.grid) > 0

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_grid_dimensions(self, dicom_image):
        grid = PatchGrid(dicom_image, (64, 64))
        total = sum(len(row) for row in grid.grid)
        assert total == len(grid.patches)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_patch_size(self, dicom_image):
        grid = PatchGrid(dicom_image, (64, 64))
        for p in grid.patches:
            assert p.height == 64
            assert p.width == 64

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_padded_flag(self, dicom_image):
        grid = PatchGrid(dicom_image, (64, 64))
        num_rows = len(grid.grid)
        num_cols = len(grid.grid[0])
        for p in grid.patches:
            r, c = p.grid_id()
            expected = (r == num_rows - 1 and grid.pad_bottom > 0) or (
                c == num_cols - 1 and grid.pad_right > 0
            )
            assert p.is_padded == expected, f"Patch[{r},{c}] is_padded mismatch"

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_reconstruct(self, dicom_image):
        """Reconstruction strips padding and matches original."""
        grid = PatchGrid(dicom_image, (64, 64))
        recon = grid.reconstruct()
        assert recon.shape == dicom_image.pixel_data.shape
        assert torch.equal(recon, dicom_image.pixel_data)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_int_patch_size(self, dicom_image):
        """Passing a single int creates square patches."""
        grid = PatchGrid(dicom_image, 128)
        assert grid.patch_h == 128
        assert grid.patch_w == 128
        assert len(grid.patches) > 0


# ---------------------------------------------------------------------------
# PatchGrid.from_image
# ---------------------------------------------------------------------------


class TestPatchGridFromImage:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_from_image_basic(self, dicom_image):
        grid = PatchGrid.from_image(dicom_image, (64, 64))
        assert len(grid.patches) > 0

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_from_image_int_size(self, dicom_image):
        grid = PatchGrid.from_image(dicom_image, 128)
        assert grid.patch_h == 128
        assert grid.patch_w == 128


# ---------------------------------------------------------------------------
# PatchGrid.to_image
# ---------------------------------------------------------------------------


class TestPatchGridToImage:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_to_image_pixel_data_matches(self, dicom_image):
        grid = PatchGrid(dicom_image, (64, 64))
        result = grid.to_image()
        assert result.pixel_data is not None
        assert result.pixel_data.shape == dicom_image.pixel_data.shape
        assert torch.equal(result.pixel_data, dicom_image.pixel_data)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_to_image_clones(self, dicom_image):
        """to_image should not share pixel_data with the parent."""
        original = dicom_image.pixel_data.clone()
        grid = PatchGrid(dicom_image, (64, 64))
        result = grid.to_image()
        result.pixel_data = result.pixel_data.float().fill_(0.0)
        assert torch.equal(dicom_image.pixel_data, original)


# ---------------------------------------------------------------------------
# Visualization / round-trip test
# ---------------------------------------------------------------------------


class TestPatchVisualization:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_patch_to_image_round_trip(self, dicom_image):
        """Each patch's to_image pixel data matches the corresponding
        region in the original image."""
        grid = PatchGrid(dicom_image, (64, 64))
        for p in grid.patches:
            patch_img = p.to_image()
            expected = dicom_image.pixel_data[p.x : p.x + p.height, p.y : p.y + p.width]
            # Padded patches extend beyond the original, so compare the valid region
            valid_h = min(p.height, dicom_image.height - p.x)
            valid_w = min(p.width, dicom_image.width - p.y)
            assert torch.equal(
                patch_img.pixel_data[:valid_h, :valid_w],
                expected[:valid_h, :valid_w],
            )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_patches_cover_entire_image(self, dicom_image):
        """All original pixels are accounted for by exactly one patch."""
        H, W = dicom_image.height, dicom_image.width
        grid = PatchGrid(dicom_image, (64, 64))

        coverage = torch.zeros(H, W)
        for p in grid.patches:
            valid_h = min(p.height, H - p.x)
            valid_w = min(p.width, W - p.y)
            coverage[p.x : p.x + valid_h, p.y : p.y + valid_w] += 1

        assert (coverage == 1).all(), "Every pixel must be covered exactly once"


# ---------------------------------------------------------------------------
# Algorithm-on-patch integration tests
# ---------------------------------------------------------------------------


class TestAlgorithmOnPatch:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_kmeans_on_patch(self, dicom_image):
        """Run KMeans segmentation on individual patches extracted from a
        real DICOM image after normalization."""
        normalized = RegionOfInterest.normalize(dicom_image.clone(), divisor=4095.0)
        grid = PatchGrid(normalized, (64, 64))

        # Pick a patch with sufficient variance to avoid degenerate k-means
        patch = max(
            (p for p in grid.patches if not p.is_padded),
            key=lambda p: float(p.pixel_data.float().std()),
        )
        patch_img = patch.to_image()

        out = patch_img.clone()
        km = KMeansAlgorithm(k=3, device="cpu")
        km(patch_img, out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == patch_img.pixel_data.shape
        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})
        assert km.centroids.shape == (3, 1)
        assert km.converged or km.n_iter == km.max_iter

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_febds_on_patch(self, dicom_image):
        """Run FEBDS algorithm on individual patches extracted from a
        real DICOM image."""
        grid = PatchGrid(dicom_image, (64, 64))

        patch = next(p for p in grid.patches if not p.is_padded)
        patch_img = patch.to_image()

        if not isinstance(patch_img.pixel_data, torch.Tensor):
            patch_img.pixel_data = torch.from_numpy(patch_img.pixel_data).float()

        out = patch_img.clone()
        algorithm = FebdsAlgorithm("dog")
        algorithm(image=patch_img, output=out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == patch_img.pixel_data.shape

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_kmeans_per_patch_then_reconstruct(self, dicom_image):
        """Run KMeans on every non-uniform patch independently, replace
        patch pixel_data with the segmentation output, then reconstruct."""
        normalized = RegionOfInterest.normalize(dicom_image.clone(), divisor=4095.0)
        grid = PatchGrid(normalized, (64, 64))

        processed = 0
        for p in grid.patches:
            # Skip uniform patches (KMeans needs variance for k-means++)
            if float(p.pixel_data.float().std()) < 1e-6:
                continue
            patch_img = p.to_image()
            out = patch_img.clone()
            km = KMeansAlgorithm(k=2, device="cpu")
            km(patch_img, out)
            # Replace patch pixel_data with algorithm output
            p.pixel_data = out.pixel_data
            processed += 1

        assert processed > 0, "Expected at least one non-uniform patch"
        result = grid.to_image()
        assert result.pixel_data.shape == normalized.pixel_data.shape
