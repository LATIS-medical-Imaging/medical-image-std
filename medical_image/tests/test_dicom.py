import copy

import numpy as np
import pytest
import torch

from log_manager import logger
from medical_image.algorithms.custom_algorithm import CustomAlgorithm
from medical_image.data.image import Image
from medical_image.data.patch import PatchGrid, Patch
from medical_image.process.threshold import Threshold
from medical_image.tests.mock_sample import (
    mock_dicom_image,
    mock_sauvola_threshold,
    mock_png_image,
)
from medical_image.utils.image_utils import ImageExporter, ImageVisualizer


class TestDicom:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_image(self, dicom_image):
        ImageExporter.save_as(dicom_image)
        assert dicom_image.pixel_data is not None
        assert dicom_image.width == 512
        assert dicom_image.height == 512

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_otsu_threshold(self, dicom_image):
        # Apply Otsu's threshold to the mock DICOM image
        output = copy.deepcopy(dicom_image)
        Threshold.otsu_threshold(dicom_image, output)

        # Check that the pixel data has been modified
        assert not np.array_equal(
            dicom_image.pixel_data.cpu().numpy(), output.pixel_data.cpu().numpy()
        )

        assert torch.all((output.pixel_data == 0) | (output.pixel_data == 255))

    @pytest.mark.parametrize("dicom_image, window_size, k", mock_sauvola_threshold())
    def test_sauvola_threshold(self, dicom_image, window_size, k):
        output = copy.deepcopy(dicom_image)
        Threshold.sauvola_threshold(dicom_image, output, window_size, k)

        # Check that pixel data has changed
        assert not torch.equal(dicom_image.pixel_data, output.pixel_data)

        # Check that output is binary (0 or 255)
        out = output.pixel_data
        assert torch.all((out == 0) | (out == 255))

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_custom_algorithm(self, dicom_image):
        # Convert input pixel data to torch.Tensor if not already
        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        # Prepare output image
        output = copy.deepcopy(dicom_image)
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        # Apply the custom algorithm
        algorithm = CustomAlgorithm()
        algorithm(image=dicom_image, output=output)

        # Check that the pixel data has been modified
        assert not torch.allclose(
            dicom_image.pixel_data.float(), output.pixel_data.detach().cpu().float()
        )

        # Check that the output is a binary image (0 or 255)
        unique_vals = torch.unique(output.pixel_data)
        assert set(unique_vals.tolist()).issubset({0, 255})

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_patches(self, dicom_image):
        patchs = PatchGrid(dicom_image, (15, 15))
        # ImageVisualizer.show(patchs.patches[0].load())

    @pytest.mark.parametrize("png_image", mock_png_image())
    def test_patches_png(self, png_image):
        logger.info(f"PNG image shape: {png_image.pixel_data.shape}")

        patch_grid = PatchGrid(png_image, (100, 100))

        # Number of patches
        total_patches = len(patch_grid.patches)
        num_rows = len(patch_grid.grid)
        num_cols = len(patch_grid.grid[0]) if num_rows > 0 else 0

        logger.info(f"Total patches: {total_patches}")
        logger.info(f"Grid rows: {num_rows}, Grid cols: {num_cols}")

        # Basic assertions
        assert total_patches == num_rows * num_cols
        assert num_rows > 0 and num_cols > 0

        # Check first two patches shapes
        patch1 = patch_grid.patches[0].load()
        patch2 = patch_grid.patches[1].load()
        print("Patch 1 shape:", patch1.pixel_data.shape)
        print("Patch 2 shape:", patch2.pixel_data.shape)

        # Ensure patch sizes are correct or <= patch_size for padded edges
        assert patch1.pixel_data.shape[0] <= 100
        assert patch1.pixel_data.shape[1] <= 100

    # patchs = PatchGrid(dicom_image, (15,15))
    # ImageVisualizer.show(patchs.patches[0].load())
