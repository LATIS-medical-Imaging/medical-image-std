import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage._shared.filters import gaussian
from skimage.filters import difference_of_gaussians, threshold_otsu

# import torchvision.transforms.functional as F

from log_manager import logger
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.algorithms.custom_algorithm import CustomAlgorithm
from medical_image.data.dicom_image import DicomImage
from medical_image.data.image import Image

from medical_image.data.patch import PatchGrid
from medical_image.process.filters import Filters
from medical_image.process.morphology import MorphologyOperations
from medical_image.process.threshold import Threshold
from medical_image.tests.mock_sample import (
    mock_dicom_image,
    mock_sauvola_threshold,
    mock_png_image,
    mock_kernel,
    mock_two_sigmas,
    mock_median_size,
    mock_kernel_sizes,
)
from medical_image.utils.image_utils import ImageExporter, ImageVisualizer


def morphoogy_closing(input):
    return ndimage.binary_closing(input, structure=np.ones((7, 7))).astype(np.int64)


def region_fill(input):
    return ndimage.binary_fill_holes(input, structure=np.ones((7, 7))).astype(int)


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

    @pytest.mark.parametrize("size, sigma", mock_kernel())
    def test_gaussian_kernel_matches_skimage(self, size, sigma):
        """
        Tests that PyTorch Gaussian kernel convolution matches skimage Gaussian filter.
        """

        image = np.random.rand(8, 8).astype(np.float32)
        sigma = 1.5
        truncate = 4.0
        image_object = DicomImage.from_array(image)
        output = copy.deepcopy(image_object)

        Filters.gaussian_filter(image_object, output, sigma, truncate=truncate)

        skimage_result = gaussian(
            image, sigma=sigma, mode="nearest", truncate=truncate, preserve_range=False
        ).astype(np.float32)

        np.testing.assert_allclose(
            output.pixel_data, skimage_result, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_febds(self, dicom_image):
        """
        Tests that PyTorch Gaussian kernel convolution matches skimage Gaussian filter.
        """

        image = dicom_image.pixel_data
        sigma1, sigma2 = 1.7, 2.0
        skimage_result = difference_of_gaussians(image, sigma1, sigma2)
        skimage_result_finished = ndimage.median_filter(
            np.abs(skimage_result), size=(5, 5)
        )
        fi = np.power(skimage_result_finished / 4095.0, 1.25)
        fi = fi * 4095
        x = threshold_otsu(fi)
        out = np.zeros_like(fi)
        out[fi > x] = 1
        # print(out.shape)
        I = morphoogy_closing(out)
        #
        # fill = region_fill(I)

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        # Prepare output image
        output = copy.deepcopy(dicom_image)
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        # Apply the custom algorithm
        algorithm = FebdsAlgorithm("dog")
        algorithm(image=dicom_image, output=output)

        image_output = (
            output.pixel_data.detach().cpu().float().numpy().reshape((3328, 2560))
        )
        print(image_output.shape)
        # print()
        # Check that the pixel data has been modified
        assert not torch.allclose(fi, output.pixel_data.detach().cpu().float())

    @pytest.mark.parametrize("kernel_size", mock_kernel_sizes())
    def test_morphology_closing_matches_ndimage(self, kernel_size):
        """
        Test that PyTorch morphology_closing_torch matches ndimage.binary_closing exactly.
        """

        # Random binary image
        image = (np.random.rand(16, 16) > 0.5).astype(np.int64)

        image_object = DicomImage.from_array(image)
        output_object = copy.deepcopy(image_object)

        # Apply PyTorch closing
        MorphologyOperations.morphology_closing(
            image_object, output_object, kernel_size=kernel_size[0], device="cpu"
        )

        output_object.pixel_data = output_object.pixel_data.to(torch.int64)
        # Apply SciPy closing
        ndimage_result = ndimage.binary_closing(
            image, structure=np.ones((kernel_size[0], kernel_size[0]))
        ).astype(np.int64)

        # Assert exact match
        np.testing.assert_array_equal(output_object.pixel_data.numpy(), ndimage_result)

    @pytest.mark.parametrize("sigma1, sigma2", mock_two_sigmas())
    def test_DoG_matches_skimage(self, sigma1, sigma2):
        """
        Tests that PyTorch Gaussian kernel convolution matches skimage Gaussian filter.
        """

        image = np.random.rand(8, 8).astype(np.float32)
        truncate = 4.0
        image_object = DicomImage.from_array(image)
        output = copy.deepcopy(image_object)

        Filters.difference_of_gaussian(
            image_object, output, sigma1, sigma2, truncate=truncate
        )

        skimage_result = difference_of_gaussians(image, sigma1, sigma2)

        np.testing.assert_allclose(
            output.pixel_data, skimage_result, rtol=1e-4, atol=1e-4
        )

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
    def test_fedbs_algorithm(self, dicom_image):
        # Convert input pixel data to torch.Tensor if not already
        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()
        print("image path", dicom_image.file_path)
        # Prepare output image
        output = copy.deepcopy(dicom_image)
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        # Apply the custom algorithm
        algorithm = FebdsAlgorithm("dog")
        algorithm(image=dicom_image, output=output)

        # Check that the pixel data has been modified
        assert not torch.allclose(
            dicom_image.pixel_data.float(), output.pixel_data.detach().cpu().float()
        )
        ImageVisualizer.show(output)
        ImageVisualizer.compare(dicom_image, output)
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
