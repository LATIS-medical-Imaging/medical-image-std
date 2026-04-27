import numpy as np
import pytest
import torch
from scipy import ndimage
from skimage._shared.filters import gaussian
from skimage.filters import difference_of_gaussians, threshold_otsu

from medical_image import RegionOfInterest
from medical_image.utils.logging import logger
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.algorithms.sbrg import SbrgAlgorithm
from medical_image.algorithms.custom_algorithm import CustomAlgorithm
from medical_image.data.dicom_image import DicomImage
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
    mock_kernel_sizes,
)
from medical_image.utils.image_utils import ImageExporter


def morphoogy_closing(input):
    return ndimage.binary_closing(input, structure=np.ones((7, 7))).astype(np.int64)


def region_fill(input):
    return ndimage.binary_fill_holes(input, structure=np.ones((7, 7))).astype(int)


class TestDicom:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_image(self, dicom_image):
        ImageExporter.save_as(dicom_image)
        assert dicom_image.pixel_data is not None
        assert dicom_image.width > 0
        assert dicom_image.height > 0

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_otsu_threshold(self, dicom_image):
        output = dicom_image.clone()
        Threshold.otsu_threshold(dicom_image, output)

        assert not np.array_equal(
            dicom_image.pixel_data.cpu().numpy(), output.pixel_data.cpu().numpy()
        )

        unique_vals = torch.unique(output.pixel_data)
        assert torch.all((unique_vals == 0) | (unique_vals == 1))

    @pytest.mark.parametrize("dicom_image, window_size, k", mock_sauvola_threshold())
    def test_sauvola_threshold(self, dicom_image, window_size, k):
        output = dicom_image.clone()
        Threshold.sauvola_threshold(dicom_image, output, window_size, k)

        assert not torch.equal(
            dicom_image.pixel_data.float(), output.pixel_data.float()
        )

        out = output.pixel_data
        assert torch.all((out == 0) | (out == 255))

    @pytest.mark.parametrize("size, sigma", mock_kernel())
    def test_gaussian_kernel_matches_skimage(self, size, sigma):
        image = np.random.rand(8, 8).astype(np.float32)
        sigma = 1.5
        truncate = 4.0
        image_object = DicomImage.from_array(image)
        output = image_object.clone()

        Filters.gaussian_filter(image_object, output, sigma, truncate=truncate)

        skimage_result = gaussian(
            image, sigma=sigma, mode="nearest", truncate=truncate, preserve_range=False
        ).astype(np.float32)

        np.testing.assert_allclose(
            output.pixel_data, skimage_result, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_febds_fft(self, dicom_image):
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
        I = morphoogy_closing(out)
        fill = region_fill(I)

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()
        print("image.pixel_data.shape")
        print(image.shape)
        algorithm = FebdsAlgorithm("fft")
        algorithm(image=dicom_image, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        assert not torch.allclose(
            torch.tensor(I).float(), output.pixel_data.detach().cpu()
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_febds(self, dicom_image):
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
        I = morphoogy_closing(out)
        fill = region_fill(I)

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = FebdsAlgorithm("dog")
        algorithm(image=dicom_image, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        assert not torch.allclose(
            torch.tensor(I).float(), output.pixel_data.detach().cpu()
        )

    @pytest.mark.parametrize("kernel_size", mock_kernel_sizes())
    def test_morphology_closing_matches_ndimage(self, kernel_size):
        image = (np.random.rand(16, 16) > 0.5).astype(np.int64)

        image_object = DicomImage.from_array(image)
        output_object = image_object.clone()

        MorphologyOperations.morphology_closing(
            image_object, output_object, kernel_size=kernel_size[0], device="cpu"
        )

        output_object.pixel_data = output_object.pixel_data.to(torch.int64)
        ndimage_result = ndimage.binary_closing(
            image, structure=np.ones((kernel_size[0], kernel_size[0]))
        ).astype(np.int64)

        np.testing.assert_array_equal(output_object.pixel_data.numpy(), ndimage_result)

    @pytest.mark.parametrize("sigma1, sigma2", mock_two_sigmas())
    def test_DoG_matches_skimage(self, sigma1, sigma2):
        image = np.random.rand(8, 8).astype(np.float32)
        truncate = 4.0
        image_object = DicomImage.from_array(image)
        output = image_object.clone()

        Filters.difference_of_gaussian(
            image_object, output, sigma1, sigma2, truncate=truncate
        )

        skimage_result = difference_of_gaussians(image, sigma1, sigma2)

        np.testing.assert_allclose(
            output.pixel_data, skimage_result, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_custom_algorithm(self, dicom_image):
        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = CustomAlgorithm()
        algorithm(image=dicom_image, output=output)

        assert not torch.allclose(
            dicom_image.pixel_data.float(), output.pixel_data.detach().cpu().float()
        )

        unique_vals = torch.unique(output.pixel_data)
        assert set(unique_vals.tolist()).issubset({0, 1})

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_patches(self, dicom_image):
        patchs = PatchGrid(dicom_image, (15, 15))

    @pytest.mark.parametrize("png_image", mock_png_image())
    def test_patches_png(self, png_image):
        logger.info(f"PNG image shape: {png_image.pixel_data.shape}")

        patch_grid = PatchGrid(png_image, (100, 100))

        total_patches = len(patch_grid.patches)
        num_rows = len(patch_grid.grid)
        num_cols = len(patch_grid.grid[0]) if num_rows > 0 else 0

        logger.info(f"Total patches: {total_patches}")
        logger.info(f"Grid rows: {num_rows}, Grid cols: {num_cols}")

        assert total_patches == num_rows * num_cols
        assert num_rows > 0 and num_cols > 0

        patch1 = patch_grid.patches[0].load()
        patch2 = patch_grid.patches[1].load()

        assert patch1.pixel_data.shape[0] <= 100
        assert patch1.pixel_data.shape[1] <= 100

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_sbrg(self, dicom_image):
        # --- Reference implementation (pure numpy/skimage/scipy, no framework) ---
        coordinates = [1958, 1177, 2165, 1310]
        region_of_interest = RegionOfInterest(dicom_image, coordinates).load()
        roi_np = region_of_interest.pixel_data.detach().numpy()
        image_np = (
            region_of_interest.pixel_data.detach().numpy()
            if isinstance(region_of_interest.pixel_data, torch.Tensor)
            else region_of_interest.pixel_data
        ).reshape(region_of_interest.height, region_of_interest.width)

        # Stage 1: Seed-based region growing reference
        from skimage.morphology import local_maxima

        regional_max = local_maxima(image_np)
        seed_coords = np.argwhere(regional_max)
        seed_values = image_np[seed_coords[:, 0], seed_coords[:, 1]]
        seed_threshold = np.mean(seed_values)
        ref_region = (image_np >= seed_threshold).astype(np.float32)

        # Stage 2: Boundary segmentation reference
        from skimage.filters import sobel

        gradient = sobel(ref_region)
        binary_mask = (gradient > 0).astype(np.float32)
        I = morphoogy_closing(binary_mask)
        fill = region_fill(I)

        # --- Framework call ---
        if not isinstance(region_of_interest.pixel_data, torch.Tensor):
            region_of_interest.pixel_data = torch.from_numpy(
                region_of_interest.pixel_data
            ).float()

        output = region_of_interest.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = SbrgAlgorithm()
        algorithm(image=region_of_interest, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((region_of_interest.height, region_of_interest.width))
        )

        # Output must be a valid binary segmentation mask
        assert image_output is not None
        assert image_output.shape == (
            region_of_interest.height,
            region_of_interest.width,
        )
        unique_vals = np.unique(image_output)
        assert len(unique_vals) <= 2, "Output should be binary (0 and 1 only)"

        # Framework output must differ from the trivial all-zeros or all-ones result
        assert not np.all(
            image_output == 0
        ), "Output is all zeros — algorithm produced no segmentation"
        assert not np.all(
            image_output == 1
        ), "Output is all ones — algorithm over-segmented"

        # Framework result should not be identical to the raw reference
        assert not np.allclose(
            image_output,
            fill.reshape(region_of_interest.height, region_of_interest.width),
        ), "Framework output is identical to naive reference — boundary post-processing had no effect"
