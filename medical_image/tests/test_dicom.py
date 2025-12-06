import copy

import numpy as np
import pytest
import torch

from medical_image.algorithms.custom_algorithm import CustomAlgorithm
from medical_image.data.image import Image
from medical_image.data.patch import PatchGrid, Patch
from medical_image.process.threshold import Threshold
from medical_image.tests.mock_sample import mock_dicom_image, mock_sauvola_threshold
from medical_image.utils.image_utils import ImageExporter, ImageVisualizer


def split(image: Image, patch_size):
    img = image.pixel_data
    patch_h, patch_w = patch_size
    pad_bottom = 0
    pad_right = 0
    patches = []  # flat list
    grid = []  # 2D grid-like structure

    H, W = img.shape[-2], img.shape[-1]

    # Compute padding (same logic as divide_raster_band)
    if H % patch_h != 0:
        pad_bottom = patch_h - (H % patch_h)
    if W % patch_w != 0:
        pad_right = patch_w - (W % patch_w)

    if pad_bottom or pad_right:
        img = torch.nn.functional.pad(
            img, (0, pad_right, 0, pad_bottom), mode="constant", value=0
        )

    new_H, new_W = img.shape[-2], img.shape[-1]
    num_rows = new_H // patch_h
    num_cols = new_W // patch_w

    # Extract patches with coordinates
    for r in range(num_rows):
        row_list = []
        for c in range(num_cols):
            x = r * patch_h
            y = c * patch_w
            print("img.shape")
            print(img.shape)
            patch_tensor = img[:, x: x + patch_h, y: y + patch_w]

            is_padded = (r == num_rows - 1 and pad_bottom > 0) or (
                    c == num_cols - 1 and pad_right > 0
            )

            patch = Patch(
                parent=image,
                row_idx=r,
                col_idx=c,
                x=x,
                y=y,
                pixel_data=patch_tensor,
                is_padded=is_padded,
            )

            row_list.append(patch)
            patches.append(patch)
        grid.append(row_list)

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
        patchs = PatchGrid(dicom_image, (15,15))
        ImageVisualizer.show(patchs.patches[0].load())

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_patches_test(self, dicom_image):
        split(dicom_image, (3,3))
        # patchs = PatchGrid(dicom_image, (15,15))
        # ImageVisualizer.show(patchs.patches[0].load())

