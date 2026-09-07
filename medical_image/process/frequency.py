# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 [YOUR NAME / ORGANIZATION]
#
# This file is part of medical-image-std.
#
# medical-image-std is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# medical-image-std is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with medical-image-std. If not, see
# <https://www.gnu.org/licenses/>.

import torch

from medical_image.data.image import Image, requires_loaded
from medical_image.utils.device import resolve_device


class FrequencyOperations:
    @staticmethod
    @requires_loaded
    def fft(image: Image, output: Image, device=None) -> Image:
        """
        Computes the 2-dimensional Fast Fourier Transform (FFT) of an image.

        Args:
            image: Input image.
            output: Output image to store the complex FFT result.
            device: Device to perform computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()
        fft_result = torch.fft.fft2(img)
        output.pixel_data = fft_result.to(device)
        return output

    @staticmethod
    @requires_loaded
    def inverse_fft(image: Image, output: Image, device=None) -> Image:
        """
        Computes the inverse 2-dimensional Fast Fourier Transform (IFFT) of an image.

        Args:
            image: Input image in the frequency domain (complex tensor).
            output: Output image to store the inverse FFT result.
            device: Device to perform computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device)
        ifft_result = torch.fft.ifft2(img)
        output.pixel_data = ifft_result.to(device)
        return output
