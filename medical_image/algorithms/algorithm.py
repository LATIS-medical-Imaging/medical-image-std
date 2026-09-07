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

from abc import ABC, abstractmethod
from typing import List

import torch

from medical_image.data.image import Image
from medical_image.utils.device import Precision


class Algorithm(ABC):
    def __init__(self, device: str = None, precision: Precision = Precision.FULL):
        super().__init__()
        self.device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.precision = precision

    @abstractmethod
    def apply(self, image: Image, output: Image) -> Image:
        """
        Apply the defined operations to the input image.

        Args:
            image: The input image.
            output: The output image to store results.

        Returns:
            The output image after applying the operations.
        """
        pass

    def __call__(self, image: Image, output: Image) -> Image:
        if self.precision == Precision.FULL:
            self.apply(image, output)
            return output

        # torch.autocast, not the torch.cuda.amp alias: the latter is deprecated
        # and CUDA-only, while bf16 on CPU is exactly where this is worth having
        # when there is no GPU to fall back on.
        device_type = torch.device(self.device).type
        with torch.autocast(device_type=device_type, dtype=self.precision.value):
            self.apply(image, output)
        return output

    def apply_batch(self, images: List[Image], outputs: List[Image]) -> List[Image]:
        """
        Process a batch of images. Default: loop over apply().
        Subclasses can override for truly batched GPU processing.
        """
        for img, out in zip(images, outputs):
            self.apply(img, out)
        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}', precision={self.precision.name})"
