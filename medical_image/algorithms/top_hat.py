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

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.morphology import MorphologyOperations


class TopHatAlgorithm(Algorithm):
    """
    White Top-Hat enhancement algorithm for microcalcification detection.

    Highlights bright structures (e.g., microcalcifications) that are smaller than
    the selected structuring element.

    Math and Logic:
        TopHat(I) = I - opening(I, SE)

    Pipeline:
        1. Create a disk structuring element of the specified radius.
        2. Perform morphological opening (erosion followed by dilation).
        3. Subtract the opened image from the original image.

    Args:
        radius: Disk SE radius (default 4 -> 9x9 footprint).
        device: Torch device (e.g. "cpu", "cuda:0").
    """

    def __init__(self, radius: int = 4, device: str = "cpu"):
        super().__init__(device=device)
        self.radius = radius

        self.top_hat = lambda img, out: MorphologyOperations.white_top_hat(
            image=img, output=out, radius=self.radius, device=self.device
        )

    def apply(self, image: Image, output: Image) -> Image:
        """
        Apply white top-hat to the image.

        Args:
            image: Input Image (2D float, e.g. normalized [0,1]).
            output: Output Image — pixel_data will contain top-hat result.

        Returns:
            The output Image.
        """
        self.top_hat(image, output)
        return output
