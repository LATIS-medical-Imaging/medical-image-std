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
from medical_image.process.filters import Filters
from medical_image.process.threshold import Threshold


class CustomAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.gaussian = lambda img, output: Filters.gaussian_filter(
            image=img, output=output, sigma=2.0, device=self.device
        )
        self.otsu = lambda img, output: Threshold.otsu_threshold(
            image=img, output=output, device=self.device
        )

    def apply(self, image: Image, output: Image) -> Image:
        self.gaussian(img=image, output=output)
        self.gaussian(img=output, output=output)
        self.otsu(img=output, output=output)
        return output
