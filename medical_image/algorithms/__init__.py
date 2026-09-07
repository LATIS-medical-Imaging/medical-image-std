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
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.algorithms.pfcm import PFCMAlgorithm
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.algorithms.breast_mask import BreastMaskAlgorithm
from medical_image.algorithms.dicom_window import (
    DicomWindowAlgorithm,
    GrailWindowAlgorithm,
)
from medical_image.algorithms.bit_depth_norm import BitDepthNormAlgorithm
from medical_image.algorithms.sbrg import SbrgAlgorithm
from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm

__all__ = [
    "Algorithm",
    "FebdsAlgorithm",
    "FCMAlgorithm",
    "PFCMAlgorithm",
    "TopHatAlgorithm",
    "KMeansAlgorithm",
    "BreastMaskAlgorithm",
    "DicomWindowAlgorithm",
    "GrailWindowAlgorithm",
    "BitDepthNormAlgorithm",
    "SbrgAlgorithm",
    "DeepSegmentationAlgorithm",
]
