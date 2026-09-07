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

import os
from typing import Optional, Union

import numpy as np
import pydicom
import torch

from medical_image.data.image import Image
from medical_image.utils.ErrorHandler import ErrorMessages
from medical_image.utils.image_utils import TensorConverter


class DicomImage(Image):
    """DICOM image backed by *pydicom*.

    Supports lazy loading: the constructor stores the file path and
    validates the extension; pixel data is read only when :meth:`load`
    is called.

    Attributes:
        dicom_data (Optional[pydicom.Dataset]): The parsed DICOM dataset
            (``None`` until :meth:`load` is called).
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        array: Optional[Union[np.ndarray, torch.Tensor]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_image: Optional[Image] = None,
    ):
        """Initialise a DICOM image.

        Args:
            file_path: Path to a ``.dcm`` file.
            array: Pre-existing pixel data (numpy or tensor).
            width: Explicit width hint.
            height: Explicit height hint.
            source_image: Another Image to clone from.

        Raises:
            ValueError: If *file_path* does not have a ``.dcm`` extension.
        """
        super().__init__(
            file_path=file_path,
            array=array,
            width=width,
            height=height,
            source_image=source_image,
        )

        if file_path:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext != ".dcm":
                raise ErrorMessages.unsupported_file_type(ext)
        self.dicom_data = None

    def load(self) -> None:
        """Read the DICOM file and populate ``pixel_data``, ``width``, and ``height``."""
        self.dicom_data = pydicom.dcmread(self.file_path)
        self.pixel_data = torch.tensor(self.dicom_data.pixel_array)
        self._width = self.dicom_data.Columns
        self._height = self.dicom_data.Rows
        self._post_load()

    def save(self) -> None:
        """Write modified pixel data back to ``{name}_modified.dcm``.

        Raises:
            ValueError: If ``dicom_data`` has not been loaded.
        """
        if self.dicom_data is None:
            raise ErrorMessages.dicom_data_not_loaded()
        filename, extension = os.path.splitext(self.file_path)
        self.dicom_data.set_pixel_data(
            TensorConverter.to_numpy(self), "MONOCHROME2", 16
        )
        self.dicom_data.save_as(filename + "_modified.dcm")

    def __repr__(self) -> str:
        """Return a one-line summary of the DICOM image."""
        status = "loaded" if self.pixel_data is not None else "unloaded"
        return (
            f"DicomImage(path='{self.file_path}', "
            f"{self.width}x{self.height}, {status})"
        )
