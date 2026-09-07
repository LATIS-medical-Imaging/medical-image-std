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

import numpy as np
import torch
from PIL import Image as PILImage

from medical_image.data.image import Image
from medical_image.utils.ErrorHandler import ErrorMessages


class PNGImage(Image):
    """PNG image backed by *Pillow*.

    Supports lazy loading: the constructor validates the file extension;
    pixel data is read only when :meth:`load` is called.

    Attributes:
        _pil_image (Optional[PIL.Image.Image]): The Pillow image object
            (``None`` until :meth:`load` is called).
    """

    def __init__(self, file_path: str):
        """Initialise a PNG image.

        Args:
            file_path: Path to a ``.png`` file.

        Raises:
            ValueError: If *file_path* does not have a ``.png`` extension.
        """
        super().__init__(file_path)
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in [".png"]:
            raise ErrorMessages.unsupported_file_type(ext)

        self._pil_image = None

    def load(self) -> None:
        """Open the PNG file via Pillow and populate ``pixel_data`` as a float tensor."""
        self._pil_image = PILImage.open(self.file_path)
        img = np.array(self._pil_image)
        self.pixel_data = torch.from_numpy(img).float()
        self._post_load()

    def save(self) -> None:
        """Write pixel data to ``{name}_modified.png`` as uint8.

        Raises:
            DicomDataNotLoadedError: If ``pixel_data`` is ``None``.
        """
        if self.pixel_data is None:
            raise ErrorMessages.dicom_data_not_loaded()

        img_np = self.pixel_data.detach().cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        filename, _ = os.path.splitext(self.file_path)
        out_path = filename + "_modified.png"
        PILImage.fromarray(img_np).save(out_path)
