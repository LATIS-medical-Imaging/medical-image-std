from typing import Optional, Union

import numpy as np
import torch

from medical_image.data.image import Image


class InMemoryImage(Image):
    """Concrete image that lives only in memory (no file I/O)."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        array: Optional[Union[np.ndarray, torch.Tensor]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_image: Optional[Image] = None,
    ):
        super().__init__(
            file_path=file_path,
            array=array,
            width=width,
            height=height,
            source_image=source_image,
        )

    def load(self):
        pass

    def save(self):
        pass
