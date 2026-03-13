"""
Bit Depth Normalization Algorithm.

Auto-detects ``BitsStored`` from the DICOM header and normalizes pixel
values to a target range (default [0, 255]).

Pipeline:
    1. Detect bit depth from DICOM tag ``BitsStored`` (or infer from pixel range).
    2. Compute source max = 2^bits - 1.
    3. Linear map [0, source_max] → [0, target_max].

Example:
    >>> algo = BitDepthNormAlgorithm()         # auto-detect from header
    >>> output = image.clone()
    >>> algo(image, output)                    # output in [0, 255]

    >>> algo = BitDepthNormAlgorithm(bits_stored=12, target_max=1.0)
    >>> algo(image, output)                    # output in [0, 1]
"""

from typing import Optional

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.mammography import MammographyPreprocessing


class BitDepthNormAlgorithm(Algorithm):
    """
    Bit depth normalization algorithm for DICOM images.

    Detects bit depth automatically from the DICOM header (``BitsStored``)
    and normalizes pixel values from [0, 2^bits - 1] to [0, target_max].

    Args:
        bits_stored: Explicit bit depth override. If None, read from the
                     DICOM header or inferred from the pixel range.
        target_max: Upper bound of the output range (default 255.0).
        device: Torch device.
    """

    def __init__(
        self,
        bits_stored: Optional[int] = None,
        target_max: float = 255.0,
        device: str = None,
    ):
        super().__init__(device=device)
        self.bits_stored = bits_stored
        self.target_max = target_max

        self._normalize = lambda img, out: MammographyPreprocessing.normalize_bit_depth(
            image=img,
            output=out,
            bits_stored=self.bits_stored,
            target_max=self.target_max,
            device=self.device,
        )

    def apply(self, image: Image, output: Image) -> Image:
        """
        Normalize pixel values to the target range.

        Args:
            image: Input image (ideally DicomImage with dicom_data).
            output: Output Image — pixel_data in [0, target_max].

        Returns:
            The output Image.
        """
        self._normalize(image, output)
        return output