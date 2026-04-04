"""
Breast Region Masking Algorithm.

Extracts the breast area from mammogram background using Otsu thresholding
followed by largest connected component selection.

Reference:
    Nguyen et al. (2025), "A Robust Approach for Breast Cancer Classification
    from DICOM Images," ETASR Vol. 15, No. 3.

Pipeline:
    1. Otsu threshold → binary mask.
    2. Largest connected component selection → breast region.
    3. Multiply mask with original image → masked output.

Example:
    >>> algo = BreastMaskAlgorithm(device="cpu")
    >>> output = image.clone()
    >>> algo(image, output)
"""

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.mammography import MammographyPreprocessing


class BreastMaskAlgorithm(Algorithm):
    """
    Breast region masking algorithm for mammograms.

    Uses Otsu thresholding + largest connected component to isolate the
    breast from the background, then applies the mask to the original image.

    Args:
        mask_only: If True, output contains the binary mask (0/1) instead
                   of the masked image. Default False.
        device: Torch device (e.g. "cpu", "cuda:0").
    """

    def __init__(self, mask_only: bool = False, device: str = None):
        super().__init__(device=device)
        self.mask_only = mask_only

        self._breast_mask = lambda img, out: MammographyPreprocessing.breast_mask(
            image=img, output=out, device=self.device
        )
        self._apply_mask = lambda img, out: MammographyPreprocessing.apply_breast_mask(
            image=img, output=out, device=self.device
        )

    def apply(self, image: Image, output: Image) -> Image:
        """
        Apply breast region masking.

        Args:
            image: Input mammogram.
            output: Output Image — will contain either the binary mask
                    (if ``mask_only=True``) or the masked mammogram.

        Returns:
            The output Image.
        """
        if self.mask_only:
            self._breast_mask(image, output)
        else:
            self._apply_mask(image, output)
        return output
