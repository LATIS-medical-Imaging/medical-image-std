"""
DICOM Windowing Algorithms.

Provides two windowing strategies as Algorithm subclasses:

1. **DicomWindowAlgorithm** — Simple linear WC/WW mapping (DICOM PS3 standard).
2. **GrailWindowAlgorithm** — Automatic intensity windowing using the GRAIL
   perceptual metric (Gabor-filtered mutual information optimisation).

Reference (GRAIL):
    Albiol, Corbi & Albiol (2017), "Automatic intensity windowing of
    mammographic images based on a perceptual metric," Medical Physics 44(4).

Example:
    >>> algo = DicomWindowAlgorithm(window_center=2000, window_width=3000)
    >>> output = image.clone()
    >>> algo(image, output)

    >>> algo = GrailWindowAlgorithm(n_scales=3, n_orientations=6)
    >>> output = image.clone()
    >>> algo(image, output)
    >>> print(algo.grail_a, algo.grail_b)
"""

from typing import Optional

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.mammography import MammographyPreprocessing


class DicomWindowAlgorithm(Algorithm):
    """
    Simple DICOM Window Center / Window Width algorithm.

    Maps pixel intensities to [0, 255] using the standard formula:
        output = clamp((pixel - (WC - WW/2)) / WW, 0, 1) * 255

    If WC/WW are not provided, they are read from the DICOM header.
    Falls back to the full dynamic range if unavailable.

    Args:
        window_center: Explicit window center (None = read from header).
        window_width: Explicit window width (None = read from header).
        device: Torch device.
    """

    def __init__(
        self,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        device: str = None,
    ):
        super().__init__(device=device)
        self.window_center = window_center
        self.window_width = window_width

        self._window = lambda img, out: MammographyPreprocessing.dicom_window(
            image=img,
            output=out,
            window_center=self.window_center,
            window_width=self.window_width,
            device=self.device,
        )

    def apply(self, image: Image, output: Image) -> Image:
        """
        Apply DICOM windowing to the image.

        Args:
            image: Input image (ideally DicomImage with dicom_data).
            output: Output Image — pixel_data will be in [0, 255].

        Returns:
            The output Image.
        """
        self._window(image, output)
        return output


class GrailWindowAlgorithm(Algorithm):
    """
    GRAIL automatic intensity windowing algorithm.

    Finds optimal lower (a) and upper (b) intensity bounds by maximising
    a Gabor-filtered mutual information metric between the 12-bit original
    and 8-bit windowed representations, then applies linear intensity
    windowing IW(i, a, b) → [0, 255].

    After ``apply()``, the optimal bounds are available as ``self.grail_a``
    and ``self.grail_b``.

    Reference:
        Albiol, Corbi & Albiol (2017), Medical Physics 44(4).

    Args:
        n_scales: Number of Gabor frequency scales (default 3).
        n_orientations: Number of Gabor orientations (default 6).
        delta: Initial search grid spacing (default 300).
        k_max: Maximum optimisation iterations (default 3).
        device: Torch device.
    """

    def __init__(
        self,
        n_scales: int = 3,
        n_orientations: int = 6,
        delta: int = 300,
        k_max: int = 3,
        device: str = None,
    ):
        super().__init__(device=device)
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.delta = delta
        self.k_max = k_max

        # Populated after apply()
        self.grail_a: Optional[float] = None
        self.grail_b: Optional[float] = None

    def apply(self, image: Image, output: Image) -> Image:
        """
        Apply GRAIL windowing to the image.

        Args:
            image: Input 12-bit mammogram image.
            output: Output Image — pixel_data will be in [0, 255].

        Returns:
            The output Image. ``self.grail_a`` and ``self.grail_b`` are set.
        """
        result = MammographyPreprocessing.grail_window(
            image=image,
            output=output,
            n_scales=self.n_scales,
            n_orientations=self.n_orientations,
            delta=self.delta,
            k_max=self.k_max,
            device=self.device,
        )
        self.grail_a = getattr(result, "grail_a", None)
        self.grail_b = getattr(result, "grail_b", None)
        return output
