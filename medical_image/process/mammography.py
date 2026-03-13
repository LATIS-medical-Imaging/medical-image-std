"""
Mammography Preprocessing Module.

Provides GPU-accelerated preprocessing operations specific to mammogram images:

- **Breast Region Masking**: Otsu thresholding + largest connected component.
  Reference: Nguyen et al. (2025), "A Robust Approach for Breast Cancer
  Classification from DICOM Images," ETASR Vol. 15, No. 3.

- **DICOM Windowing (WC/WW)**: Simple linear windowing and the GRAIL algorithm.
  Reference: Albiol, Corbi & Albiol (2017), "Automatic intensity windowing of
  mammographic images based on a perceptual metric," Medical Physics 44(4).

- **Bit Depth Normalization**: Auto-detect BitsStored from DICOM header and
  normalize pixel values to [0, 255].

All methods follow the library's standard patterns:
    - Static methods decorated with ``@requires_loaded``
    - ``device=None`` with ``resolve_device()`` for automatic GPU inference
    - ``(image, output)`` parameter convention
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from medical_image.data.image import Image, requires_loaded
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.metrics import Metrics
from medical_image.utils.device import resolve_device


class MammographyPreprocessing:
    """Static preprocessing methods for mammogram images."""

    # ------------------------------------------------------------------
    # 1. Breast Region Masking
    # ------------------------------------------------------------------

    @staticmethod
    @requires_loaded
    def breast_mask(
        image: Image,
        output: Image = None,
        device=None,
    ) -> Image:
        """
        Extract the breast region from a mammogram background.

        Uses Otsu thresholding followed by largest connected component
        selection to produce a binary mask of the breast area.

        Reference:
            Nguyen et al. (2025), "A Robust Approach for Breast Cancer
            Classification from DICOM Images," ETASR Vol. 15, No. 3.

        Algorithm:
            1. Apply Otsu threshold to binarize the image.
            2. Find connected components in the binary image.
            3. Select the largest connected component (breast region).
            4. Return the binary mask.

        Args:
            image: Input mammogram image.
            output: Optional output Image for the masked result.
                    If None, a new InMemoryImage is created.
            device: Computation device (None = infer from image).

        Returns:
            Image with pixel_data set to the breast mask (uint8, 0/1).
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()

        # --- Otsu threshold ---
        mask = MammographyPreprocessing._otsu_binary(img, device)

        # --- Largest connected component ---
        mask = MammographyPreprocessing._largest_connected_component(mask, device)

        if output is None:
            output = InMemoryImage(array=mask)
        else:
            output.pixel_data = mask
        return output

    @staticmethod
    @requires_loaded
    def apply_breast_mask(
        image: Image,
        output: Image = None,
        device=None,
    ) -> Image:
        """
        Mask a mammogram so that only the breast region is retained.

        Computes the breast mask via :meth:`breast_mask` and multiplies it
        with the original pixel data, setting background pixels to zero.

        Args:
            image: Input mammogram image.
            output: Optional output Image for the masked image.
            device: Computation device (None = infer from image).

        Returns:
            Image with background pixels zeroed out.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()

        mask_img = MammographyPreprocessing.breast_mask(image, device=device)
        mask = mask_img.pixel_data.to(device).float()

        masked = img * mask

        if output is None:
            output = InMemoryImage(array=masked)
        else:
            output.pixel_data = masked
        return output

    # ------------------------------------------------------------------
    # 2. DICOM Windowing (WC/WW)
    # ------------------------------------------------------------------

    @staticmethod
    @requires_loaded
    def dicom_window(
        image: Image,
        output: Image = None,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        device=None,
    ) -> Image:
        """
        Apply DICOM Window Center / Window Width (WC/WW) transformation.

        Maps pixel intensities from the diagnostic window to [0, 255]
        using the standard DICOM PS3 formula:

            output = clamp((pixel - (WC - WW/2)) / WW, 0, 1) * 255

        If ``window_center`` or ``window_width`` are not provided, they are
        read from the DICOM header (``image.dicom_data``). If the header
        also lacks them, the full dynamic range of the image is used.

        Args:
            image: Input image (ideally a DicomImage with ``dicom_data``).
            output: Optional output Image.
            window_center: Explicit window center override.
            window_width: Explicit window width override.
            device: Computation device (None = infer from image).

        Returns:
            Image with pixel_data in [0, 255] float32.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()

        wc, ww = MammographyPreprocessing._resolve_wc_ww(
            image, window_center, window_width, img
        )

        lower = wc - ww / 2.0
        result = ((img - lower) / ww).clamp(0.0, 1.0) * 255.0

        if output is None:
            output = InMemoryImage(array=result)
        else:
            output.pixel_data = result
        return output

    @staticmethod
    @requires_loaded
    def grail_window(
        image: Image,
        output: Image = None,
        n_scales: int = 3,
        n_orientations: int = 6,
        delta: int = 300,
        k_max: int = 3,
        device=None,
    ) -> Image:
        """
        GRAIL algorithm for automatic intensity windowing of mammograms.

        Finds optimal lower (*a*) and upper (*b*) intensity bounds by
        maximising a perceptual quality metric based on Gabor-filtered
        mutual information between the 12-bit original and 8-bit windowed
        representations.

        Reference:
            Albiol, Corbi & Albiol (2017), "Automatic intensity windowing of
            mammographic images based on a perceptual metric," Medical Physics
            44(4).

        Algorithm:
            1. Compute Gabor filter bank responses on the original image.
            2. Iteratively optimise *b* (upper bound) then *a* (lower bound)
               by evaluating MI between original and windowed Gabor responses.
            3. Refine the search grid each iteration (delta /= 10).
            4. Apply final IW(i, a, b) to produce [0, 255] output.

        Args:
            image: Input 12-bit mammogram image.
            output: Optional output Image.
            n_scales: Number of Gabor frequency scales (default 3).
            n_orientations: Number of Gabor orientations (default 6).
            delta: Initial search grid spacing (default 300).
            k_max: Maximum optimisation iterations (default 3).
            device: Computation device (None = infer from image).

        Returns:
            Image with pixel_data in [0, 255] float32. The optimal *a* and *b*
            values are stored as ``output.grail_a`` and ``output.grail_b``.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()

        i_min = img.min().item()
        i_max = img.max().item()

        # --- Build Gabor filter bank ---
        gabor_kernels = MammographyPreprocessing._build_gabor_bank(
            n_scales, n_orientations, device
        )

        # --- Compute Gabor responses on original image ---
        orig_responses = MammographyPreprocessing._gabor_responses(
            img, gabor_kernels, device
        )

        # --- Iterative optimisation ---
        a = i_min
        b = i_max

        for k in range(k_max):
            # Optimise b with a fixed
            b = MammographyPreprocessing._optimise_bound(
                img, orig_responses, gabor_kernels, a, b, delta,
                optimise_upper=True, device=device,
            )

            # Optimise a with b fixed
            a = MammographyPreprocessing._optimise_bound(
                img, orig_responses, gabor_kernels, a, b, delta,
                optimise_upper=False, device=device,
            )

            delta = max(delta // 10, 1)

        # --- Apply final windowing ---
        result = MammographyPreprocessing._intensity_window(img, a, b)

        if output is None:
            output = InMemoryImage(array=result)
        else:
            output.pixel_data = result
        output.grail_a = a
        output.grail_b = b
        return output

    # ------------------------------------------------------------------
    # 3. Bit Depth Normalization
    # ------------------------------------------------------------------

    @staticmethod
    @requires_loaded
    def normalize_bit_depth(
        image: Image,
        output: Image = None,
        bits_stored: Optional[int] = None,
        target_max: float = 255.0,
        device=None,
    ) -> Image:
        """
        Normalize pixel values based on the DICOM ``BitsStored`` tag.

        Automatically detects the bit depth from the DICOM header instead
        of hardcoding (e.g. 4095). Maps values from ``[0, 2^bits - 1]``
        to ``[0, target_max]``.

        Args:
            image: Input image (ideally a DicomImage with ``dicom_data``).
            output: Optional output Image.
            bits_stored: Explicit bit depth override. If None, read from
                         the DICOM header. Falls back to inferring from
                         the maximum pixel value.
            target_max: Upper bound of the output range (default 255.0).
            device: Computation device (None = infer from image).

        Returns:
            Image with pixel_data in [0, target_max] float32.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()

        if bits_stored is None:
            bits_stored = MammographyPreprocessing._detect_bits_stored(image)

        source_max = float((1 << bits_stored) - 1)  # 2^bits - 1
        result = (img / source_max).clamp(0.0, 1.0) * target_max

        if output is None:
            output = InMemoryImage(array=result)
        else:
            output.pixel_data = result
        return output

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _otsu_binary(img: torch.Tensor, device) -> torch.Tensor:
        """Compute Otsu threshold on *img* and return a uint8 0/1 mask."""
        min_val = img.min()
        max_val = img.max()
        bins = 256 if max_val <= 255 else 4096

        hist = torch.histc(img, bins=bins, min=min_val.item(), max=max_val.item())
        bin_centers = torch.linspace(min_val, max_val, steps=bins, device=device)

        weight1 = torch.cumsum(hist, dim=0)
        weight2 = hist.sum() - weight1
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / weight1.clamp(min=1e-6)
        mean2_cum = torch.cumsum(hist * bin_centers, dim=0)
        mean2 = ((hist * bin_centers).sum() - mean2_cum) / weight2.clamp(min=1e-6)

        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        threshold_idx = torch.argmax(variance_between)
        threshold_value = bin_centers[threshold_idx]

        return (img > threshold_value).to(torch.uint8)

    @staticmethod
    def _largest_connected_component(
        mask: torch.Tensor, device
    ) -> torch.Tensor:
        """
        Select the largest connected component from a binary mask.

        Uses iterative flood-fill labelling implemented in pure PyTorch.
        For each unlabelled foreground pixel, a dilation-based flood fill
        expands the region, and the largest region by pixel count is kept.
        """
        mask_bool = mask.bool()
        labelled = torch.zeros_like(mask, dtype=torch.int32, device=device)
        current_label = 0
        remaining = mask_bool.clone()

        while remaining.any():
            # Pick the first unlabelled foreground pixel
            ys, xs = torch.where(remaining)
            seed = torch.zeros_like(mask_bool)
            seed[ys[0], xs[0]] = True

            # Flood-fill via iterative dilation within the mask
            kernel = torch.ones(1, 1, 3, 3, device=device)
            prev = seed
            while True:
                dilated = (
                    F.conv2d(
                        prev.float().unsqueeze(0).unsqueeze(0),
                        kernel,
                        padding=1,
                    ).squeeze()
                    > 0
                )
                dilated = dilated & mask_bool
                if torch.equal(dilated, prev):
                    break
                prev = dilated

            current_label += 1
            labelled[prev] = current_label
            remaining = remaining & ~prev

        if current_label == 0:
            return mask

        # Find the label with the most pixels
        counts = torch.bincount(labelled.flatten())
        # Ignore label 0 (background)
        counts[0] = 0
        best_label = counts.argmax().item()

        return (labelled == best_label).to(torch.uint8)

    @staticmethod
    def _resolve_wc_ww(
        image: Image,
        wc: Optional[float],
        ww: Optional[float],
        img: torch.Tensor,
    ) -> Tuple[float, float]:
        """Resolve Window Center / Width from explicit args, DICOM header, or image range."""
        if wc is not None and ww is not None:
            return float(wc), float(ww)

        # Try DICOM header
        dicom = getattr(image, "dicom_data", None)
        if dicom is not None:
            try:
                header_wc = dicom.WindowCenter
                header_ww = dicom.WindowWidth
                # pydicom may return a list for multi-frame
                if isinstance(header_wc, (list, pydicom.multival.MultiValue)):
                    header_wc = header_wc[0]
                if isinstance(header_ww, (list, pydicom.multival.MultiValue)):
                    header_ww = header_ww[0]
                return float(header_wc), float(header_ww)
            except (AttributeError, TypeError):
                pass

        # Fallback: full dynamic range
        i_min = img.min().item()
        i_max = img.max().item()
        return (i_min + i_max) / 2.0, float(i_max - i_min) or 1.0

    @staticmethod
    def _detect_bits_stored(image: Image) -> int:
        """Detect bit depth from DICOM header or infer from pixel range."""
        dicom = getattr(image, "dicom_data", None)
        if dicom is not None:
            try:
                return int(dicom.BitsStored)
            except (AttributeError, TypeError):
                pass

        # Infer from max pixel value
        max_val = image.pixel_data.max().item()
        if max_val <= 255:
            return 8
        if max_val <= 4095:
            return 12
        if max_val <= 65535:
            return 16
        return 16

    @staticmethod
    def _intensity_window(
        img: torch.Tensor, a: float, b: float
    ) -> torch.Tensor:
        """Linear mapping from [a, b] to [0, 255], clamped."""
        span = b - a
        if span <= 0:
            span = 1.0
        return ((img - a) / span).clamp(0.0, 1.0) * 255.0

    # --- Gabor filter helpers for GRAIL ---

    @staticmethod
    def _build_gabor_bank(
        n_scales: int, n_orientations: int, device
    ) -> list:
        """
        Build a bank of 2-D Gabor kernels.

        Frequencies: f_m  in  {f_max / (sqrt(2))^m} for m in 0..n_scales-1
        Orientations: theta_n  in  {n * pi / n_orientations}
        """
        f_max = 0.25
        gamma_val = math.sqrt(2)

        kernels = []
        for m in range(n_scales):
            f_m = f_max / (gamma_val ** m)
            sigma = 1.0 / (2.0 * f_m)
            # kernel size: 6*sigma, rounded to nearest odd
            ksize = int(math.ceil(6 * sigma))
            if ksize % 2 == 0:
                ksize += 1
            half = ksize // 2

            for n in range(n_orientations):
                theta = n * math.pi / n_orientations
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(-half, half + 1, dtype=torch.float32, device=device),
                    torch.arange(-half, half + 1, dtype=torch.float32, device=device),
                    indexing="ij",
                )
                x_prime = x_coords * math.sin(theta) + y_coords * math.cos(theta)
                gauss = torch.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * sigma ** 2))
                real = gauss * torch.cos(2 * math.pi * f_m * x_prime)
                kernels.append(real)

        return kernels

    @staticmethod
    def _gabor_responses(
        img: torch.Tensor, kernels: list, device
    ) -> list:
        """Apply each Gabor kernel to *img* and return magnitude responses."""
        img4d = img.unsqueeze(0).unsqueeze(0)
        responses = []
        for k in kernels:
            k4d = k.unsqueeze(0).unsqueeze(0)
            pad_h = k.shape[0] // 2
            pad_w = k.shape[1] // 2
            resp = F.conv2d(
                F.pad(img4d, (pad_w, pad_w, pad_h, pad_h), mode="replicate"),
                k4d,
            )
            responses.append(resp.squeeze().abs())
        return responses

    @staticmethod
    def _gabor_mutual_information(
        orig_responses: list,
        windowed_responses: list,
    ) -> float:
        """Sum of MI between corresponding Gabor responses."""
        total_mi = 0.0
        for orig_r, win_r in zip(orig_responses, windowed_responses):
            # Quantise to integer bins for histogram
            o = orig_r.flatten()
            w = win_r.flatten()
            # Normalise both to 0-255 range for stable MI
            o_min, o_max = o.min(), o.max()
            w_min, w_max = w.min(), w.max()
            if o_max - o_min > 0:
                o = ((o - o_min) / (o_max - o_min) * 255).long()
            else:
                o = torch.zeros_like(o, dtype=torch.long)
            if w_max - w_min > 0:
                w = ((w - w_min) / (w_max - w_min) * 255).long()
            else:
                w = torch.zeros_like(w, dtype=torch.long)

            # Joint histogram (256 x 256)
            joint = torch.zeros(256, 256, device=o.device)
            o = o.clamp(0, 255)
            w = w.clamp(0, 255)
            idx = o * 256 + w
            joint.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            joint = joint / joint.sum()

            p_o = joint.sum(dim=1)
            p_w = joint.sum(dim=0)
            nz = joint > 0
            mi = (joint[nz] * torch.log2(joint[nz] / (p_o.unsqueeze(1).expand_as(joint)[nz] * p_w.unsqueeze(0).expand_as(joint)[nz]).clamp(min=1e-12))).sum()
            total_mi += mi.item()
        return total_mi

    @staticmethod
    def _optimise_bound(
        img: torch.Tensor,
        orig_responses: list,
        gabor_kernels: list,
        a: float,
        b: float,
        delta: int,
        optimise_upper: bool,
        device,
    ) -> float:
        """Search for the best upper (*b*) or lower (*a*) bound."""
        current = b if optimise_upper else a

        candidates = []
        for offset in range(-5, 6):
            c = current + offset * delta
            candidates.append(c)

        best_val = -float("inf")
        best_c = current

        for c in candidates:
            if optimise_upper:
                if c <= a:
                    continue
                windowed = MammographyPreprocessing._intensity_window(img, a, c)
            else:
                if c >= b:
                    continue
                windowed = MammographyPreprocessing._intensity_window(img, c, b)

            win_responses = MammographyPreprocessing._gabor_responses(
                windowed, gabor_kernels, device
            )
            mi = MammographyPreprocessing._gabor_mutual_information(
                orig_responses, win_responses
            )
            if mi > best_val:
                best_val = mi
                best_c = c

        return best_c


# Convenience alias — avoid importing private module name
try:
    import pydicom  # noqa: F811  (used in _resolve_wc_ww)
except ImportError:
    pass