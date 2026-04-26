import numpy as np
import torch
from skimage.filters import sobel
from skimage.morphology import local_maxima

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.morphology import MorphologyOperations
from medical_image.utils.device import resolve_device


class SbrgAlgorithm(Algorithm):
    """Seed-Based Region Growing (SBRG) microcalcification segmentation.

    Two-stage algorithm: seed-based region growing followed by boundary
    segmentation using mathematical morphology.

    References:
        Malek, R. et al. (2010). "Region and Boundary Segmentation of
        Microcalcifications using Seed-Based Region Growing and Mathematical
        Morphology."
    """

    def __init__(self, device=None):
        super().__init__(device=device)

        self.find_seed = lambda img, out: self._find_seed(img, out)
        self.region_grow = lambda img, out: self._region_grow(img, out)
        self.boundary_seg = lambda img, out: self._boundary_seg(img, out)

    def _find_seed(self, image: Image, output: Image) -> Image:
        """Stage 1a: Identify seed threshold from regional maxima."""
        device = resolve_device(image, explicit=self.device)
        img_np = image.pixel_data.detach().cpu().numpy().astype(np.float64)
        while img_np.ndim > 2:
            img_np = img_np.squeeze(0)

        # Step 1: Find regional maxima (8-connected)
        reg_max_mask = local_maxima(img_np)

        # Step 2: Extract intensity values at regional maxima
        seed_values = img_np[reg_max_mask]
        if seed_values.size == 0:
            output.pixel_data = torch.zeros_like(image.pixel_data)
            return output

        # Step 3: Eliminate plateau pixels via perturbation
        sorted_vals = np.sort(seed_values)
        diffs = np.diff(sorted_vals)
        positive_diffs = diffs[diffs > 0]
        min_diff = positive_diffs.min() if positive_diffs.size > 0 else 1e-6
        rng = np.random.RandomState(42)
        perturbed = seed_values + rng.random(seed_values.shape) * min_diff

        # Step 4: Dilate to get local context max
        from scipy.ndimage import maximum_filter

        # Build a full image of perturbed values at maxima positions
        perturbed_img = np.zeros_like(img_np)
        perturbed_img[reg_max_mask] = perturbed
        dilated = maximum_filter(perturbed_img, size=3)

        # Step 5: Local maxima where perturbed == dilated (among regional max positions)
        local_max_mask = reg_max_mask & (perturbed_img == dilated) & (perturbed_img > 0)
        local_max_values = img_np[local_max_mask]

        # Step 6: Seed = average of local maxima values
        seed_threshold = (
            np.mean(local_max_values)
            if local_max_values.size > 0
            else np.mean(seed_values)
        )

        # Store seed threshold on output for use in region_grow
        output._sbrg_seed_threshold = float(seed_threshold)
        output.pixel_data = image.pixel_data.clone()
        return output

    def _region_grow(self, image: Image, output: Image) -> Image:
        """Stage 1b: Grow region from seed threshold."""
        device = resolve_device(image, explicit=self.device)
        img_np = image.pixel_data.detach().cpu().numpy().astype(np.float64)
        while img_np.ndim > 2:
            img_np = img_np.squeeze(0)

        seed_threshold = getattr(output, "_sbrg_seed_threshold", np.mean(img_np))

        # Region growing: pixels >= seed threshold belong to the region
        region = (img_np >= seed_threshold).astype(np.float32)

        output.pixel_data = torch.from_numpy(region).to(device)
        return output

    def _boundary_seg(self, image: Image, output: Image) -> Image:
        """Stage 2: Boundary segmentation via Sobel + morphology."""
        device = resolve_device(image, explicit=self.device)
        region_np = output.pixel_data.detach().cpu().numpy().astype(np.float64)
        while region_np.ndim > 2:
            region_np = region_np.squeeze(0)

        # Step 1: Sobel edge detection
        gradient = sobel(region_np)
        binary_mask = (gradient > 0).astype(np.float32)

        # Step 2: Dilate gradient mask (using framework dilation)
        mask_img = InMemoryImage(array=torch.from_numpy(binary_mask).to(device))
        dilated_img = mask_img.clone()
        MorphologyOperations.dilation(mask_img, dilated_img, radius=2, device=device)

        # Step 3: Erode to smooth
        eroded_img = dilated_img.clone()
        MorphologyOperations.erosion(dilated_img, eroded_img, radius=2, device=device)

        # Step 4: Boundary extraction E(A) = (A+B) - ((A+B)-B)
        # A is the SBRG region result, B is structuring element
        region_tensor = output.pixel_data.to(device).float()
        while region_tensor.ndim > 2:
            region_tensor = region_tensor.squeeze(0)

        region_img = InMemoryImage(array=region_tensor)
        dilated_region = region_img.clone()
        MorphologyOperations.dilation(
            region_img, dilated_region, radius=2, device=device
        )

        eroded_dilated = dilated_region.clone()
        MorphologyOperations.erosion(
            dilated_region, eroded_dilated, radius=2, device=device
        )

        boundary = dilated_region.pixel_data.float() - eroded_dilated.pixel_data.float()
        boundary = torch.clamp(boundary, 0, 1)

        # Combine: region minus boundary artifacts, keep as binary
        result = eroded_img.pixel_data.float()
        result = torch.clamp(result, 0, 1)
        result = (result > 0).float()

        output.pixel_data = result.to(device)
        return output

    def apply(self, image: Image, output: Image) -> Image:
        self.find_seed(image, output)
        self.region_grow(image, output)
        self.boundary_seg(image, output)
        return output
