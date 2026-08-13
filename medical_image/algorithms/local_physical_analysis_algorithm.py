from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.data.physical_features import (
    CandidatePhysicalFeatures,
)


class LocalPhysicalAnalysisAlgorithm(Algorithm):
    """
    Native-resolution local physical analysis of candidate
    microcalcifications.

    The mammogram is never resized or downsampled.

    Processing pipeline
    -------------------
    1. Candidate connected components are detected once using SciPy.
    2. The center of each candidate is computed.
    3. Candidates are processed in batches.
    4. A native-resolution local window is extracted around each center.
    5. Physical/image features are computed entirely on the selected device.
    6. Only the final scalar values are transferred to CPU.

    Extracted features
    ------------------
    Intensity:
        - peak value
        - neighborhood median
        - neighborhood MAD
        - robust peak Z-score

    Radial:
        - radial center value
        - radial ring mean
        - center-to-ring contrast
        - radial decay
        - radial peak width
        - radial symmetry

    Shape:
        - area
        - perimeter
        - circularity
        - eccentricity
        - solidity
        - aspect ratio
        - equivalent diameter

    Notes
    -----
    Candidate detection is intentionally the only global SciPy operation.
    All subsequent candidate analysis is batched and performed with PyTorch.
    """

    def __init__(
        self,
        window_size: int = 31,
        radial_radius: int | None = None,
        shape_threshold_k: float = 2.0,
        min_area: int = 1,
        max_area: int = 200,
        connectivity: int = 2,
        batch_size: int = 512,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        if window_size < 7:
            raise ValueError(
                "window_size must be at least 7."
            )

        if window_size % 2 == 0:
            raise ValueError(
                "window_size must be odd."
            )

        if radial_radius is not None:
            if radial_radius < 1:
                raise ValueError(
                    "radial_radius must be >= 1."
                )

            expected_window_size = (
                2 * radial_radius + 1
            )

            if expected_window_size != window_size:
                raise ValueError(
                    "radial_radius and window_size are inconsistent. "
                    f"For window_size={window_size}, "
                    f"radial_radius must be "
                    f"{window_size // 2}."
                )

        if shape_threshold_k < 0:
            raise ValueError(
                "shape_threshold_k must be >= 0."
            )

        if min_area < 1:
            raise ValueError(
                "min_area must be >= 1."
            )

        if max_area < min_area:
            raise ValueError(
                "max_area must be >= min_area."
            )

        if connectivity not in (1, 2):
            raise ValueError(
                "connectivity must be either 1 or 2."
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1."
            )

        self.window_size = window_size

        self.radius = (
            radial_radius
            if radial_radius is not None
            else window_size // 2
        )

        self.shape_threshold_k = shape_threshold_k
        self.min_area = min_area
        self.max_area = max_area
        self.connectivity = connectivity
        self.batch_size = batch_size

        # --------------------------------------------------
        # Precompute radial geometry.
        #
        # This geometry is independent of the image and
        # therefore only needs to be constructed once.
        # --------------------------------------------------

        coordinates = torch.arange(
            -self.radius,
            self.radius + 1,
            dtype=torch.float32,
        )

        yy, xx = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )

        self._distance = torch.sqrt(
            xx.square() + yy.square()
        )

        # --------------------------------------------------
        # Build one binary mask for every radial ring.
        #
        # Shape:
        #     [num_rings, H, W]
        #
        # Ring 0 contains only the center pixel.
        # --------------------------------------------------

        radial_masks = []

        for radius in range(self.radius + 1):
            if radius == 0:
                ring = self._distance < 0.5
            else:
                ring = (
                    (self._distance >= radius - 0.5)
                    & (self._distance < radius + 0.5)
                )

            radial_masks.append(ring)

        self._radial_masks = torch.stack(
            radial_masks,
            dim=0,
        )

    # ======================================================
    # Utilities
    # ======================================================

    @staticmethod
    def _squeeze_2d(
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """
        Convert [1,H,W] tensors to [H,W].

        Only singleton leading dimensions are removed.
        """

        while tensor.ndim > 2:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"{name} must be [H,W] or [1,H,W], "
                    f"got {tuple(tensor.shape)}"
                )

            tensor = tensor.squeeze(0)

        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must be 2D, "
                f"got {tuple(tensor.shape)}"
            )

        return tensor

    # ======================================================
    # Candidate detection
    # ======================================================

    def _find_candidates(
        self,
        candidate_data: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Find connected candidate objects once.

        This is intentionally the only global SciPy operation.

        Parameters
        ----------
        candidate_data:
            Binary candidate map [H,W].

        mask:
            Binary breast mask [H,W].

        Returns
        -------
        centers:
            List of integer (cy, cx) candidate centers.

        labels:
            Connected-component labels corresponding to centers.

        areas:
            Area of each connected component.
        """

        # Restrict candidates to the breast region before
        # transferring the data to NumPy.
        candidate_data = candidate_data & mask

        candidate_np = (
            candidate_data
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8, copy=False)
        )

        structure = ndimage.generate_binary_structure(
            rank=2,
            connectivity=self.connectivity,
        )

        labels, num_labels = ndimage.label(
            candidate_np,
            structure=structure,
        )

        if num_labels == 0:
            return [], [], []

        component_ids = np.arange(
            1,
            num_labels + 1,
            dtype=np.int32,
        )

        # Compute all component areas in one operation.
        areas = ndimage.sum(
            candidate_np,
            labels,
            index=component_ids,
        )

        keep = (
            (areas >= self.min_area)
            & (areas <= self.max_area)
        )

        if not np.any(keep):
            return [], [], []

        kept_indices = np.flatnonzero(keep)

        labels_kept = component_ids[keep]
        areas_kept = areas[keep].astype(
            np.float32
        )

        # Compute all component centers once.
        centers_all = ndimage.center_of_mass(
            candidate_np,
            labels,
            index=component_ids,
        )

        centers = [
            (
                int(round(centers_all[index][0])),
                int(round(centers_all[index][1])),
            )
            for index in kept_indices
        ]

        return (
            centers,
            labels_kept.tolist(),
            areas_kept.tolist(),
        )

    # ======================================================
    # Batched native-resolution window extraction
    # ======================================================

    def _extract_windows(
        self,
        image: torch.Tensor,
        centers,
    ) -> torch.Tensor:
        """
        Extract native-resolution local windows.

        The image is padded once and then all candidate
        windows are gathered using vectorized indexing.

        Parameters
        ----------
        image:
            Full mammogram [H,W].

        centers:
            List of (cy, cx).

        Returns
        -------
        windows:
            [N, window_size, window_size]
        """

        if not centers:
            return torch.empty(
                (
                    0,
                    self.window_size,
                    self.window_size,
                ),
                device=image.device,
                dtype=image.dtype,
            )

        radius = self.radius

        # Pad once for the complete batch.
        padded = F.pad(
            image[None, None],
            (
                radius,
                radius,
                radius,
                radius,
            ),
            mode="reflect",
        )[0, 0]

        centers_tensor = torch.as_tensor(
            centers,
            device=image.device,
            dtype=torch.long,
        )

        y = centers_tensor[:, 0] + radius
        x = centers_tensor[:, 1] + radius

        offsets = torch.arange(
            -radius,
            radius + 1,
            device=image.device,
        )

        yy = (
            y[:, None]
            + offsets[None, :]
        )

        xx = (
            x[:, None]
            + offsets[None, :]
        )

        # Advanced indexing produces:
        #
        # [N, window_size, window_size]
        #
        windows = padded[
            yy[:, :, None],
            xx[:, None, :],
        ]

        return windows

    # ======================================================
    # Robust statistics
    # ======================================================

    @staticmethod
    def _batch_median_mad(
        windows: torch.Tensor,
    ):
        """
        Compute median and MAD independently for every window.

        Parameters
        ----------
        windows:
            [N,H,W]

        Returns
        -------
        median:
            [N]

        mad:
            [N]
        """

        flat = windows.flatten(
            start_dim=1
        )

        median = torch.median(
            flat,
            dim=1,
        ).values

        mad = torch.median(
            torch.abs(
                flat
                - median[:, None]
            ),
            dim=1,
        ).values

        return median, mad

    # ======================================================
    # Peak features
    # ======================================================

    def _peak_features(
        self,
        windows: torch.Tensor,
    ):
        """
        Compute robust local peak prominence.

        Robust Z-score:

            Z = (center - median)
                / (MAD + epsilon)

        Returns
        -------
        center:
            Intensity at the candidate center.

        median:
            Local window median.

        mad:
            Local window MAD.

        robust_z:
            Robust peak prominence.
        """

        center = windows[
            :,
            self.radius,
            self.radius,
        ]

        median, mad = (
            self._batch_median_mad(
                windows
            )
        )

        robust_z = (
            center - median
        ) / (
            mad + 1e-8
        )

        return (
            center,
            median,
            mad,
            robust_z,
        )

    # ======================================================
    # Radial features
    # ======================================================

    def _radial_features(
        self,
        windows: torch.Tensor,
    ):
        """
        Compute radial intensity features.

        Parameters
        ----------
        windows:
            [N,H,W]

        Returns
        -------
        radial_profile:
            Mean intensity at every radial distance [N,R].

        ring_mean:
            Mean intensity outside the center ring [N].

        center_ring_contrast:
            Center intensity minus ring mean [N].

        radial_decay:
            Slope of the radial profile. A negative value
            indicates decreasing intensity with radius.

        radial_peak_width:
            First radius at which the radial response
            falls below half of the center-to-ring contrast.

        radial_symmetry:
            Inverse coefficient-of-variation averaged
            across non-center rings.
        """

        radial_masks = (
            self._radial_masks
            .to(
                device=windows.device
            )
        )

        # --------------------------------------------------
        # Radial profile
        #
        # windows:
        #     [N,H,W]
        #
        # radial_masks:
        #     [R,H,W]
        #
        # result:
        #     [N,R]
        # --------------------------------------------------

        weighted = (
            windows[:, None]
            * radial_masks[None]
        )

        counts = (
            radial_masks.sum(
                dim=(1, 2)
            )
            .clamp_min(1)
        )

        radial_profile = (
            weighted.sum(
                dim=(2, 3)
            )
            / counts[None]
        )

        center = radial_profile[:, 0]

        # --------------------------------------------------
        # Center-to-ring contrast
        # --------------------------------------------------

        ring_values = radial_profile[:, 1:]

        ring_mean = ring_values.mean(
            dim=1
        )

        center_ring_contrast = (
            center
            - ring_mean
        )

        # --------------------------------------------------
        # Radial decay
        #
        # Linear regression slope of radial intensity
        # against radius.
        # --------------------------------------------------

        if radial_profile.shape[1] > 2:
            radii = torch.arange(
                1,
                radial_profile.shape[1],
                device=windows.device,
                dtype=windows.dtype,
            )

            profile_without_center = (
                radial_profile[:, 1:]
            )

            y_centered = (
                profile_without_center
                - profile_without_center.mean(
                    dim=1,
                    keepdim=True,
                )
            )

            r_centered = (
                radii
                - radii.mean()
            )

            denominator = (
                r_centered.square().sum()
                + 1e-8
            )

            radial_decay = (
                (
                    y_centered
                    * r_centered[None]
                ).sum(dim=1)
                / denominator
            )

        else:
            radial_decay = torch.zeros(
                windows.shape[0],
                device=windows.device,
                dtype=windows.dtype,
            )

        # --------------------------------------------------
        # Radial peak width
        #
        # Find the first radius where the response falls
        # below the half-maximum level.
        # --------------------------------------------------

        half_level = (
            ring_mean
            + 0.5
            * (
                center
                - ring_mean
            )
        )

        above_half = (
            radial_profile
            > half_level[:, None]
        )

        # The center itself is always considered above
        # the threshold.
        above_half[:, 0] = True

        below_half = ~above_half

        has_fall = below_half.any(
            dim=1
        )

        first_fall = torch.argmax(
            below_half.to(torch.int64),
            dim=1,
        ).to(radial_profile.dtype)

        radial_peak_width = torch.where(
            has_fall,
            first_fall,
            torch.full_like(
                first_fall,
                float(
                    radial_profile.shape[1] - 1
                ),
            ),
        )

        # --------------------------------------------------
        # Radial symmetry
        #
        # Compute the coefficient of variation within
        # every radial ring.
        # --------------------------------------------------

        ring_count = (
            radial_masks.sum(
                dim=(1, 2)
            )
            .clamp_min(2)
        )

        ring_mean_values = (
            weighted.sum(
                dim=(2, 3)
            )
            / ring_count[None]
        )

        ring_squared = (
            windows[:, None].square()
            * radial_masks[None]
        )

        ring_second_moment = (
            ring_squared.sum(
                dim=(2, 3)
            )
            / ring_count[None]
        )

        ring_variance = torch.clamp(
            ring_second_moment
            - ring_mean_values.square(),
            min=0.0,
        )

        ring_std = torch.sqrt(
            ring_variance
        )

        coefficient = (
            ring_std
            / (
                ring_mean_values.abs()
                + 1e-8
            )
        )

        # Exclude the center ring.
        coefficient = coefficient[:, 1:]

        radial_symmetry = (
            1.0
            / (
                1.0
                + coefficient.mean(dim=1)
            )
        )

        return (
            radial_profile,
            ring_mean,
            center_ring_contrast,
            radial_decay,
            radial_peak_width,
            radial_symmetry,
        )

    # ======================================================
    # Shape features
    # ======================================================

    def _shape_features(
        self,
        windows: torch.Tensor,
    ):
        """
        Compute GPU-native local shape features.

        A candidate's connected-component centroid is not
        assumed to coincide with its intensity peak.

        Therefore the local intensity maximum is used as
        the physical anchor when validating the thresholded
        object.

        The threshold is:

            threshold = median + k * MAD
        """

        n = windows.shape[0]

        # --------------------------------------------------
        # Robust local threshold
        # --------------------------------------------------

        median, mad = (
            self._batch_median_mad(
                windows
            )
        )

        threshold = (
            median
            + self.shape_threshold_k * mad
        )

        binary = (
            windows
            > threshold[:, None, None]
        )

        # --------------------------------------------------
        # Find the brightest pixel in every window.
        # --------------------------------------------------

        flat = windows.flatten(
            start_dim=1
        )

        peak_index = flat.argmax(
            dim=1
        )

        peak_y = (
            peak_index
            // self.window_size
        )

        peak_x = (
            peak_index
            % self.window_size
        )

        batch_index = torch.arange(
            n,
            device=windows.device,
        )

        peak_is_bright = binary[
            batch_index,
            peak_y,
            peak_x,
        ]

        # A candidate is valid if its local intensity peak
        # survives the robust threshold.
        valid = peak_is_bright

        # --------------------------------------------------
        # Area
        # --------------------------------------------------

        area = binary.sum(
            dim=(1, 2)
        ).float()

        # Ensure a valid candidate always has at least one
        # foreground pixel.
        area = torch.where(
            valid,
            area.clamp_min(1.0),
            torch.zeros_like(area),
        )

        # --------------------------------------------------
        # Coordinate grids
        # --------------------------------------------------

        y_grid = torch.arange(
            self.window_size,
            device=windows.device,
        )[None, :, None]

        x_grid = torch.arange(
            self.window_size,
            device=windows.device,
        )[None, None, :]

        # --------------------------------------------------
        # Bounding box
        # --------------------------------------------------

        invalid_y = torch.full(
            (
                n,
                self.window_size,
                self.window_size,
            ),
            self.window_size,
            device=windows.device,
        )

        invalid_x = torch.full(
            (
                n,
                self.window_size,
                self.window_size,
            ),
            self.window_size,
            device=windows.device,
        )

        y_min = torch.where(
            binary,
            y_grid.expand(
                n,
                -1,
                self.window_size,
            ),
            invalid_y,
        ).amin(dim=(1, 2))

        x_min = torch.where(
            binary,
            x_grid.expand(
                n,
                self.window_size,
                -1,
            ),
            invalid_x,
        ).amin(dim=(1, 2))

        y_max = torch.where(
            binary,
            y_grid.expand(
                n,
                -1,
                self.window_size,
            ),
            torch.zeros_like(
                invalid_y
            ),
        ).amax(dim=(1, 2))

        x_max = torch.where(
            binary,
            x_grid.expand(
                n,
                self.window_size,
                -1,
            ),
            torch.zeros_like(
                invalid_x
            ),
        ).amax(dim=(1, 2))

        valid = area > 0

        height = (
            y_max
            - y_min
            + 1
        )

        width = (
            x_max
            - x_min
            + 1
        )

        aspect_ratio = (
            torch.maximum(
                height,
                width,
            )
            / (
                torch.minimum(
                    height,
                    width,
                )
                + 1e-8
            )
        )

        aspect_ratio = torch.where(
            valid,
            aspect_ratio,
            torch.zeros_like(
                aspect_ratio
            ),
        )

        # --------------------------------------------------
        # Equivalent diameter
        # --------------------------------------------------

        equivalent_diameter = (
            2.0
            * torch.sqrt(
                area
                / torch.pi
            )
        )

        # --------------------------------------------------
        # Perimeter
        #
        # Count exposed 4-connected edges.
        # --------------------------------------------------

        binary_float = binary.float()

        up = F.pad(
            binary_float[:, 1:],
            (0, 0, 1, 0),
        )

        down = F.pad(
            binary_float[:, :-1],
            (0, 0, 0, 1),
        )

        left = F.pad(
            binary_float[:, :, 1:],
            (1, 0, 0, 0),
        )

        right = F.pad(
            binary_float[:, :, :-1],
            (0, 1, 0, 0),
        )

        perimeter = (
            binary_float
            * (
                4.0
                - up
                - down
                - left
                - right
            )
        ).sum(
            dim=(1, 2)
        )

        # --------------------------------------------------
        # Circularity
        #
        # C = 4*pi*A / P^2
        # --------------------------------------------------

        circularity = (
            4.0
            * torch.pi
            * area
            / (
                perimeter.square()
                + 1e-8
            )
        )

        circularity = torch.clamp(
            circularity,
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # Binary-object centroid
        # --------------------------------------------------

        weights = binary.float()

        y_coordinates = torch.arange(
            self.window_size,
            device=windows.device,
            dtype=windows.dtype,
        )[None, :, None]

        x_coordinates = torch.arange(
            self.window_size,
            device=windows.device,
            dtype=windows.dtype,
        )[None, None, :]

        area_safe = area.clamp_min(
            1.0
        )

        centroid_y = (
            (
                weights
                * y_coordinates
            ).sum(dim=(1, 2))
            / area_safe
        )

        centroid_x = (
            (
                weights
                * x_coordinates
            ).sum(dim=(1, 2))
            / area_safe
        )

        # --------------------------------------------------
        # Second-order moments
        #
        # These describe the spatial spread and elongation
        # of the thresholded particle.
        # --------------------------------------------------

        dy = (
            y_coordinates
            - centroid_y[:, None, None]
        )

        dx = (
            x_coordinates
            - centroid_x[:, None, None]
        )

        mu_xx = (
            weights
            * dx.square()
        ).sum(dim=(1, 2)) / area_safe

        mu_yy = (
            weights
            * dy.square()
        ).sum(dim=(1, 2)) / area_safe

        mu_xy = (
            weights
            * dx
            * dy
        ).sum(dim=(1, 2)) / area_safe

        trace = (
            mu_xx
            + mu_yy
        )

        determinant = (
            mu_xx * mu_yy
            - mu_xy.square()
        )

        discriminant = torch.clamp(
            trace.square()
            - 4.0 * determinant,
            min=0.0,
        )

        sqrt_discriminant = torch.sqrt(
            discriminant
        )

        lambda_max = (
            trace
            + sqrt_discriminant
        ) / 2.0

        lambda_min = (
            trace
            - sqrt_discriminant
        ) / 2.0

        eccentricity = torch.sqrt(
            torch.clamp(
                1.0
                - (
                    lambda_min
                    / (
                        lambda_max
                        + 1e-8
                    )
                ),
                0.0,
                1.0,
            )
        )

        # --------------------------------------------------
        # Solidity approximation
        #
        # Exact convex-hull solidity would require a
        # per-object geometric operation.
        #
        # For small microcalcification particles, use
        # bounding-box occupancy as a stable GPU-native
        # approximation.
        # --------------------------------------------------

        bbox_area = (
            height
            * width
        ).float()

        solidity = (
            area
            / (
                bbox_area
                + 1e-8
            )
        )

        # --------------------------------------------------
        # Explicitly handle invalid candidates.
        # --------------------------------------------------

        zeros = torch.zeros_like(
            area
        )

        eccentricity = torch.where(
            valid,
            eccentricity,
            torch.ones_like(
                eccentricity
            ),
        )

        circularity = torch.where(
            valid,
            circularity,
            zeros,
        )

        solidity = torch.where(
            valid,
            solidity,
            zeros,
        )

        equivalent_diameter = torch.where(
            valid,
            equivalent_diameter,
            zeros,
        )

        perimeter = torch.where(
            valid,
            perimeter,
            zeros,
        )

        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "equivalent_diameter": (
                equivalent_diameter
            ),
        }

    # ======================================================
    # Main analysis
    # ======================================================

    @torch.inference_mode()
    def apply(
        self,
        image: Image,
        candidates: Image,
        breast_mask: Image,
    ) -> List[CandidatePhysicalFeatures]:
        """
        Analyze all candidate microcalcifications.

        The complete mammogram remains at native resolution.
        """

        device = torch.device(
            self.device
        )

        # --------------------------------------------------
        # Prepare input tensors
        # --------------------------------------------------

        image_data = self._squeeze_2d(
            image.pixel_data.to(
                device=device,
                dtype=torch.float32,
            ),
            "image",
        )

        candidate_data = self._squeeze_2d(
            candidates.pixel_data.to(
                device=device,
            ).bool(),
            "candidates",
        )

        mask = self._squeeze_2d(
            breast_mask.pixel_data.to(
                device=device,
            ).bool(),
            "breast_mask",
        )

        # --------------------------------------------------
        # Validate dimensions
        # --------------------------------------------------

        if not (
            image_data.shape
            == candidate_data.shape
            == mask.shape
        ):
            raise ValueError(
                "Image, candidates and breast mask "
                "must have identical spatial dimensions. "
                f"Got image={tuple(image_data.shape)}, "
                f"candidates={tuple(candidate_data.shape)}, "
                f"breast_mask={tuple(mask.shape)}."
            )

        # --------------------------------------------------
        # Candidate connected components
        #
        # This is intentionally performed only once.
        # --------------------------------------------------

        centers, labels, component_areas = (
            self._find_candidates(
                candidate_data,
                mask,
            )
        )

        if not centers:
            return []

        num_candidates = len(centers)

        print(
            "Local physical analysis: "
            f"{num_candidates} candidates found."
        )

        results: List[
            CandidatePhysicalFeatures
        ] = []

        # --------------------------------------------------
        # Process candidates in batches.
        # --------------------------------------------------

        for start in range(
            0,
            num_candidates,
            self.batch_size,
        ):
            end = min(
                start + self.batch_size,
                num_candidates,
            )

            batch_centers = centers[
                start:end
            ]

            batch_labels = labels[
                start:end
            ]

            # ----------------------------------------------
            # Native-resolution windows
            # ----------------------------------------------

            windows = self._extract_windows(
                image_data,
                batch_centers,
            )

            # ----------------------------------------------
            # Intensity / peak features
            # ----------------------------------------------

            (
                peak,
                neighborhood_median,
                neighborhood_mad,
                robust_z,
            ) = self._peak_features(
                windows
            )

            # ----------------------------------------------
            # Radial features
            # ----------------------------------------------

            (
                radial_profile,
                ring_mean,
                center_ring_contrast,
                radial_decay,
                radial_width,
                radial_symmetry,
            ) = self._radial_features(
                windows
            )

            # ----------------------------------------------
            # Shape features
            # ----------------------------------------------

            shape = self._shape_features(
                windows
            )

            # ----------------------------------------------
            # Transfer only final scalar results to CPU.
            #
            # This avoids repeatedly synchronizing the GPU
            # while constructing the result objects.
            # ----------------------------------------------

            peak_cpu = peak.cpu()
            median_cpu = neighborhood_median.cpu()
            mad_cpu = neighborhood_mad.cpu()
            robust_z_cpu = robust_z.cpu()

            radial_center_cpu = (
                radial_profile[:, 0].cpu()
            )

            ring_mean_cpu = ring_mean.cpu()
            contrast_cpu = (
                center_ring_contrast.cpu()
            )
            decay_cpu = radial_decay.cpu()
            width_cpu = radial_width.cpu()
            symmetry_cpu = radial_symmetry.cpu()

            shape_cpu = {
                key: value.cpu()
                for key, value in shape.items()
            }

            # ----------------------------------------------
            # Construct output objects.
            # ----------------------------------------------

            for i, (
                cy,
                cx,
            ) in enumerate(
                batch_centers
            ):
                results.append(
                    CandidatePhysicalFeatures(
                        label=batch_labels[i],

                        center_y=float(cy),
                        center_x=float(cx),

                        peak_value=float(
                            peak_cpu[i]
                        ),

                        neighborhood_median=float(
                            median_cpu[i]
                        ),

                        neighborhood_mad=float(
                            mad_cpu[i]
                        ),

                        robust_peak_z=float(
                            robust_z_cpu[i]
                        ),

                        radial_center=float(
                            radial_center_cpu[i]
                        ),

                        radial_ring_mean=float(
                            ring_mean_cpu[i]
                        ),

                        center_ring_contrast=float(
                            contrast_cpu[i]
                        ),

                        radial_decay=float(
                            decay_cpu[i]
                        ),

                        radial_peak_width=float(
                            width_cpu[i]
                        ),

                        radial_symmetry=float(
                            symmetry_cpu[i]
                        ),

                        area=float(
                            shape_cpu["area"][i]
                        ),

                        perimeter=float(
                            shape_cpu["perimeter"][i]
                        ),

                        circularity=float(
                            shape_cpu["circularity"][i]
                        ),

                        eccentricity=float(
                            shape_cpu["eccentricity"][i]
                        ),

                        solidity=float(
                            shape_cpu["solidity"][i]
                        ),

                        aspect_ratio=float(
                            shape_cpu["aspect_ratio"][i]
                        ),

                        equivalent_diameter=float(
                            shape_cpu[
                                "equivalent_diameter"
                            ][i]
                        ),
                    )
                )

        print(
            "Local physical analysis completed: "
            f"{len(results)} candidates analyzed."
        )

        return results
