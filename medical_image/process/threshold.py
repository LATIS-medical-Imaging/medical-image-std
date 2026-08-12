import torch
import torch.nn.functional as F

from medical_image.data.image import Image, requires_loaded
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.metrics import Metrics
from medical_image.utils.device import resolve_device


class Threshold:

    @staticmethod
    def percentile_threshold(
            image: Image,
            output: Image,
            mask: Image,
            percentile: float = 0.995,
            device=None,
    ) -> Image:
        """
        Breast-masked percentile threshold for a single 2D
        feature map.

        The threshold is computed only from pixels inside
        the breast mask:

            T = percentile(X[B], p)

        Candidate pixels are:

            C = (X > T) AND B
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        x = image.pixel_data.to(device).float()
        m = mask.pixel_data.to(device).bool()

        while x.ndim > 2:
            x = x.squeeze(0)

        while m.ndim > 2:
            m = m.squeeze(0)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D feature map, got {x.shape}"
            )

        if m.ndim != 2:
            raise ValueError(
                f"Expected 2D mask, got {m.shape}"
            )

        if x.shape != m.shape:
            raise ValueError(
                f"Feature/mask shape mismatch: "
                f"{x.shape} vs {m.shape}"
            )

        breast_values = x[m]

        if breast_values.numel() == 0:
            raise ValueError(
                "Breast mask contains no valid pixels"
            )

        if not 0.0 < percentile < 1.0:
            raise ValueError(
                f"Percentile must be in (0, 1), "
                f"got {percentile}"
            )

        threshold = torch.quantile(
            breast_values,
            percentile,
        )

        candidates = (
                (x > threshold)
                & m
        ).to(torch.uint8)

        output.pixel_data = candidates

        return output

    @staticmethod
    def percentile_threshold_multiscale(
            image: Image,
            output: Image,
            mask: Image,
            percentile: float = 0.995,
            device=None,
    ) -> Image:
        """
        Breast-masked percentile threshold for multi-scale
        feature maps.

        Input:
            [N, H, W]

        Output:
            [N, H, W]

        Each scale receives an independent threshold:

            T_i = percentile(X_i[B], p)

        Candidate pixels:

            C_i = (X_i > T_i) AND B
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        x = image.pixel_data.to(device).float()
        m = mask.pixel_data.to(device).bool()

        while m.ndim > 2:
            m = m.squeeze(0)

        if x.ndim != 3:
            raise ValueError(
                f"Expected [N,H,W], got {x.shape}"
            )

        if m.ndim != 2:
            raise ValueError(
                f"Expected [H,W] mask, got {m.shape}"
            )

        if x.shape[-2:] != m.shape:
            raise ValueError(
                f"Feature/mask shape mismatch: "
                f"{x.shape[-2:]} vs {m.shape}"
            )

        if not 0.0 < percentile < 1.0:
            raise ValueError(
                f"Percentile must be in (0, 1), "
                f"got {percentile}"
            )

        candidates = torch.zeros_like(
            x,
            dtype=torch.uint8,
        )

        breast_values_mask = m

        for i in range(x.shape[0]):

            xi = x[i]

            breast_values = xi[breast_values_mask]

            if breast_values.numel() == 0:
                raise ValueError(
                    "Breast mask contains no valid pixels"
                )

            threshold = torch.quantile(
                breast_values,
                percentile,
            )

            candidates[i] = (
                    (xi > threshold)
                    & m
            ).to(torch.uint8)

        output.pixel_data = candidates

        return output
    @staticmethod
    def robust_threshold_multiscale(
            image: Image,
            output: Image,
            k: float = 3.0,
            device=None,
    ) -> Image:
        """
        Apply robust median + k*MAD thresholding independently
        to each feature channel.

        Input:

            [N, H, W]

        Output:

            [N, H, W]

        Each channel receives its own threshold.
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        x = image.pixel_data.to(
            device
        ).float()

        if x.ndim != 3:
            raise ValueError(
                f"Expected [N,H,W], got {x.shape}"
            )

        median = torch.median(
            x.flatten(start_dim=1),
            dim=1,
        ).values

        mad = torch.median(
            torch.abs(
                x
                - median[:, None, None]
            ).flatten(start_dim=1),
            dim=1,
        ).values

        threshold = (
                median
                + k * mad
        )

        candidates = (
                x
                > threshold[:, None, None]
        )

        output.pixel_data = (
            candidates.to(torch.uint8)
        )

        return output
    
    @staticmethod
    def robust_threshold(
            image: Image,
            output: Image,
            k: float = 3.0,
            device=None,
    ) -> Image:
        """
        Robust adaptive threshold using:

            T = median(X) + k * MAD(X)

        where:

            MAD = median(|X - median(X)|)

        Produces a binary candidate map.

        Args:
            image:
                2D feature map.

            output:
                Output binary Image.

            k:
                Robust threshold multiplier.

            device:
                Computation device.

        Returns:
            Binary candidate map with values {0, 1}.
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        x = image.pixel_data.to(
            device
        ).float()

        while x.ndim > 2:
            x = x.squeeze(0)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D feature map, got {x.shape}"
            )

        # ----------------------------------------------------------
        # Median
        # ----------------------------------------------------------

        median = torch.median(x)

        # ----------------------------------------------------------
        # Median Absolute Deviation
        # ----------------------------------------------------------

        mad = torch.median(
            torch.abs(x - median)
        )

        # ----------------------------------------------------------
        # Robust threshold
        # ----------------------------------------------------------

        threshold = median + k * mad

        # ----------------------------------------------------------
        # Binary candidate map
        # ----------------------------------------------------------

        output.pixel_data = (
                x > threshold
        ).to(torch.uint8)

        return output

    @staticmethod
    def otsu_threshold_multiscale(
            image: Image,
            output: Image,
            mask: Image = None,
            device=None,
    ) -> Image:
        """
        Apply Otsu thresholding independently to each scale.

        Input:
            [N, H, W]

        Output:
            [N, H, W]
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        x = image.pixel_data.to(device).float()

        if x.ndim != 3:
            raise ValueError(
                f"Expected [N,H,W], got {x.shape}"
            )

        # ----------------------------------------------------------
        # Prepare mask
        # ----------------------------------------------------------

        if mask is not None:

            m = mask.pixel_data.to(device).bool()

            while m.ndim > 2:
                m = m.squeeze(0)

            if m.ndim != 2:
                raise ValueError(
                    f"Expected 2D mask, got {m.shape}"
                )

            if m.shape != x.shape[-2:]:
                raise ValueError(
                    f"Feature/mask shape mismatch: "
                    f"{x.shape[-2:]} vs {m.shape}"
                )

        else:
            m = None

        # ----------------------------------------------------------
        # Output
        # ----------------------------------------------------------

        candidates = torch.zeros_like(
            x,
            dtype=torch.uint8,
        )

        # ----------------------------------------------------------
        # Process each scale independently
        # ----------------------------------------------------------

        for i in range(x.shape[0]):

            xi = x[i]

            if m is not None:
                values = xi[m]
            else:
                values = xi.flatten()

            if values.numel() == 0:
                raise ValueError(
                    f"No valid pixels for scale {i}"
                )

            if not torch.isfinite(values).all():
                raise ValueError(
                    f"NaN or Inf detected at scale {i}"
                )

            min_val = values.min()
            max_val = values.max()

            # ------------------------------------------------------
            # Constant feature map
            # ------------------------------------------------------

            if min_val == max_val:

                candidates[i] = (
                        xi > min_val
                ).to(torch.uint8)

                if m is not None:
                    candidates[i] &= m

                continue

            # ------------------------------------------------------
            # Histogram
            # ------------------------------------------------------

            hist = torch.histc(
                values,
                bins=256,
                min=min_val.item(),
                max=max_val.item(),
            )

            bin_centers = torch.linspace(
                min_val,
                max_val,
                steps=256,
                device=device,
            )

            # ------------------------------------------------------
            # Cumulative statistics
            # ------------------------------------------------------

            weight1 = torch.cumsum(
                hist,
                dim=0,
            )

            weight2 = (
                    hist.sum()
                    - weight1
            )

            cumulative_mean = torch.cumsum(
                hist * bin_centers,
                dim=0,
            )

            mean1 = (
                    cumulative_mean
                    / torch.clamp(
                weight1,
                min=1e-12,
            )
            )

            total_mean = cumulative_mean[-1]

            mean2 = (
                            total_mean
                            - cumulative_mean
                    ) / torch.clamp(
                weight2,
                min=1e-12,
            )

            # ------------------------------------------------------
            # Between-class variance
            # ------------------------------------------------------

            variance_between = (
                    weight1
                    * weight2
                    * (mean1 - mean2) ** 2
            )

            threshold_idx = torch.argmax(
                variance_between
            )

            threshold_value = (
                bin_centers[threshold_idx]
            )

            # ------------------------------------------------------
            # Candidate map
            # ------------------------------------------------------

            candidate = (
                    xi > threshold_value
            )

            # ------------------------------------------------------
            # Breast mask
            # ------------------------------------------------------

            if m is not None:
                candidate &= m

            candidates[i] = candidate.to(
                torch.uint8
            )

        output.pixel_data = candidates

        return output
    @staticmethod
    def otsu_threshold_breast_mask(
            image: Image,
            output: Image = None,
            mask: Image = None,
            device=None,
    ) -> Image:
        """
        Otsu thresholding for a single 2D feature map.

        If a mask is provided, Otsu statistics are computed only
        from pixels inside the mask.

        The final binary result is also constrained by the mask:

            C = (X > T_otsu) AND mask
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        x = image.pixel_data.to(device).float()

        # ----------------------------------------------------------
        # Normalize dimensionality
        # ----------------------------------------------------------

        while x.ndim > 2:
            x = x.squeeze(0)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D feature map, got {x.shape}"
            )

        # ----------------------------------------------------------
        # Prepare mask
        # ----------------------------------------------------------

        if mask is not None:

            m = mask.pixel_data.to(device).bool()

            while m.ndim > 2:
                m = m.squeeze(0)

            if m.ndim != 2:
                raise ValueError(
                    f"Expected 2D mask, got {m.shape}"
                )

            if m.shape != x.shape:
                raise ValueError(
                    f"Feature/mask shape mismatch: "
                    f"{x.shape} vs {m.shape}"
                )

            values = x[m]

        else:

            m = None
            values = x.flatten()

        # ----------------------------------------------------------
        # Validate input
        # ----------------------------------------------------------

        if values.numel() == 0:
            raise ValueError(
                "No valid pixels available for Otsu thresholding"
            )

        if not torch.isfinite(values).all():
            raise ValueError(
                "Otsu input contains NaN or Inf"
            )

        # ----------------------------------------------------------
        # Constant image
        # ----------------------------------------------------------

        min_val = values.min()
        max_val = values.max()

        if min_val == max_val:

            threshold_value = min_val

            binary_image = (
                    x > threshold_value
            ).to(torch.uint8)

            if m is not None:
                binary_image &= m

        else:

            # ------------------------------------------------------
            # Histogram
            # ------------------------------------------------------

            bins = 256

            hist = torch.histc(
                values,
                bins=bins,
                min=min_val.item(),
                max=max_val.item(),
            )

            bin_centers = torch.linspace(
                min_val,
                max_val,
                steps=bins,
                device=device,
            )

            # ------------------------------------------------------
            # Class probabilities / weights
            # ------------------------------------------------------

            weight1 = torch.cumsum(
                hist,
                dim=0,
            )

            weight2 = (
                    hist.sum()
                    - weight1
            )

            # ------------------------------------------------------
            # Class means
            # ------------------------------------------------------

            cumulative_mean = torch.cumsum(
                hist * bin_centers,
                dim=0,
            )

            mean1 = (
                    cumulative_mean
                    / torch.clamp(
                weight1,
                min=1e-12,
            )
            )

            total_mean = cumulative_mean[-1]

            mean2 = (
                            total_mean
                            - cumulative_mean
                    ) / torch.clamp(
                weight2,
                min=1e-12,
            )

            # ------------------------------------------------------
            # Between-class variance
            # ------------------------------------------------------

            variance_between = (
                    weight1
                    * weight2
                    * (mean1 - mean2) ** 2
            )

            threshold_idx = torch.argmax(
                variance_between
            )

            threshold_value = (
                bin_centers[threshold_idx]
            )

            # ------------------------------------------------------
            # Binary candidate map
            # ------------------------------------------------------

            binary_image = (
                    x > threshold_value
            ).to(torch.uint8)

            # ------------------------------------------------------
            # Hard spatial constraint
            # ------------------------------------------------------

            if m is not None:
                binary_image &= m

        # ----------------------------------------------------------
        # Output
        # ----------------------------------------------------------

        if output is None:
            output = InMemoryImage(
                array=binary_image
            )
        else:
            output.pixel_data = binary_image

        return output
    @staticmethod
    @requires_loaded
    def otsu_threshold(image: Image, output: Image = None, device=None) -> Image:
        """
        Applies Otsu's thresholding method to a grayscale image using PyTorch.

        Args:
            image: Input image with pixel_data as torch.Tensor.
            output: Optional output Image object to store the result.
            device: Device to perform computation (None = infer from image).

        Returns:
            The output Image (or a new InMemoryImage if output is None).
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).to(torch.float32)

        min_val = torch.min(img)
        max_val = torch.max(img)
        bins = 256 if max_val <= 255 else 4096

        hist = torch.histc(img, bins=bins, min=min_val.item(), max=max_val.item())
        bin_centers = torch.linspace(min_val, max_val, steps=bins, device=device)

        weight1 = torch.cumsum(hist, dim=0)
        weight2 = hist.sum() - weight1
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / torch.clamp(weight1, min=1e-6)
        mean2 = (hist * bin_centers).sum() - torch.cumsum(hist * bin_centers, dim=0)
        mean2 = mean2 / torch.clamp(weight2, min=1e-6)

        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        threshold_idx = torch.argmax(variance_between)
        threshold_value = bin_centers[threshold_idx]

        binary_image = (img > threshold_value).to(torch.uint8)

        if output is None:
            output = InMemoryImage(array=binary_image)
        else:
            output.pixel_data = binary_image
        return output

    @staticmethod
    @requires_loaded
    def sauvola_threshold(
        image: Image,
        output: Image = None,
        window_size: int = 10,
        k: float = 0.5,
        r: int = 128,
        device=None,
    ) -> Image:
        """
        Applies Sauvola adaptive thresholding to a grayscale image using PyTorch.

        Args:
            image: Input grayscale image.
            output: Optional Image object for result.
            window_size: Odd size of the local window.
            k: Scaling factor in threshold formula.
            r: Dynamic range of standard deviation.
            device: Device for computation (None = infer from image).

        Returns:
            The output Image (or a new InMemoryImage if output is None).
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer.")
        pad = window_size // 2

        img4d = img.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, window_size, window_size), device=device) / (
            window_size**2
        )

        mean = F.conv2d(F.pad(img4d, (pad, pad, pad, pad), mode="replicate"), kernel)
        mean_sq = F.conv2d(
            F.pad(img4d**2, (pad, pad, pad, pad), mode="replicate"), kernel
        )
        std = torch.sqrt(mean_sq - mean**2 + 1e-8)

        thresh = mean * (1 + k * (std / r - 1))
        binary = torch.where(
            img > thresh.squeeze(0).squeeze(0),
            torch.tensor(255, device=device, dtype=torch.uint8),
            torch.tensor(0, device=device, dtype=torch.uint8),
        )

        if output is None:
            output = InMemoryImage(array=binary)
        else:
            output.pixel_data = binary
        return output

    @staticmethod
    @requires_loaded
    def binarize(image: Image, output: Image, alpha: float, device=None) -> Image:
        """
        Binarizes an image based on local and global variance using PyTorch.

        Formula:
            binary = local_variance^2 < alpha * global_variance^2

        Args:
            image: Input grayscale image.
            output: Output Image object for storing result.
            alpha: Scaling factor relating local and global variances.
            device: Device for computation (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()

        # Local variance
        local_var_img = InMemoryImage(array=torch.empty_like(img))
        Metrics.local_variance(image, output=local_var_img, kernel=5)

        # Global variance
        global_var_img = InMemoryImage(array=torch.empty(1, device=device))
        Metrics.variance(image, output=global_var_img)

        # Compute binary mask
        binary = (
            local_var_img.pixel_data**2 >= alpha * global_var_img.pixel_data**2
        ).to(torch.uint8)

        output.pixel_data = binary
        return output
