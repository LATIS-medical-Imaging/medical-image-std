import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image


class CandidateEvidenceAlgorithm(Algorithm):

    def __init__(
        self,
        w_top_hat: float = 1.0,
        w_log: float = 1.0,
        w_dog: float = 0.75,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        self.w_top_hat = w_top_hat
        self.w_log = w_log
        self.w_dog = w_dog

    @staticmethod
    def _robust_normalize(
        x: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Robustly normalize a feature using statistics computed
        only inside the breast region.

        The output is approximately in [0, 1].
        Pixels outside the breast are forced to zero.
        """

        x = x.float()
        mask = mask.bool()

        if x.shape != mask.shape:
            raise ValueError(
                f"Feature shape {x.shape} does not match "
                f"mask shape {mask.shape}"
            )

        values = x[mask]

        if values.numel() == 0:
            raise ValueError(
                "Breast mask contains no valid pixels."
            )

        median = torch.median(values)

        mad = torch.median(
            torch.abs(values - median)
        )

        denominator = 5.0 * mad

        # ------------------------------------------------------
        # Degenerate case
        # ------------------------------------------------------

        if denominator.abs() < epsilon:

            normalized = torch.zeros_like(x)

            normalized[mask] = (
                x[mask] > median
            ).float()

            return normalized

        # ------------------------------------------------------
        # Robust normalization
        # ------------------------------------------------------

        normalized = (
            x - median
        ) / (
            denominator + epsilon
        )

        normalized = torch.clamp(
            normalized,
            min=0.0,
            max=1.0,
        )

        normalized[~mask] = 0.0

        return normalized

    @staticmethod
    def _scale_max(
        feature: torch.Tensor,
    ) -> torch.Tensor:

        if feature.ndim == 2:
            return feature

        if feature.ndim != 3:
            raise ValueError(
                f"Expected [H,W] or [S,H,W], "
                f"got {feature.shape}"
            )

        return torch.max(
            feature,
            dim=0,
        ).values

    def apply(
        self,
        top_hat: Image,
        log: Image,
        dog: Image,
        breast_mask: Image,
        output: Image,
    ) -> Image:

        device = torch.device(self.device)

        # ======================================================
        # 1. Breast mask
        # ======================================================

        mask = breast_mask.pixel_data.to(
            device=device,
            dtype=torch.bool,
        )

        while mask.ndim > 2:
            mask = mask.squeeze(0)

        if mask.ndim != 2:
            raise ValueError(
                f"Expected breast mask [H,W], "
                f"got {mask.shape}"
            )

        # ======================================================
        # 2. Load feature maps
        # ======================================================

        top_hat_data = top_hat.pixel_data.to(
            device=device,
            dtype=torch.float32,
        )

        log_data = log.pixel_data.to(
            device=device,
            dtype=torch.float32,
        )

        dog_data = dog.pixel_data.to(
            device=device,
            dtype=torch.float32,
        )

        # ======================================================
        # 3. Top-Hat evidence
        # ======================================================

        top_hat_max = self._scale_max(
            top_hat_data
        )

        T = self._robust_normalize(
            top_hat_max,
            mask=mask,
        )

        # ======================================================
        # 4. LoG evidence
        # ======================================================

        log_max = self._scale_max(
            torch.abs(log_data)
        )

        L = self._robust_normalize(
            log_max,
            mask=mask,
        )

        # ======================================================
        # 5. DoG evidence
        # ======================================================

        # DoG now contains ONLY the DoG response.
        #
        # Expected representation:
        #
        #     [1, H, W]
        #
        # or:
        #
        #     [H, W]
        #
        # Channel 0 is the actual DoG response.

        if dog_data.ndim == 3:

            if dog_data.shape[0] != 1:
                raise ValueError(
                    "CandidateEvidenceAlgorithm expects "
                    "DoG-only input with shape [1,H,W]. "
                    f"Got {dog_data.shape}"
                )

            F = dog_data[0]

        elif dog_data.ndim == 2:

            F = dog_data

        else:

            raise ValueError(
                f"Invalid DoG shape: "
                f"{dog_data.shape}"
            )

        # DoG can contain positive and negative responses.
        #
        # For bright compact structures, use magnitude.
        F = torch.abs(F)

        F = self._robust_normalize(
            F,
            mask=mask,
        )

        # ======================================================
        # 6. Spatial validation
        # ======================================================

        shape = T.shape

        for name, feature in (
            ("LoG", L),
            ("DoG", F),
        ):

            if feature.shape != shape:
                raise ValueError(
                    f"{name} feature has shape "
                    f"{feature.shape}, expected {shape}"
                )

        # ======================================================
        # 7. Evidence score
        # ======================================================

        score = (
            self.w_top_hat * T
            + self.w_log * L
            + self.w_dog * F
        )

        # Outside breast = no evidence.
        score[~mask] = 0.0

        # ======================================================
        # 8. Output
        # ======================================================

        output.pixel_data = score

        return output