import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image


class CandidateGenerationAlgorithm(Algorithm):

    SUPPORTED_THRESHOLD_METHODS = {
        "percentile",
        "otsu",
    }

    def __init__(
        self,
        threshold_method: str = "percentile",
        percentile: float = 0.995,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        threshold_method = threshold_method.lower()

        if threshold_method not in self.SUPPORTED_THRESHOLD_METHODS:
            raise ValueError(
                f"Unsupported threshold method: "
                f"{threshold_method!r}. "
                f"Supported methods: "
                f"{sorted(self.SUPPORTED_THRESHOLD_METHODS)}"
            )

        if not 0.0 < percentile < 1.0:
            raise ValueError(
                f"percentile must be in (0, 1), "
                f"got {percentile}"
            )

        self.threshold_method = threshold_method
        self.percentile = percentile

    def _percentile_threshold(
        self,
        evidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        values = evidence[mask]

        if values.numel() == 0:
            raise ValueError(
                "Breast mask contains no pixels."
            )

        threshold = torch.quantile(
            values,
            self.percentile,
        )

        return threshold

    def _otsu_threshold(
        self,
        evidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        values = evidence[mask]

        if values.numel() == 0:
            raise ValueError(
                "Breast mask contains no pixels."
            )

        # ------------------------------------------------------
        # Otsu on the continuous evidence values
        # ------------------------------------------------------

        values = values.float()

        min_value = values.min()
        max_value = values.max()

        if torch.isclose(
            min_value,
            max_value,
        ):
            return min_value

        # Normalize values to [0, 255]
        normalized = (
            (values - min_value)
            / (max_value - min_value)
        )

        bins = 256

        histogram = torch.histc(
            normalized,
            bins=bins,
            min=0.0,
            max=1.0,
        )

        probability = (
            histogram
            / histogram.sum()
        )

        indices = torch.arange(
            bins,
            device=values.device,
            dtype=torch.float32,
        )

        omega = torch.cumsum(
            probability,
            dim=0,
        )

        mu = torch.cumsum(
            probability * indices,
            dim=0,
        )

        total_mean = mu[-1]

        denominator = (
            omega
            * (1.0 - omega)
        )

        numerator = (
            total_mean * omega
            - mu
        ).pow(2)

        between_class_variance = (
            numerator
            / (denominator + 1e-8)
        )

        threshold_index = torch.argmax(
            between_class_variance
        )

        threshold_normalized = (
            threshold_index.float()
            / (bins - 1)
        )

        threshold = (
            min_value
            + threshold_normalized
            * (max_value - min_value)
        )

        return threshold

    def apply(
        self,
        evidence: Image,
        breast_mask: Image,
        output: Image,
    ) -> Image:

        device = torch.device(self.device)

        # ======================================================
        # 1. Evidence
        # ======================================================

        evidence_data = evidence.pixel_data.to(
            device=device,
            dtype=torch.float32,
        )

        # Evidence must be 2D.
        while evidence_data.ndim > 2:
            evidence_data = evidence_data.squeeze(0)

        if evidence_data.ndim != 2:
            raise ValueError(
                f"Expected evidence [H,W], "
                f"got {evidence_data.shape}"
            )

        # ======================================================
        # 2. Breast mask
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

        if evidence_data.shape != mask.shape:
            raise ValueError(
                f"Evidence shape {evidence_data.shape} "
                f"does not match breast mask "
                f"{mask.shape}"
            )

        # ======================================================
        # 3. Remove invalid evidence
        # ======================================================

        valid = mask & torch.isfinite(
            evidence_data
        )

        if not valid.any():
            raise ValueError(
                "No valid evidence pixels inside "
                "the breast mask."
            )

        # ======================================================
        # 4. Threshold evidence
        # ======================================================

        if self.threshold_method == "percentile":

            threshold = self._percentile_threshold(
                evidence_data,
                valid,
            )

        elif self.threshold_method == "otsu":

            threshold = self._otsu_threshold(
                evidence_data,
                valid,
            )

        else:
            raise RuntimeError(
                f"Unknown threshold method: "
                f"{self.threshold_method}"
            )

        # ======================================================
        # 5. Generate candidates
        # ======================================================

        candidate_map = (
            evidence_data >= threshold
        ) & valid

        # ======================================================
        # 6. Debug statistics
        # ======================================================

        print(
            "Candidate generation:",
            flush=True,
        )

        print(
            "  threshold:",
            threshold.item(),
            flush=True,
        )

        print(
            "  evidence min:",
            evidence_data[valid].min().item(),
            flush=True,
        )

        print(
            "  evidence max:",
            evidence_data[valid].max().item(),
            flush=True,
        )

        print(
            "  breast pixels:",
            valid.sum().item(),
            flush=True,
        )

        print(
            "  candidate pixels:",
            candidate_map.sum().item(),
            flush=True,
        )

        print(
            "  candidate ratio:",
            (
                candidate_map.sum().float()
                / valid.sum().float()
            ).item(),
            flush=True,
        )

        # ======================================================
        # 7. Output
        # ======================================================

        output.pixel_data = (
            candidate_map.to(torch.uint8)
        )

        return output