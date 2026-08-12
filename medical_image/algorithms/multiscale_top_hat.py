import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.morphology import MorphologyOperations


class MultiScaleTopHatAlgorithm(Algorithm):
    """
    Multi-Scale White Top-Hat Enhancement Algorithm.

    Computes the white Top-Hat transform at multiple spatial scales:

        TH_r = I - (I ∘ B_r)

    where:
        I   = normalized mammogram
        B_r = circular disk structuring element of radius r
        ∘   = morphological opening

    The complete scale-space response is preserved.

    Output:
        output.pixel_data -> Tensor of shape [N, H, W]

        where N = number of radii and:

            output.pixel_data[0] = TH_r1
            output.pixel_data[1] = TH_r2
            ...

    Derived features:
        - Maximum Top-Hat response
        - Scale of maximum response
        - Persistence across scales
        - Scale bandwidth

    The input image should normally be the robustly normalized
    mammogram I_norm in [0, 1].
    """

    def __init__(
        self,
        radii=(1, 2, 3, 4, 5, 6),
        persistence_threshold=0.05,
        bandwidth_ratio=0.5,
        device="cpu",
    ):
        super().__init__(device=device)

        if not radii:
            raise ValueError(
                "At least one radius is required."
            )

        if any(r <= 0 for r in radii):
            raise ValueError(
                "All radii must be positive."
            )

        if not 0.0 <= persistence_threshold <= 1.0:
            raise ValueError(
                "persistence_threshold must be in [0, 1]."
            )

        if not 0.0 < bandwidth_ratio <= 1.0:
            raise ValueError(
                "bandwidth_ratio must be in (0, 1]."
            )

        self.radii = tuple(radii)
        self.persistence_threshold = persistence_threshold
        self.bandwidth_ratio = bandwidth_ratio

        # ----------------------------------------------------------
        # Morphological operations
        # ----------------------------------------------------------

        self.white_top_hat = (
            lambda img, out, radius:
            MorphologyOperations.white_top_hat(
                image=img,
                output=out,
                radius=radius,
                device=self.device,
            )
        )

    def apply(
        self,
        image: Image,
        output: Image,
    ) -> Image:
        """
        Apply multi-scale white Top-Hat transformation.

        The output contains the complete Top-Hat scale signature.

        Args:
            image:
                Normalized mammogram I_norm.

            output:
                Output Image. Its pixel_data will have shape
                [num_scales, H, W].

        Returns:
            The output Image.
        """

        responses = []

        # ----------------------------------------------------------
        # Step 1: Compute Top-Hat at every scale
        # ----------------------------------------------------------

        for radius in self.radii:

            th = InMemoryImage(
                array=torch.zeros_like(
                    image.pixel_data
                )
            )

            self.white_top_hat(
                image,
                th,
                radius,
            )

            responses.append(
                th.pixel_data
            )

        # ----------------------------------------------------------
        # Step 2: Build scale signature
        # ----------------------------------------------------------

        signature = torch.stack(
            responses,
            dim=0,
        )

        # [num_scales, H, W]
        output.pixel_data = signature

        return output