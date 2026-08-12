import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.filters import Filters
from medical_image.process.differential import DifferentialOperations


class DifferentialBlobAlgorithm(Algorithm):
    """
    Multi-scale differential/blob analysis for microcalcification detection.

    Computes two complementary descriptions:

    1. Multi-scale Laplacian of Gaussian (LoG)

        LoG_sigma(I)

    2. Hessian-based blobness analysis

        H = [[Ixx, Ixy],
             [Ixy, Iyy]]

    followed by the Hessian eigenvalues:

        lambda1, lambda2

    and anisotropy:

        AH = (|lambda1| + |lambda2|)
             / (|lambda1 - lambda2| + epsilon)

    A compact blob tends to have similar curvature in both
    directions, while a line-like structure has asymmetric
    curvature.

    Output shape:

        [num_log_scales + 3, H, W]

    Channels:

        0 ... N-1 -> LoG responses
        N         -> lambda1
        N+1       -> lambda2
        N+2       -> Hessian anisotropy/blobness
    """

    def __init__(
        self,
        sigmas=(1.0, 2.0, 3.0),
        epsilon=1e-6,
        device="cpu",
    ):
        super().__init__(device=device)

        if not sigmas:
            raise ValueError(
                "At least one LoG scale is required."
            )

        if any(sigma <= 0 for sigma in sigmas):
            raise ValueError(
                "All sigma values must be positive."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive."
            )

        self.sigmas = tuple(sigmas)
        self.epsilon = epsilon

        # ----------------------------------------------------------
        # LoG
        # ----------------------------------------------------------

        self.log = lambda img, out, sigma: (
            Filters.laplacian_of_gaussian(
                image=img,
                output=out,
                sigma=sigma,
                device=self.device,
            )
        )

        # ----------------------------------------------------------
        # Hessian
        # ----------------------------------------------------------

        self.hessian = lambda img, out: (
            DifferentialOperations.hessian(
                image=img,
                output=out,
                device=self.device,
            )
        )

        # ----------------------------------------------------------
        # Hessian eigenvalues
        # ----------------------------------------------------------

        self.hessian_eigenvalues = (
            lambda hessian:
            DifferentialOperations.hessian_eigenvalues(
                hessian=hessian,
            )
        )

    def apply(
        self,
        image: Image,
        output: Image,
    ) -> Image:

        # ----------------------------------------------------------
        # Step 1: Multi-scale LoG
        # ----------------------------------------------------------

        log_responses = []

        for sigma in self.sigmas:

            log = InMemoryImage(
                array=torch.zeros_like(
                    image.pixel_data
                )
            )

            self.log(
                image,
                log,
                sigma,
            )

            log_response = log.pixel_data

            while log_response.ndim > 2:
                log_response = log_response.squeeze(0)

            if log_response.ndim != 2:
                raise ValueError(
                    f"Expected 2D LoG response, got {log_response.shape}"
                )

            log_responses.append(log_response)

        log_signature = torch.stack(
            log_responses,
            dim=0,
        )

        # ----------------------------------------------------------
        # Step 2: Hessian
        # ----------------------------------------------------------

        hessian = InMemoryImage(
            array=torch.zeros(
                (
                    3,
                    image.height,
                    image.width,
                ),
                dtype=torch.float32,
                device=image.pixel_data.device,
            )
        )

        self.hessian(
            image,
            hessian,
        )

        # ----------------------------------------------------------
        # Step 3: Hessian eigenvalues
        # ----------------------------------------------------------

        lambda1, lambda2 = self.hessian_eigenvalues(
            hessian.pixel_data
        )

        # ----------------------------------------------------------
        # Step 4: Hessian anisotropy/blobness
        # ----------------------------------------------------------

        anisotropy = (
            torch.abs(lambda1)
            + torch.abs(lambda2)
        ) / (
            torch.abs(lambda1 - lambda2)
            + self.epsilon
        )

        # ----------------------------------------------------------
        # Step 5: Final representation
        # ----------------------------------------------------------

        output.pixel_data = torch.cat(
            [
                log_signature,
                lambda1.unsqueeze(0),
                lambda2.unsqueeze(0),
                anisotropy.unsqueeze(0),
            ],
            dim=0,
        )

        return output