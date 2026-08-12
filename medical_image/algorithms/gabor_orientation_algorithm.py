import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.filters import Filters

class GaborOrientationAlgorithm(Algorithm):
    """
    Gabor orientation analysis for microcalcification candidate
    discrimination.

    Gabor responses are used primarily as negative evidence.

    A compact particle tends to produce relatively isotropic
    responses across orientations.

    An elongated or vessel-like structure tends to produce a
    dominant orientation.

    Output:

        [6, H, W]

    Channels:

        0 -> G0
        1 -> G45
        2 -> G90
        3 -> G135
        4 -> Gmax
        5 -> orientation anisotropy
    """

    def __init__(
        self,
        orientations=(0.0, 45.0, 90.0, 135.0),
        frequency=0.25,
        sigma=None,
        gamma=1.0,
        epsilon=1e-6,
        device="cpu",
    ):
        super().__init__(
            device=device
        )

        if not orientations:
            raise ValueError(
                "At least one orientation is required."
            )

        if frequency <= 0:
            raise ValueError(
                "frequency must be positive."
            )

        if sigma is not None and sigma <= 0:
            raise ValueError(
                "sigma must be positive."
            )

        if gamma <= 0:
            raise ValueError(
                "gamma must be positive."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive."
            )

        self.orientations = tuple(
            orientations
        )

        self.frequency = frequency
        self.sigma = sigma
        self.gamma = gamma
        self.epsilon = epsilon

        # ----------------------------------------------------------
        # Gabor filter bank
        # ----------------------------------------------------------

        self.gabor = (
            lambda img, out:
            Filters.gabor_orientation(
                image=img,
                output=out,
                orientations=self.orientations,
                frequency=self.frequency,
                sigma=self.sigma,
                gamma=self.gamma,
                device=self.device,
            )
        )

    def apply(
        self,
        image: Image,
        output: Image,
    ) -> Image:

        # ----------------------------------------------------------
        # Step 1: Gabor orientation responses
        # ----------------------------------------------------------

        gabor = InMemoryImage(
            array=torch.zeros(
                (
                    len(self.orientations),
                    image.height,
                    image.width,
                ),
                dtype=torch.float32,
                device=image.pixel_data.device,
            )
        )

        self.gabor(
            image,
            gabor,
        )

        responses = gabor.pixel_data

        # ----------------------------------------------------------
        # Step 2: Maximum orientation response
        # ----------------------------------------------------------

        gmax, orientation_index = (
            responses.max(
                dim=0
            )
        )

        # ----------------------------------------------------------
        # Step 3: Minimum orientation response
        # ----------------------------------------------------------

        gmin = responses.min(
            dim=0
        ).values

        # ----------------------------------------------------------
        # Step 4: Orientation anisotropy
        # ----------------------------------------------------------

        anisotropy = (
            gmax - gmin
        ) / (
            gmax + self.epsilon
        )

        # ----------------------------------------------------------
        # Step 5: Final representation
        # ----------------------------------------------------------

        output.pixel_data = torch.cat(
            [
                responses,
                gmax.unsqueeze(0),
                anisotropy.unsqueeze(0),
            ],
            dim=0,
        )

        return output