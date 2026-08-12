import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.filters import Filters
from medical_image.process.frequency import FrequencyOperations


class DoG(Algorithm):
    """
    Difference-of-Gaussians frequency feature for
    microcalcification candidate detection.

    Computes:

        DoG = G_sigma1(I) - G_sigma2(I)

    Output:

        [1, H, W]

    Channel:

        0 -> DoG response
    """

    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        if sigma1 <= 0:
            raise ValueError(
                "sigma1 must be > 0."
            )

        if sigma2 <= sigma1:
            raise ValueError(
                "sigma2 must be > sigma1."
            )

        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # ----------------------------------------------------------
        # Difference of Gaussians
        # ----------------------------------------------------------

        self.dog = lambda img, out: (
            Filters.difference_of_gaussian(
                image=img,
                output=out,
                low_sigma=self.sigma1,
                high_sigma=self.sigma2,
                device=self.device,
            )
        )


    def apply(
        self,
        image: Image,
        output: Image,
    ) -> Image:

        # ----------------------------------------------------------
        # Step 1: Difference of Gaussians
        # ----------------------------------------------------------

        dog = InMemoryImage(
            array=torch.zeros_like(
                image.pixel_data
            )
        )

        self.dog(
            image,
            dog,
        )


        # ----------------------------------------------------------
        # Step 4: Build frequency representation
        # ----------------------------------------------------------

        output.pixel_data = torch.stack(
            [
                dog.pixel_data,
            ],
            dim=0,
        )

        return output