import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.morphology import MorphologyOperations


class TopHatAlgorithm(Algorithm):
    """
    White Top-Hat enhancement algorithm for microcalcification detection.

    Highlights bright structures (e.g., microcalcifications) that are smaller than
    the selected structuring element.

    References:
        @article{quintanilla2011image,
          title={Image segmentation by fuzzy and possibilistic clustering algorithms for the identification of microcalcifications},
          author={Quintanilla-Dom{\\'i}nguez, Joel and Ojeda-Maga{\\~n}a, Benjam{\\'i}n and Cortina-Januchs, Maria Guadalupe and Ruelas, Rub{\\'e}n and Vega-Corona, Antonio and Andina, Diego},
          journal={Scientia Iranica},
          volume={18},
          number={3},
          pages={580--589},
          year={2011},
          publisher={Elsevier}
        }

    Math and Logic:
        The White Top-Hat transform of an image I is defined as the difference between
        the original image and its morphological opening using a structuring element (SE):
        TopHat(I) = I - opening(I, SE)

        This retains the high-intensity peaks (calcifications) while removing the larger
        background structures (healthy tissue).

    Pipeline:
        1. Create a disk structuring element of the specified radius.
        2. Perform morphological opening (erosion followed by dilation).
        3. Subtract the opened image from the original image.

    Example Usage:
        ```python
        import copy
        import numpy as np
        import matplotlib.pyplot as plt
        from medical_image.algorithms.top_hat import TopHatAlgorithm
        from medical_image.data.dicom_image import DicomImage

        # Load and prepare image
        img = DicomImage("20527054.dcm")
        img.load()

        # Initialize output and apply algorithm
        output = copy.deepcopy(img)
        algo = TopHatAlgorithm(radius=4, device="cpu")
        algo(img, output)

        # Plot output
        plt.imshow(output.pixel_data.numpy(), cmap='gray')
        plt.title('Top Hat Output')
        plt.show()
        ```

    Args:
        radius: Disk SE radius (default 4 -> 9x9 footprint).
        device: Torch device (e.g. "cpu", "cuda:0").
    """

    def __init__(self, radius: int = 4, device: str = "cpu"):
        super().__init__(device=device)
        self.radius = radius

        # Operations (FEBDS-style lambdas)
        self.top_hat = lambda img, out: MorphologyOperations.white_top_hat(
            image_data=img, output=out, radius=self.radius, device=self.device
        )

    def apply(self, image: Image, output: Image):
        """
        Apply white top-hat to the image.

        Args:
            image: Input Image (2D float, e.g. normalized [0,1]).
            output: Output Image — pixel_data will contain top-hat result.
        """
        self.top_hat(image, output)
