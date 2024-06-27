import numpy as np

from medical_image.data.image import Image


class Metrics:
    @staticmethod
    def entropy(image: Image, decimals = 4):
        """
        This function calculates Shannon Entropy of an image.
        For more information about the Entropy this link:
        https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Parameters:
            input: 2d ndarray to process.

        Returns:
            entropy: float rounded to 4 decimal places

        Notes:
            The logarithm used is the bit logarithm (base-2).

        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (512,512))
            >>> ent = entropy_main(a)
            >>> ent
            11.9883
        """
        image_array = image.pixel_data
        # Flatten the input to a 1D array for histogram calculation
        flat_image_array = image_array.flatten()
        histogram, _ = np.histogram(
            flat_image_array,
            bins=np.arange(flat_image_array.min(), flat_image_array.max() + 2) - 0.5,
        )

        # Calculate probabilities
        probabilities = histogram / flat_image_array.size

        # Filter out zero probabilities to avoid log2(0)
        probabilities = probabilities[probabilities > 0]

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return np.around(entropy, decimals=decimals)
    @staticmethod
    def joint_entorpy(image1: Image, image2: Image):
        pass

    @staticmethod
    def mutual_information(image1: Image, image2: Image):
        pass