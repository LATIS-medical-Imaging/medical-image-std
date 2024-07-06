from typing import Union

import numpy as np
from scipy import ndimage

from medical_image.data.image import Image


class Metrics:
    def entropy(image: Image, decimals=4):
        """
        This function calculates the Shannon Entropy of an image.
        For more information about entropy, see:
        https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Parameters:
            image (Image): An Image object containing the 2D ndarray to process.
            decimals (int, optional): Number of decimal places to round the entropy to. Default is 4.

        Returns:
            float: Entropy value rounded to the specified number of decimal places.

        Notes:
            The logarithm used is the bit logarithm (base-2).

        Examples:
            >>> import numpy as np
            >>> class ExampleImage(Image):
            >>>     def load(self):
            >>>         self.width = 512
            >>>         self.height = 512
            >>>         self.pixel_data = np.random.randint(0, 4095, (self.width, self.height))
            >>>     def save(self):
            >>>         pass
            >>> example_image = ExampleImage("example_path")
            >>> example_image.load()
            >>> ent = entropy(example_image)
            >>> print(ent)
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
    def joint_entropy(image1: Image, image2: Image, decimals=4):
        """
        This function calculates the joint Shannon Entropy of two images.
        For more information about joint entropy, see:
        https://en.wikipedia.org/wiki/Joint_entropy

        Parameters:
            image1 (Image): The first Image object containing the 2D ndarray to process.
            image2 (Image): The second Image object containing the 2D ndarray to process.
            decimals (int, optional): Number of decimal places to round the entropy to. Default is 4.

        Returns:
            float: Joint entropy value rounded to the specified number of decimal places.

        Notes:
            The logarithm used is the bit logarithm (base-2).

        Examples:
            >>> import numpy as np
            >>> class ExampleImage(Image):
            >>>     def load(self):
            >>>         self.width = 512
            >>>         self.height = 512
            >>>         self.pixel_data = np.random.randint(0, 4095, (self.width, self.height))
            >>>     def save(self):
            >>>         pass
            >>> image1 = ExampleImage("example_path1")
            >>> image2 = ExampleImage("example_path2")
            >>> image1.load()
            >>> image2.load()
            >>> joint_ent = joint_entropy(image1, image2)
            >>> print(joint_ent)
            6.6435
        """
        joint_histogram, _, _ = np.histogram2d(
            image1.pixel_data.flatten(), image2.pixel_data.flatten()
        )
        joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
        joint_prob = joint_histogram_without_zero / (image2.width * image2.height)

        return np.around(-np.sum(joint_prob * np.log2(joint_prob)), decimals=decimals)

    @staticmethod
    def mutual_information(image1: Image, image2: Image, decimals=4):
        """
        This function calculates the mutual information between two images.
        For more information about mutual information, see:
        https://en.wikipedia.org/wiki/Mutual_information

        Parameters:
            image1 (Image): The first Image object containing the 2D ndarray to process.
            image2 (Image): The second Image object containing the 2D ndarray to process.
            decimals (int, optional): Number of decimal places to round the mutual information to. Default is 4.

        Returns:
            float: Mutual information value rounded to the specified number of decimal places.

        Notes:
            The calculation is based on the entropy and joint entropy of the two images.

        Examples:
            >>> import numpy as np
            >>> class ExampleImage(Image):
            >>>     def load(self):
            >>>         self.width = 512
            >>>         self.height = 512
            >>>         self.pixel_data = np.random.randint(0, 4095, (self.width, self.height))
            >>>     def save(self):
            >>>         pass
            >>> image1 = ExampleImage("example_path1")
            >>> image2 = ExampleImage("example_path2")
            >>> image1.load()
            >>> image2.load()
            >>> mi = mutual_information(image1, image2)
            >>> print(mi)
            17.3331
        """
        mi = (
            Metrics.entropy(image1, decimals=decimals)
            + Metrics.entropy(image2, decimals=decimals)
            - Metrics.joint_entropy(image1, image2, decimals=decimals)
        )
        return mi
    # tODO: Update Docstring
    @staticmethod
    def local_variance(image: Image, output: Image, kernel: Union[float, tuple]) -> np.ndarray:
        """
        Calculate the variance a specified sub-regions in image.

        Parameters:
            input: 2d ndarray to process.
            kernel: size of sub-region
        Returns:
            2D ndarray with the same size as the input contains the local variance of each region with size = kernel


        Examples:
            >>> a = np.random.randint(0, 5, (9,9))
            >>> Metrics.local_variance(a, 3)
            array([[0, 1, 1, 2, 1, 1, 1, 1, 2],
                   [0, 1, 2, 2, 1, 1, 1, 1, 1],
                   [0, 0, 1, 1, 2, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1, 1, 1, 1, 0],
                   [2, 1, 0, 1, 1, 2, 2, 2, 1],
                   [3, 2, 1, 0, 1, 2, 1, 1, 0],
                   [2, 2, 1, 0, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 0, 1, 2, 1, 1],
                   [0, 0, 0, 0, 0, 1, 2, 2, 1]])

        """
        output.pixel_data = ndimage.generic_filter(image.pixel_data, np.var, size=kernel)

    # tODO: Update Docstring
    @staticmethod
    def variance(image: Image, output: Image):
        """
        Calculate the variance of the values of an 2-D image array

        Parameters:
            input: 2d ndarray to process.

        Returns:
            variance : float


        Examples:
            >>> a = np.array([[1, 2, 0, 0],
            ...               [5, 3, 0, 4],
            ...               [0, 0, 0, 7],
            ...               [9, 3, 0, 0]])
            >>> Metrics.variance(a)
            7.609375
        """
        output.pixel_data = ndimage.variance(image.pixel_data)
