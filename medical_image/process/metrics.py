import numpy as np

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
        joint_histogram, _, _ = np.histogram2d(image1.pixel_data.flatten(), image2.pixel_data.flatten())
        joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
        joint_prob = joint_histogram_without_zero / (image2.width * image2.height)

        return np.around(-np.sum(joint_prob * np.log2(joint_prob)), decimals=decimals)

    @staticmethod
    def mutual_information(image1: Image, image2: Image, decimals=4):
        mi = (
                Metrics.entropy(image1, decimals=decimals)
                + Metrics.entropy(image2, decimals=decimals)
                - Metrics.joint_entropy(image1, image2,decimals=decimals)
        )
        return mi