import numpy as np
from scipy import ndimage

from medical_image.data.image import Image


class MorpohologyOperations:
    # TODO: Update docstrings
    @staticmethod
    def morphoogy_closing(image: Image, output: Image):
        """
        Two-dimensional binary closing with the given structuring element.

        The *closing* of an input image by a structuring element is the
        *erosion* of the *dilation* of the image by the structuring element.
        For more information:
            https://en.wikipedia.org/wiki/Closing_%28morphology%29
            https://en.wikipedia.org/wiki/Mathematical_morphology

        Parameters:
            input : 2d ndarray
                Binary array_like to be closed. Non-zero (True) elements form
                the subset to be closed.

        Returns:
            binary_closing : ndarray of bools
                Closing of the input by the structuring element.

        Examples:
            >>> a = np.zeros((5,5), dtype=int)
            >>> a[1:-1, 1:-1] = 1; a[2,2] = 0
            >>> a
            array([[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 0, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]])
            >>> # Closing removes small holes
            >>> PixelArrayOperation.morphoogy_closing(a)
            array([[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]])

            >>> a = np.zeros((7,7), dtype=int)
            >>> a[1:6, 2:5] = 1; a[1:3,3] = 0
            >>> a
            array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
            >>> # In addition to removing holes, closing can also
            >>> # coarsen boundaries with fine hollows.
            >>> PixelArrayOperation.morphoogy_closing(a)
            array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
        """
        output.pixel_data = ndimage.binary_closing(
            image.pixel_data, structure=np.ones((7, 7))
        ).astype(int)

    @staticmethod
    def region_fill(image: Image, output: Image):
        """
        Fill the holes in binary images.
        For more:
            https://en.wikipedia.org/wiki/Mathematical_morphology
        Parameters:
            input : array_like
                2-D binary array with holes to be filled

        Returns:
            an ndarray
            Transformation of the initial image `input` where holes have been
            filled.

        Notes:
            The algorithm used in this function consists in invading the complementary
            of the shapes in `input` from the outer boundary of the image,
            using binary dilations. Holes are not connected to the boundary and are
            therefore not invaded. The result is the complementary subset of the
            invaded region.


        Examples:

        >>> a = np.zeros((5, 5), dtype=int)
        >>> a[1:4, 1:4] = 1
        >>> a[2,2] = 0
        >>> a
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])
        >>> PixelArrayOperation.region_fill(a)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])

        """
        output.pixel_data = ndimage.binary_fill_holes(
            image.pixel_data, structure=np.ones((7, 7))
        ).astype(int)
