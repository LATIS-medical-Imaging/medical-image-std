import numpy as np

from medical_image.data.image import Image


class FrequencyOperations:
    # TODO: update Docstring
    @staticmethod
    def fft(image: Image, output: Image):
        """
        This function calculates 2-dimensional discrete Fourier Transform using Fast Fourier Transform Algorithms (FFT)
        For more information about the Entropy this link:
        https://en.wikipedia.org/wiki/Fast_Fourier_transform

        Parameters:
            input: 2d ndarray to process.

        Returns:
            out: complex ndarray


        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (3,3))
            >>> fft = PixelArrayOperation.fft(a)
            >>> fft
            array([[19218.    +0.j        ,  1506. +1307.69835971j,
                     1506. -1307.69835971j],
                   [ 1455. +2527.06212824j,  2893.5 +995.06318895j,
                     1290.  +299.64478971j],
                   [ 1455. -2527.06212824j,  1290.  -299.64478971j,
                     2893.5 -995.06318895j]])

        """
        output.pixel_data = np.fft.fft2(image.pixel_data)

    @staticmethod
    def inverse_fft(image: Image, output: Image):
        """
        This function calculates the inverse of 2-dimensional discrete Fourier Transform using Fast Fourier Transform Algorithms (FFT)
        For more information about the Entropy this link:
        https://en.wikipedia.org/wiki/Fast_Fourier_transform

        Parameters:
            input: 2d ndarray (it can be complex) to process.

        Returns:
            out: complex ndarray


        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (3,3))
            >>> ifft = PixelArrayOperation.inverse_fft(a)
            >>> ifft
            array([[19218.    +0.j        ,  1506. +1307.69835971j,
                     1506. -1307.69835971j],
                   [ 1455. +2527.06212824j,  2893.5 +995.06318895j,
                     1290.  +299.64478971j],
                   [ 1455. -2527.06212824j,  1290.  -299.64478971j,
                     2893.5 -995.06318895j]])

        """
        output.pixel_data = np.fft.ifft2(image.pixel_data)
