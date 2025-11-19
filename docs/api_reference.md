# API Reference

## Table of Contents

1. [Data Module](#data-module)
2. [Process Module](#process-module)
3. [Algorithms Module](#algorithms-module)
4. [Utils Module](#utils-module)

---

## Data Module

### `medical_image.data.image`

#### `Image` (Abstract Base Class)

Base class for all medical image types.

**Attributes:**
- `file_path` (str): Path to the image file
- `width` (int): Image width in pixels
- `height` (int): Image height in pixels
- `pixel_data` (torch.Tensor): Image pixel data as PyTorch tensor
- `label` (Annotation): Associated annotation/label
- `device` (str): Computation device ('cuda' or 'cpu')

**Methods:**

##### `__init__(file_path: str)`
Initialize an Image object.

**Parameters:**
- `file_path` (str): Path to the image file

**Raises:**
- `FileNotFoundError`: If the file does not exist

##### `load()` (Abstract)
Load image data from file. Must be implemented by subclasses.

##### `save()` (Abstract)
Save image data to file. Must be implemented by subclasses.

##### `display_info()`
Display basic information about the image (file path, dimensions).

##### `to_png()`
Save the image as PNG format.

**Raises:**
- `ValueError`: If pixel_data is None or invalid

##### `plot(cmap='gray')`
Display the image using matplotlib.

**Parameters:**
- `cmap` (str): Colormap to use. Default: 'gray'

##### `to_numpy() -> np.ndarray`
Convert pixel data to NumPy array.

**Returns:**
- `np.ndarray`: Image data as NumPy array

---

### `medical_image.data.dicom_image`

#### `DicomImage`

DICOM image implementation.

**Inherits:** `Image`

**Additional Attributes:**
- `dicom_data` (pydicom.Dataset): DICOM metadata and data

**Methods:**

##### `__init__(file_path: str)`
Initialize a DICOM image.

**Parameters:**
- `file_path` (str): Path to .dcm file

**Raises:**
- `ValueError`: If file extension is not .dcm

##### `load()`
Load DICOM data and pixel array.

##### `save()`
Save modified DICOM data to a new file (with '_modified' suffix).

**Raises:**
- `ValueError`: If DICOM data not loaded

---

### `medical_image.data.patch`

#### `Patch`

Represents a single patch extracted from an image.

**Attributes:**
- `parent` (Image): Reference to parent Image
- `row_idx` (int): Patch row index in grid
- `col_idx` (int): Patch column index in grid
- `x` (int): Top-left x-coordinate in original image
- `y` (int): Top-left y-coordinate in original image
- `pixel_data` (torch.Tensor): Patch pixel data
- `is_padded` (bool): Whether patch includes padding
- `height` (int): Patch height
- `width` (int): Patch width

**Methods:**

##### `grid_id() -> Tuple[int, int]`
Get patch position in grid.

**Returns:**
- `Tuple[int, int]`: (row_idx, col_idx)

##### `pixel_position() -> Tuple[int, int]`
Get pixel coordinates in original image.

**Returns:**
- `Tuple[int, int]`: (x, y)

##### `to_numpy() -> np.ndarray`
Convert patch to NumPy array.

---

#### `PatchGrid`

Manages a grid of patches for an image.

**Attributes:**
- `parent` (Image): Parent image
- `patch_h` (int): Patch height
- `patch_w` (int): Patch width
- `patches` (List[Patch]): Flat list of all patches
- `grid` (List[List[Patch]]): 2D grid structure
- `pad_bottom` (int): Bottom padding added
- `pad_right` (int): Right padding added

**Methods:**

##### `__init__(parent_image: Image, patch_size: Tuple[int, int])`
Create a patch grid.

**Parameters:**
- `parent_image` (Image): Image to divide into patches
- `patch_size` (Tuple[int, int]): (height, width) of each patch

##### `reconstruct() -> torch.Tensor`
Reassemble full image from patches, removing padding.

**Returns:**
- `torch.Tensor`: Reconstructed image

---

### `medical_image.data.region_of_interest`

#### `RegionOfInterest`

Represents a Region of Interest in a medical image.

**Attributes:**
- `image` (Image): Original image
- `coordinates` (Union[List, np.ndarray]): ROI definition
- `annotation_type` (AnnotationType): Type of ROI

**Methods:**

##### `__init__(image: Image, coordinates: Union[List[int], List[Tuple[int, int]], np.ndarray])`
Create an ROI.

**Parameters:**
- `image` (Image): Source image
- `coordinates`: ROI definition
  - Bounding box: `[x_min, y_min, x_max, y_max]`
  - Polygon: `[(x1, y1), (x2, y2), ...]`
  - Mask: 2D NumPy array

##### `load() -> Image`
Extract and crop the ROI from the image.

**Returns:**
- `Image`: New Image object containing only the ROI

---

### `medical_image.data.medical_dataset`

#### `MedicalDataset` (Abstract Base Class)

PyTorch Dataset for medical images.

**Inherits:** `torch.utils.data.Dataset`, `ABC`

**Attributes:**
- `base_path` (str): Root directory of dataset
- `file_format` (str): Image file format (e.g., '.dcm')
- `transform` (Callable): Optional transform function
- `images_path` (List[str]): List of image file paths
- `current_image` (Image): Currently loaded image
- `train` (bool): Whether this is training data
- `test` (bool): Whether this is test data

**Methods:**

##### `__init__(base_path: str, file_format: str = '.dcm', transform: Optional[Callable] = None, train: bool = True, test: bool = False)`
Initialize the dataset.

##### `__len__() -> int`
Get dataset size.

##### `__getitem__(idx: int) -> Union[Tuple[torch.Tensor, Any], torch.Tensor]`
Get item at index.

**Returns:**
- `Tuple[torch.Tensor, label]` if labels exist, else `torch.Tensor`

##### `load_batch(batch_size: int) -> Image` (Abstract)
Load a batch of images.

##### `destroy_batch()` (Abstract)
Free memory from current batch.

##### `load_mask(image_path: str) -> Image`
Load corresponding mask for an image.

##### `apply_transform(transform, pixel_data, label)` (Abstract)
Apply transformations to data.

---

### `medical_image.data.cbis_ddsm`

#### `CbisDdsm`

Dataset loader for CBIS-DDSM mammography dataset.

**Inherits:** `MedicalDataset`

---

### `medical_image.utils.annotation`

#### `AnnotationType` (Enum)

Enumeration of annotation types.

**Values:**
- `BOUNDING_BOX`: Rectangular bounding box
- `POLYGON`: Polygon region
- `MASK`: Binary mask

---

#### `Annotation`

Stores annotation information for medical images.

**Attributes:**
- `annotation_type` (str): Type of annotation
- `coordinates` (List): List of coordinate sets
- `classes` (List[str]): Class labels
- `image_view` (str): Image view/perspective
- `abnormality_type` (str): Type of abnormality ('calcification' or 'mass')
- `pathology` (str): Pathology classification
- `calcification_type` (str, optional): Calcification type
- `calcification_distribution` (str, optional): Calcification distribution
- `mass_shape` (str, optional): Mass shape
- `mass_margin` (str, optional): Mass margin

**Methods:**

##### `__init__(...)`
Initialize annotation with all metadata.

**Raises:**
- `ValueError`: If required fields for abnormality type are missing

---

## Process Module

### `medical_image.process.filters`

#### `Filters`

Static methods for image filtering operations.

**Methods:**

##### `convolution(image_data: Image, output: Image, kernel: np.ndarray)`
Apply convolution filter with custom kernel.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `kernel` (np.ndarray): Convolution kernel (2D array)

##### `gaussian_filter(image_data: Image, output: Image, sigma: float)`
Apply Gaussian blur filter.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `sigma` (float): Standard deviation of Gaussian kernel

##### `median_filter(image_data: Image, output: Image, size: int)`
Apply median filter for noise reduction.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `size` (int): Filter window size (must be odd)

##### `butterworth_kernel(image_data: Image, D_0: int = 21, W: int = 32, n: int = 3) -> np.ndarray`
Generate Butterworth band-pass filter kernel.

**Parameters:**
- `image_data` (Image): Input image
- `D_0` (int): Center frequency. Default: 21
- `W` (int): Bandwidth. Default: 32
- `n` (int): Filter order. Default: 3

**Returns:**
- `np.ndarray`: Butterworth kernel

##### `difference_of_gaussian(image_data: Image, output: Image, sigma_1: float, sigma_2: float)`
Apply Difference of Gaussian (DoG) filter.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `sigma_1` (float): First Gaussian sigma
- `sigma_2` (float): Second Gaussian sigma

##### `laplacian_of_gaussian(image_data: Image, output: Image, sigma: float)`
Apply Laplacian of Gaussian (LoG) filter.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `sigma` (float): Gaussian sigma

##### `gamma_correction(image_data: Image, output: Image, gamma: float)`
Apply gamma correction for brightness adjustment.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `gamma` (float): Gamma value (> 1 brightens, < 1 darkens)

##### `ContrastAdjust(image_data: Image, output: Image, contrast: float, brightness: float)`
Adjust image contrast and brightness.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `contrast` (float): Contrast adjustment value
- `brightness` (float): Brightness adjustment value

---

### `medical_image.process.threshold`

#### `Threshold`

Static methods for image thresholding.

**Methods:**

##### `otsu_threshold(image_data: Image, output: Image = None)`
Apply Otsu's automatic thresholding method.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image, optional): Output image object

**Description:**
Automatically determines optimal threshold by maximizing between-class variance.

##### `sauvola_threshold(image_data: Image, output: Image = None, window_size: int = 10, k: float = 0.5, r: int = 128)`
Apply Sauvola's local adaptive thresholding.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image, optional): Output image object
- `window_size` (int): Local window size (must be odd). Default: 10
- `k` (float): Weighting factor. Default: 0.5
- `r` (int): Dynamic range. Default: 128

##### `binarize(image_data: Image, output: Image, alpha: float)`
Binarize image using local and global variance.

**Parameters:**
- `image_data` (Image): Input image
- `output` (Image): Output image object
- `alpha` (float): Scaling factor relating local and global variance

---

### `medical_image.process.metrics`

#### `Metrics`

Static methods for image quality metrics.

**Methods:**

##### `entropy(image: Image, decimals: int = 4) -> float`
Calculate Shannon entropy of image.

**Parameters:**
- `image` (Image): Input image
- `decimals` (int): Decimal places for rounding. Default: 4

**Returns:**
- `float`: Entropy value (in bits)

##### `joint_entropy(image1: Image, image2: Image, decimals: int = 4) -> float`
Calculate joint entropy of two images.

**Parameters:**
- `image1` (Image): First image
- `image2` (Image): Second image
- `decimals` (int): Decimal places. Default: 4

**Returns:**
- `float`: Joint entropy value

##### `mutual_information(image1: Image, image2: Image, decimals: int = 4) -> float`
Calculate mutual information between two images.

**Parameters:**
- `image1` (Image): First image
- `image2` (Image): Second image
- `decimals` (int): Decimal places. Default: 4

**Returns:**
- `float`: Mutual information value

##### `local_variance(image: Image, output: Image, kernel: Union[float, Tuple])`
Calculate local variance in specified sub-regions.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `kernel` (Union[float, Tuple]): Kernel size for local window

##### `variance(image: Image, output: Image)`
Calculate global variance of image.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object

---

### `medical_image.process.morphology`

#### `MorphologyOperations`

Static methods for morphological operations.

**Methods:**

##### `morphology_closing(image: Image, output: Image, kernel_size: int)`
Apply morphological closing operation.

##### `region_fill(image: Image, output: Image)`
Fill regions in binary image.

---

### `medical_image.process.frequency`

#### `FrequencyOperations`

Static methods for frequency domain operations.

**Methods:**

##### `fft(image: Image, output: Image)`
Compute Fast Fourier Transform.

##### `inverse_fft(image: Image, output: Image)`
Compute Inverse Fast Fourier Transform.

---

## Algorithms Module

### `medical_image.algorithms.algorithm`

#### `Algorithm` (Abstract Base Class)

Base class for image processing algorithms.

**Methods:**

##### `apply(image: Image, output: Image)` (Abstract)
Apply the algorithm to an image.

---

### `medical_image.algorithms.FEBDS`

#### `FebdsAlgorithm`

Frequency-Enhanced Band-pass Detection System for microcalcification detection.

**Inherits:** `Algorithm`

**Methods:**

##### `__init__(method: str)`
Initialize FEBDS algorithm.

**Parameters:**
- `method` (str): Detection method ('dog', 'log', or 'fft')

##### `apply(image: Image, output: Image)`
Apply FEBDS algorithm to detect microcalcifications.

**Parameters:**
- `image` (Image): Input mammogram
- `output` (Image): Output segmentation

**Algorithm Steps:**
1. Apply frequency enhancement (DoG/LoG/FFT+Butterworth)
2. Median filtering for noise reduction
3. Gamma correction
4. Thresholding (Otsu or variance-based)
5. Morphological closing
6. Region filling

---

## Utils Module

### `medical_image.utils.ErrorHandler`

#### `ErrorMessages`

Custom error message handlers for the library.

**Methods:**

##### `file_not_found(file_path: str)`
Raise error for missing file.

##### `unsupported_file_type(extension: str)`
Raise error for unsupported file format.

##### `invalid_pixel_data()`
Raise error for invalid pixel data.

##### `dicom_data_not_loaded()`
Raise error when DICOM data not loaded.

##### `empty_dataset()`
Raise error for empty dataset.

##### `annotation_type_not_recognized(annotation_type)`
Raise error for unknown annotation type.

##### `input_none(field_name: str)`
Raise error for None input in required field.

---

## Type Hints

The library uses Python type hints throughout for better IDE support and type checking.

**Common Types:**
- `Image`: Base image class
- `torch.Tensor`: PyTorch tensor for pixel data
- `np.ndarray`: NumPy array
- `Callable`: Function/transform
- `Optional[T]`: Optional type T
- `Union[A, B]`: Either type A or B
- `List[T]`: List of type T
- `Tuple[T, ...]`: Tuple of types

---

## GPU Acceleration

All image operations use PyTorch tensors, which automatically utilize GPU when available:

```python
# Check device
print(image.device)  # 'cuda' or 'cpu'

# Pixel data is automatically on the correct device
print(image.pixel_data.device)
```

---

## Error Handling

The library uses custom error handlers for clear error messages:

```python
from medical_image.utils.ErrorHandler import ErrorMessages

# Errors are raised with descriptive messages
try:
    image = DicomImage("invalid.txt")
except Exception as e:
    print(e)  # "Unsupported file type: .txt"
```
