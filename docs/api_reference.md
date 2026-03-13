# API Reference

## Table of Contents

1. [Data Module](#data-module)
2. [Process Module](#process-module)
3. [Algorithms Module](#algorithms-module)
4. [Utils Module](#utils-module)
5. [GPU Acceleration](#gpu-acceleration)

---

## Data Module

### `medical_image.data.image`

#### `Image` (Abstract Base Class)

Abstract base class for medical images supporting lazy loading and multiple constructors.

Width and height are computed properties derived from `pixel_data.shape` when pixel data is loaded. Before loading, they fall back to cached values set during construction or by subclass `.load()` methods.

**Attributes:**
- `file_path` (Optional[str]): Path to the image file (None for in-memory images)
- `pixel_data` (Optional[torch.Tensor]): Image pixel data as PyTorch tensor
- `_width` (Optional[int]): Backing field for width
- `_height` (Optional[int]): Backing field for height
- `_device` (torch.device): Backing field for device (default: `cpu`)

**Properties:**
- `width` (Optional[int]): Image width in pixels. Derived from `pixel_data.shape[-1]` when loaded, otherwise returns `_width`.
- `height` (Optional[int]): Image height in pixels. Derived from `pixel_data.shape[-2]` when loaded, otherwise returns `_height`.
- `device` (torch.device): Computation device. Derived from `pixel_data.device` when loaded, otherwise returns `_device`.

**Methods:**

##### `__init__(file_path=None, array=None, width=None, height=None, source_image=None)`
Initialize an Image object via one of four construction paths.

**Parameters:**
- `file_path` (Optional[str]): Path to an image file. Raises `FileNotFoundError` if it does not exist.
- `array` (Optional[Union[np.ndarray, torch.Tensor]]): Pre-existing array to wrap as pixel data.
- `width` (Optional[int]): Explicit width hint (used before pixel data is loaded).
- `height` (Optional[int]): Explicit height hint (used before pixel data is loaded).
- `source_image` (Optional[Image]): Another Image to copy metadata and pixel data from.

##### `load()` (Abstract)
Load pixel data from file. Must be implemented by subclasses.

##### `save()` (Abstract)
Save pixel data to file. Must be implemented by subclasses.

##### `clone() -> Image`
Lightweight clone: copies pixel_data tensor and metadata without copying heavy objects (DICOM data, PIL images).

**Returns:**
- `Image`: A new Image instance with cloned pixel data.

##### `to(device: Union[str, torch.device]) -> Image`
Move pixel_data to the target device (in-place). Returns `self` for chaining.

**Parameters:**
- `device` (Union[str, torch.device]): Target device.

**Returns:**
- `Image`: Self, for method chaining.

##### `pin_memory() -> Image`
Pin pixel_data to page-locked memory for faster GPU transfers. No-op if already pinned or pixel_data is None.

**Returns:**
- `Image`: Self, for method chaining.

##### `ensure_loaded() -> Image`
Guard method: raises `DicomDataNotLoadedError` if pixel_data is None.

**Returns:**
- `Image`: Self, for method chaining.

##### `display_info()`
Log basic information about the image (file path, dimensions, device, pixel data status, annotations).

##### `from_file(file_path: str) -> Image` (classmethod)
Construct an Image from a file path.

##### `from_array(array: Union[np.ndarray, torch.Tensor]) -> Image` (classmethod)
Construct an Image from a NumPy array or PyTorch tensor.

##### `from_image(other_image: Image) -> Image` (classmethod)
Construct an Image by copying another Image.

##### `empty(width=None, height=None) -> Image` (classmethod)
Construct an empty Image with optional width/height hints.

---

### `medical_image.data.in_memory_image`

#### `InMemoryImage`

Concrete Image subclass that lives only in memory (no file I/O).

**Inherits:** `Image`

Both `load()` and `save()` are no-ops. This class is useful for intermediate processing results and temporary images.

**Methods:**

##### `__init__(file_path=None, array=None, width=None, height=None, source_image=None)`
Same parameters as `Image.__init__()`.

##### `load()`
No-op.

##### `save()`
No-op.

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

#### `requires_loaded` (Decorator)

Decorator that checks `pixel_data` is not None on any `Image` argument before calling the wrapped function. Raises `DicomDataNotLoadedError` if an unloaded image is passed.

#### `Filters`

Static methods for image filtering operations. All methods accept `device=None`, which defaults to inferring the device from the input image via `resolve_device()`.

**Methods:**

##### `convolution(image: Image, output: Image, kernel: torch.Tensor, device=None) -> Image`
Apply convolution filter with custom kernel.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `kernel` (torch.Tensor): 2D convolution kernel
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `gaussian_filter(image: Image, output: Image, sigma: float, device=None, truncate: float = 4.0) -> Image`
Apply Gaussian blur filter.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `sigma` (float): Standard deviation of Gaussian kernel
- `device`: Device for computation (None = infer from image)
- `truncate` (float): Kernel truncation factor. Default: 4.0

**Returns:**
- `Image`: The output Image.

##### `gaussian_filter_batch(images: torch.Tensor, sigma: float, device=None, truncate: float = 4.0) -> torch.Tensor`
Apply Gaussian filter to a batch of images.

**Parameters:**
- `images` (torch.Tensor): Batched tensor of shape `(B, C, H, W)`
- `sigma` (float): Gaussian sigma
- `device`: Target device (None = infer from tensor)
- `truncate` (float): Kernel truncation factor. Default: 4.0

**Returns:**
- `torch.Tensor`: Filtered batch `(B, C, H, W)`

##### `median_filter(image: Image, output: Image, size: int, device=None) -> Image`
Apply median filter for noise reduction.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `size` (int): Filter window size (must be odd)
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `butterworth_kernel(image: Image, output: Image, D_0: float = 21, W: float = 32, n: int = 3, device=None) -> Image`
Apply a Butterworth band-pass filter in the frequency domain.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `D_0` (float): Cutoff frequency. Default: 21
- `W` (float): Bandwidth. Default: 32
- `n` (int): Filter order. Default: 3
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `difference_of_gaussian(image: Image, output: Image, low_sigma: float, high_sigma: float | None = None, device=None, truncate=4.0) -> Image`
Apply Difference of Gaussian (DoG) filter.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `low_sigma` (float): First Gaussian sigma
- `high_sigma` (float | None): Second Gaussian sigma. Default: `low_sigma * 1.6`
- `device`: Device for computation (None = infer from image)
- `truncate` (float): Kernel truncation factor. Default: 4.0

**Returns:**
- `Image`: The output Image.

##### `laplacian_of_gaussian(image: Image, output: Image, sigma: float, device=None) -> Image`
Apply Laplacian of Gaussian (LoG) filter.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `sigma` (float): Gaussian sigma
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `gamma_correction(image: Image, output: Image, gamma: float, device=None) -> Image`
Apply gamma correction for brightness adjustment.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `gamma` (float): Gamma value (> 1 brightens, < 1 darkens)
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `contrast_adjust(image: Image, output: Image, contrast: float, brightness: float, device=None) -> Image`
Adjust image contrast and brightness.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `contrast` (float): Contrast adjustment value
- `brightness` (float): Brightness adjustment value
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

---

### `medical_image.process.threshold`

#### `Threshold`

Static methods for image thresholding. All methods accept `device=None`, which defaults to inferring the device from the input image.

**Methods:**

##### `otsu_threshold(image: Image, output: Image = None, device=None) -> Image`
Apply Otsu's automatic thresholding method.

**Parameters:**
- `image` (Image): Input image
- `output` (Image, optional): Output image object. If None, a new `InMemoryImage` is created.
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

**Description:**
Automatically determines optimal threshold by maximizing between-class variance.

##### `sauvola_threshold(image: Image, output: Image = None, window_size: int = 10, k: float = 0.5, r: int = 128, device=None) -> Image`
Apply Sauvola's local adaptive thresholding.

**Parameters:**
- `image` (Image): Input image
- `output` (Image, optional): Output image object. If None, a new `InMemoryImage` is created.
- `window_size` (int): Local window size (must be odd). Default: 10
- `k` (float): Weighting factor. Default: 0.5
- `r` (int): Dynamic range. Default: 128
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `binarize(image: Image, output: Image, alpha: float, device=None) -> Image`
Binarize image using local and global variance.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `alpha` (float): Scaling factor relating local and global variance
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

---

### `medical_image.process.metrics`

#### `Metrics`

Static methods for image quality metrics. All methods accept `device=None`, which defaults to inferring the device from the input image.

**Methods:**

##### `entropy(image: Image, decimals: int = 4, device=None) -> float`
Calculate Shannon entropy of image.

**Parameters:**
- `image` (Image): Input image
- `decimals` (int): Decimal places for rounding. Default: 4
- `device`: Device for computation (None = infer from image)

**Returns:**
- `float`: Entropy value (in bits)

##### `joint_entropy(image1: Image, image2: Image, decimals: int = 4, device=None) -> float`
Calculate joint entropy of two images.

**Parameters:**
- `image1` (Image): First image
- `image2` (Image): Second image
- `decimals` (int): Decimal places. Default: 4
- `device`: Device for computation (None = infer from image)

**Returns:**
- `float`: Joint entropy value

##### `mutual_information(image1: Image, image2: Image, decimals: int = 4, device=None) -> float`
Calculate mutual information between two images.

**Parameters:**
- `image1` (Image): First image
- `image2` (Image): Second image
- `decimals` (int): Decimal places. Default: 4
- `device`: Device for computation (None = infer from image)

**Returns:**
- `float`: Mutual information value

##### `local_variance(image: Image, output: Image, kernel: Union[int, tuple], device=None) -> Image`
Calculate local variance in specified sub-regions.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `kernel` (Union[int, tuple]): Kernel size for local window
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `variance(image: Image, output: Image, device=None) -> Image`
Calculate global variance of image.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image object
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

---

### `medical_image.process.morphology`

#### `MorphologyOperations`

Static methods for morphological operations. All methods accept `device=None`, which defaults to inferring the device from the input image.

**Methods:**

##### `morphology_closing(image: Image, output: Image, kernel_size: int = 7, device=None) -> Image`
Apply morphological closing operation (dilation followed by erosion).

**Parameters:**
- `image` (Image): Input binary image (0/1)
- `output` (Image): Output image object
- `kernel_size` (int): Size of the square structuring element. Default: 7
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `region_fill(image: Image, output: Image, device=None) -> Image`
Fill holes in a binary image using iterative dilation.

**Parameters:**
- `image` (Image): Input binary image (0/1)
- `output` (Image): Output image object
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `erosion(image: Image, output: Image, radius: int = 4, device=None) -> Image`
Grayscale erosion using a flat disk structuring element.

**Parameters:**
- `image` (Image): Input image (2D float)
- `output` (Image): Output image object
- `radius` (int): Disk SE radius. Default: 4
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `dilation(image: Image, output: Image, radius: int = 4, device=None) -> Image`
Grayscale dilation using a flat disk structuring element.

**Parameters:**
- `image` (Image): Input image (2D float)
- `output` (Image): Output image object
- `radius` (int): Disk SE radius. Default: 4
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `white_top_hat(image: Image, output: Image, radius: int = 4, device=None) -> Image`
White Top-Hat transform: `TopHat(I) = I - opening(I)`. Highlights bright structures smaller than the structuring element (e.g., microcalcifications).

**Parameters:**
- `image` (Image): Input image (2D float)
- `output` (Image): Output image object
- `radius` (int): Disk SE radius. Default: 4
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

---

### `medical_image.process.frequency`

#### `FrequencyOperations`

Static methods for frequency domain operations. All methods accept `device=None`, which defaults to inferring the device from the input image.

**Methods:**

##### `fft(image: Image, output: Image, device=None) -> Image`
Compute 2D Fast Fourier Transform.

**Parameters:**
- `image` (Image): Input image
- `output` (Image): Output image (stores complex FFT result)
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

##### `inverse_fft(image: Image, output: Image, device=None) -> Image`
Compute Inverse 2D Fast Fourier Transform.

**Parameters:**
- `image` (Image): Input image in the frequency domain (complex tensor)
- `output` (Image): Output image (stores the inverse FFT result)
- `device`: Device for computation (None = infer from image)

**Returns:**
- `Image`: The output Image.

---

## Algorithms Module

### `medical_image.algorithms.algorithm`

#### `Algorithm` (Abstract Base Class)

Base class for image processing algorithms.

**Constructor:**

##### `__init__(device=None, precision: Precision = Precision.FULL)`
Initialize the algorithm.

**Parameters:**
- `device`: Device string (e.g., `"cpu"`, `"cuda"`). If None, auto-selects CUDA when available, otherwise CPU.
- `precision` (Precision): Mixed-precision mode. Default: `Precision.FULL` (float32).

**Attributes:**
- `device` (str): The resolved device string.
- `precision` (Precision): The precision enum value.

**Methods:**

##### `apply(image: Image, output: Image) -> Image` (Abstract)
Apply the algorithm to an image. Must be implemented by subclasses.

##### `__call__(image: Image, output: Image) -> Image`
Callable interface. Wraps `apply()` with automatic mixed-precision (`torch.cuda.amp.autocast`) when a non-FULL precision is set on a CUDA device.

##### `apply_batch(images: List[Image], outputs: List[Image]) -> List[Image]`
Process a batch of images. Default implementation loops over `apply()`. Subclasses can override for truly batched GPU processing.

##### `__repr__() -> str`
String representation showing class name, device, and precision.

---

### `medical_image.algorithms.FEBDS`

#### `FebdsAlgorithm`

Frequency-Enhanced Band-pass Detection System for microcalcification detection.

**Inherits:** `Algorithm`

**Methods:**

##### `__init__(method: str, device=None)`
Initialize FEBDS algorithm.

**Parameters:**
- `method` (str): Detection method (`'dog'`, `'log'`, or `'fft'`)
- `device`: Torch device. Default: None (auto-select).

##### `apply(image: Image, output: Image) -> Image`
Apply FEBDS algorithm to detect microcalcifications.

**Parameters:**
- `image` (Image): Input mammogram
- `output` (Image): Output segmentation

**Algorithm Steps:**
1. Apply frequency enhancement (DoG/LoG/FFT+Butterworth)
2. Absolute value and median filtering for noise reduction
3. Gamma correction
4. Thresholding (Otsu for dog/log, binarize for fft)
5. Morphological closing
6. Region filling

---

### `medical_image.algorithms.kmeans`

#### `KMeansAlgorithm`

K-Means hard clustering segmentation for microcalcification detection.

**Inherits:** `Algorithm`

**Methods:**

##### `__init__(k: int = 2, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42, device=None)`
Initialize K-Means algorithm.

**Parameters:**
- `k` (int): Number of clusters. Default: 2
- `max_iter` (int): Maximum iterations. Default: 100
- `tol` (float): Convergence tolerance. Default: 1e-4
- `random_state` (int): Random seed for reproducibility. Default: 42
- `device`: Torch device. Default: None (auto-select).

**Attributes after `apply()`:**
- `centroids` (torch.Tensor): `(k, d)` cluster centroids
- `labels` (torch.Tensor): `(H, W)` hard cluster assignments
- `quantized` (torch.Tensor): `(H, W)` quantized image
- `stats` (List[dict]): Per-cluster statistics
- `mc_label` (int): Index of the brightest (MC) cluster
- `n_iter` (int): Number of iterations run
- `converged` (bool): Whether convergence was reached

##### `apply(image: Image, output: Image) -> Image`
Segment image using K-Means. Output pixel_data is a binary MC mask isolating the brightest cluster.

---

### `medical_image.algorithms.fcm`

#### `FCMAlgorithm`

Fuzzy C-Means soft clustering segmentation for microcalcification detection.

**Inherits:** `Algorithm`

**Methods:**

##### `__init__(c: int = 2, m: float = 2.0, max_iter: int = 100, tol: float = 1e-3, random_state: int = 42, device=None)`
Initialize FCM algorithm.

**Parameters:**
- `c` (int): Number of clusters. Default: 2
- `m` (float): Fuzziness exponent. Default: 2.0
- `max_iter` (int): Max iterations. Default: 100
- `tol` (float): Error tolerance. Default: 1e-3
- `random_state` (int): Random seed for reproducibility. Default: 42
- `device`: Torch device. Default: None (auto-select).

**Attributes after `apply()`:**
- `centroids` (torch.Tensor): `(c, d)` cluster centroids
- `membership` (torch.Tensor): `(c, N)` fuzzy membership matrix U
- `labels` (torch.Tensor): `(H, W)` hard cluster assignments
- `quantized` (torch.Tensor): `(H, W)` quantized image
- `stats` (List[dict]): Per-cluster statistics
- `mc_label` (int): Index of the brightest (MC) cluster

##### `apply(image: Image, output: Image) -> Image`
Produce binary MC mask from largest fuzzy memberships.

---

### `medical_image.algorithms.pfcm`

#### `PFCMAlgorithm`

Possibilistic Fuzzy C-Means robust segmentation for microcalcification detection.

**Inherits:** `Algorithm`

**Methods:**

##### `__init__(c: int = 2, m: float = 2.0, eta: float = 2.0, a: float = 1.0, b: float = 4.0, tau: float = 0.04, max_iter: int = 100, tol: float = 1e-3, fcm_max_iter: int = 100, random_state: int = 42, device=None)`
Initialize PFCM algorithm.

**Parameters:**
- `c` (int): Number of clusters. Default: 2
- `m` (float): Fuzziness exponent. Default: 2.0
- `eta` (float): Typicality fuzziness. Default: 2.0
- `a` (float): Weighting coefficient for membership. Default: 1.0
- `b` (float): Weighting coefficient for typicality. Default: 4.0
- `tau` (float): Typicality threshold for MC detection. Default: 0.04
- `max_iter` (int): Maximum PFCM iterations. Default: 100
- `tol` (float): Convergence tolerance. Default: 1e-3
- `fcm_max_iter` (int): Maximum FCM warm-start iterations. Default: 100
- `random_state` (int): Random seed. Default: 42
- `device`: Torch device. Default: None (auto-select).

**Attributes after `apply()`:**
- `typicality` (torch.Tensor): `(c, N)` typicality matrix T
- `T_max_map` (torch.Tensor): `(H, W)` max typicality per pixel
- `centroids` (torch.Tensor): `(c, d)` cluster centroids
- `membership` (torch.Tensor): `(c, N)` fuzzy membership matrix
- `labels` (torch.Tensor): `(H, W)` hard cluster assignments
- `quantized` (torch.Tensor): `(H, W)` quantized image
- `gamma` (torch.Tensor): `(c,)` gamma values per cluster

##### `apply(image: Image, output: Image) -> Image`
Apply PFCM: warm-start from FCM, iterate PFCM, detect MCs by atypicality. Output pixel_data is a binary mask of atypical (non-background) pixels.

---

### `medical_image.algorithms.top_hat`

#### `TopHatAlgorithm`

White Top-Hat transform for extracting bright sub-regions (microcalcifications).

**Inherits:** `Algorithm`

**Methods:**

##### `__init__(radius: int = 4, device=None)`
Initialize Top-Hat algorithm.

**Parameters:**
- `radius` (int): Disk structuring element radius. Default: 4 (produces a 9x9 footprint)
- `device`: Torch device. Default: None (auto-select).

##### `apply(image: Image, output: Image) -> Image`
Apply white top-hat filtering to the input image.

---

## Utils Module

### `medical_image.utils.device`

GPU device management, memory handling, mixed precision, and multi-GPU utilities.

#### `resolve_device(*images, explicit=None) -> torch.device`
Determine the target device for a processing operation.

**Priority:**
1. Explicit device parameter (if provided)
2. Device of the first loaded image's `pixel_data`
3. Fallback to CPU

**Parameters:**
- `*images`: Zero or more Image objects to inspect.
- `explicit` (Union[str, torch.device, None]): Explicit device override.

**Returns:**
- `torch.device`: The resolved device.

---

#### `Precision` (Enum)

Mixed-precision settings for algorithms.

**Values:**
- `FULL`: `torch.float32`
- `HALF`: `torch.float16`
- `BFLOAT16`: `torch.bfloat16`

---

#### `set_default_precision(precision: Precision) -> None`
Set the global default precision.

#### `get_default_precision() -> Precision`
Get the current global default precision.

#### `get_dtype() -> torch.dtype`
Get the `torch.dtype` corresponding to the current default precision.

---

#### `DeviceContext`

Context manager for GPU-aware processing with automatic memory management.

**Features:**
- Clears GPU cache on entry and exit
- Provides memory usage tracking
- Automatic CPU fallback when CUDA is unavailable
- Suppresses `torch.cuda.OutOfMemoryError` and falls back to CPU

**Methods:**

##### `__init__(device: str = "cuda", fallback: str = "cpu", verbose: bool = False)`

**Parameters:**
- `device` (str): Primary device. Default: `"cuda"`
- `fallback` (str): Fallback device. Default: `"cpu"`
- `verbose` (bool): Log GPU memory info on entry. Default: False

##### `device` (property) -> `torch.device`
The currently active device.

##### `memory_stats() -> dict`
Return current GPU memory usage as a dictionary with keys: `device`, `allocated_gb`, `free_gb`, `total_gb`. Returns `{"device": "cpu"}` when on CPU.

**Usage:**
```python
from medical_image.utils.device import DeviceContext

with DeviceContext("cuda", verbose=True) as ctx:
    image.to(ctx.device)
    # ... processing ...
    print(ctx.memory_stats())
```

---

#### `gpu_safe` (Decorator)

Catches `torch.cuda.OutOfMemoryError` and retries the decorated function on CPU.

**Usage:**
```python
from medical_image.utils.device import gpu_safe

@gpu_safe
def my_processing(image, output, device=None):
    # processing that might OOM on GPU
    ...
```

---

#### `AsyncGPUPipeline`

Overlap disk I/O, CPU-to-GPU transfer, and GPU compute using CUDA streams. Requires CUDA.

**Methods:**

##### `__init__(device: str = "cuda")`

**Parameters:**
- `device` (str): CUDA device string.

**Raises:**
- `RuntimeError`: If CUDA is not available.

##### `process_images(images: list, algorithm) -> list`
Process pre-loaded Image objects with overlapped transfer and compute.

**Parameters:**
- `images` (list): List of Image objects (already loaded).
- `algorithm`: An Algorithm instance.

**Returns:**
- `list`: List of output Image objects.

---

#### `MultiGPUAlgorithm`

Distribute algorithm execution across available GPUs (data-parallel, round-robin).

**Methods:**

##### `__init__(algorithm_cls: type, gpu_ids: Optional[List[int]] = None, **kwargs)`

**Parameters:**
- `algorithm_cls` (type): Algorithm class to instantiate per GPU.
- `gpu_ids` (Optional[List[int]]): GPU IDs to use. Default: all available.
- `**kwargs`: Additional keyword arguments forwarded to the algorithm constructor.

**Raises:**
- `RuntimeError`: If CUDA is not available.

##### `apply_batch(images: list, outputs: list) -> list`
Distribute images across GPUs round-robin, applying the algorithm to each.

---

### `medical_image.utils.logging`

Library-wide logging configuration.

#### `logger`

A `logging.Logger` instance named `"medical_image"` with a `NullHandler` attached by default (no output unless the user configures logging).

#### `configure_logging(level=logging.DEBUG, log_file=None)`
Convenience function to enable console and optional file logging.

**Parameters:**
- `level`: Logging level (default: `logging.DEBUG`).
- `log_file` (str, optional): Path to a log file. If None, only console output.

**Usage:**
```python
from medical_image.utils.logging import configure_logging

configure_logging(level=logging.INFO, log_file="run.log")
```

---

### `medical_image.utils.image_utils`

Image utility classes for conversion, export, visualization, and mathematical operations.

#### `TensorConverter`

**Methods:**

##### `to_numpy(image: Image) -> np.ndarray` (static)
Convert `Image.pixel_data` (torch tensor) to a NumPy array on CPU.

**Parameters:**
- `image` (Image): Image instance containing pixel_data.

**Returns:**
- `np.ndarray`: The pixel data as a NumPy array.

**Raises:**
- `ValueError`: If pixel_data is None or not a tensor.

##### `ensure_tensor(image: Image, device=None, dtype=None) -> torch.Tensor` (static)
Move `Image.pixel_data` to the specified device and dtype. Updates the image in-place and returns the tensor.

**Parameters:**
- `image` (Image): Image instance.
- `device`: Target device.
- `dtype`: Target dtype.

**Returns:**
- `torch.Tensor`: The updated tensor.

---

#### `ImageExporter`

Export an Image object to PNG/JPG/TIFF.

**Methods:**

##### `save_as(image: Image, format="PNG") -> str` (static)
Save the image to disk in the given format.

**Parameters:**
- `image` (Image): Image to export.
- `format` (str): Output format (e.g., `"PNG"`, `"JPEG"`, `"TIFF"`). Default: `"PNG"`.

**Returns:**
- `str`: Path to the saved file.

---

#### `ImageVisualizer`

Visualization utilities for Image objects.

**Methods:**

##### `show(image: Image, cmap="gray", title=None)` (static)
Display the image using matplotlib.

**Parameters:**
- `image` (Image): Image to display.
- `cmap` (str): Colormap. Default: `"gray"`.
- `title` (str, optional): Plot title.

##### `compare(before: Image, after: Image, title_before="Before", title_after="After")` (static)
Show two images side by side for comparison.

**Parameters:**
- `before` (Image): First image.
- `after` (Image): Second image.
- `title_before` (str): Title for first image. Default: `"Before"`.
- `title_after` (str): Title for second image. Default: `"After"`.

---

#### `MathematicalOperations`

**Methods:**

##### `abs(image: Image, out: Image) -> Image` (static)
Compute element-wise absolute value of pixel data.

##### `euclidean_distance_sq(Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor` (static)
Compute squared Euclidean distances between N data points and c centroids.

**Parameters:**
- `Z` (torch.Tensor): `(N, d)` data matrix.
- `V` (torch.Tensor): `(c, d)` centroid matrix.

**Returns:**
- `torch.Tensor`: `(c, N)` squared distances.

##### `normalize_12bit(image: Image, out: Image) -> Image` (static)
Normalize a 12-bit DICOM image to [0, 1] by dividing by 4095.

---

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

All image operations use PyTorch tensors. Device inference is handled automatically by `resolve_device()`, which inspects the input image's tensor device when no explicit device is provided.

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.utils.device import DeviceContext, resolve_device

# Load and move to GPU
image = DicomImage("scan.dcm")
image.load()
image.to("cuda")

# Device is inferred from image tensor
print(image.device)  # device(type='cuda', index=0)

# Processing functions infer device automatically
from medical_image.process.filters import Filters
from medical_image.data.in_memory_image import InMemoryImage

output = InMemoryImage(source_image=image)
Filters.gaussian_filter(image, output, sigma=1.5)  # runs on CUDA

# Or override explicitly
Filters.gaussian_filter(image, output, sigma=1.5, device="cpu")
```

Using `DeviceContext` for safe GPU processing with OOM fallback:

```python
from medical_image.utils.device import DeviceContext

with DeviceContext("cuda", verbose=True) as ctx:
    image.to(ctx.device)
    output = image.clone()
    Filters.gaussian_filter(image, output, sigma=2.0, device=ctx.device)
    # If OOM occurs, ctx automatically falls back to CPU
```

Using `AsyncGPUPipeline` for overlapped processing:

```python
from medical_image.utils.device import AsyncGPUPipeline
from medical_image.algorithms.kmeans import KMeansAlgorithm

pipeline = AsyncGPUPipeline("cuda")
algo = KMeansAlgorithm(k=3, device="cuda")
results = pipeline.process_images(loaded_images, algo)
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