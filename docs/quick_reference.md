# Quick Reference Guide

A quick reference for common operations in the Medical Image Standard Library.

---

## Installation

```bash
# Standard install (editable mode)
pip install -e .

# With development dependencies (pytest, black, ruff, mypy)
pip install -e ".[dev]"
```

---

## Loading Images

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.data.png_image import PNGImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.utils.image_utils import ImageExporter, ImageVisualizer
import numpy as np

# Load from file
image = DicomImage.from_file("path/to/image.dcm")
image.load()

# Create from a NumPy array or torch Tensor
image = InMemoryImage.from_array(np.random.rand(512, 512).astype(np.float32))

# Clone an image (lightweight copy of pixel_data and metadata)
copy = image.clone()

# Display info
image.display_info()

# Visualize
ImageVisualizer.show(image, cmap="gray", title="My Image")

# Save to disk (PNG, JPG, TIFF)
ImageExporter.save_as(image, format="PNG")
```

---

## Common Filters

All filter functions accept an optional `device=None` parameter. When `None`,
the device is inferred from the input image.

```python
from medical_image.process.filters import Filters

# Gaussian blur
Filters.gaussian_filter(input_img, output_img, sigma=2.0, device=None)

# Median filter (noise reduction)
Filters.median_filter(input_img, output_img, size=5, device=None)

# Difference of Gaussian (uses low_sigma / high_sigma)
Filters.difference_of_gaussian(input_img, output_img, low_sigma=1.7, high_sigma=2.0, device=None)

# Gamma correction
Filters.gamma_correction(input_img, output_img, gamma=1.25, device=None)

# Contrast adjustment (lowercase method name)
Filters.contrast_adjust(input_img, output_img, contrast=50, brightness=30, device=None)

# Batch Gaussian filter (operates on a (B, C, H, W) tensor directly)
import torch
batch = torch.randn(8, 1, 256, 256)
filtered_batch = Filters.gaussian_filter_batch(batch, sigma=1.5, device=None)
```

---

## Thresholding

```python
from medical_image.process.threshold import Threshold

# Otsu's method (automatic)
Threshold.otsu_threshold(input_img, output_img, device=None)

# Sauvola's method (local adaptive)
Threshold.sauvola_threshold(input_img, output_img, window_size=15, k=0.5, device=None)

# Variance-based binarization
Threshold.binarize(input_img, output_img, alpha=0.5, device=None)
```

---

## Metrics

```python
from medical_image.process.metrics import Metrics

# Entropy
entropy = Metrics.entropy(image, decimals=4)

# Mutual information
mi = Metrics.mutual_information(image1, image2)

# Local variance
Metrics.local_variance(input_img, output_img, kernel=5)

# Global variance
Metrics.variance(input_img, output_img)
```

---

## Region of Interest

```python
from medical_image.data.region_of_interest import RegionOfInterest

# Bounding box [x_min, y_min, x_max, y_max]
roi = RegionOfInterest(image, [100, 100, 400, 400])
cropped = roi.load()

# Polygon [(x1, y1), (x2, y2), ...]
roi = RegionOfInterest(image, [(100, 100), (200, 100), (200, 200)])
cropped = roi.load()

# Binary mask
import numpy as np
mask = np.zeros((512, 512), dtype=bool)
mask[100:400, 100:400] = True
roi = RegionOfInterest(image, mask)
cropped = roi.load()
```

---

## Patch Processing

```python
from medical_image.data.patch import PatchGrid

# Create patch grid
patch_grid = PatchGrid(image, patch_size=(256, 256))

# Process each patch
for patch in patch_grid.patches:
    data = patch.pixel_data
    position = patch.grid_id()

# Reconstruct image
reconstructed = patch_grid.reconstruct()
```

---

## Algorithms

All algorithms follow the same interface: instantiate with parameters, then
call `algo.apply(image, output)` (or use the callable shorthand `algo(image, output)`).

### KMeans

```python
from medical_image.algorithms.kmeans import KMeansAlgorithm

kmeans = KMeansAlgorithm(k=3, max_iter=100, tol=1e-4, device="cpu")
output = image.clone()
kmeans.apply(image, output)

# Inspect results
print(kmeans.centroids)    # (k, d) cluster centroids
print(kmeans.labels)       # (H, W) hard assignments
print(kmeans.stats)        # per-cluster statistics
```

### FCM (Fuzzy C-Means)

```python
from medical_image.algorithms.fcm import FCMAlgorithm

fcm = FCMAlgorithm(c=3, m=2.0, max_iter=100, tol=1e-3, device="cpu")
output = image.clone()
fcm.apply(image, output)

# Inspect results
print(fcm.membership)      # (c, N) fuzzy membership matrix
print(fcm.centroids)       # (c, d) cluster centroids
print(fcm.stats)           # per-cluster statistics
```

### PFCM (Possibilistic Fuzzy C-Means)

```python
from medical_image.algorithms.pfcm import PFCMAlgorithm

pfcm = PFCMAlgorithm(c=3, m=2.0, eta=2.0, tau=0.04, device="cpu")
output = image.clone()
pfcm.apply(image, output)

# Inspect results
print(pfcm.typicality)     # (c, N) typicality matrix
print(pfcm.T_max_map)      # (H, W) max typicality per pixel
```

### TopHat

```python
from medical_image.algorithms.top_hat import TopHatAlgorithm

tophat = TopHatAlgorithm(radius=4, device="cpu")
output = image.clone()
tophat.apply(image, output)
```

### FEBDS

```python
from medical_image.algorithms.FEBDS import FebdsAlgorithm

# Method: "dog" (fast), "log" (balanced), "fft" (high quality)
febds = FebdsAlgorithm(method="dog", device="cpu")
output = image.clone()
febds.apply(image, output)
```

---

## Datasets

```python
from medical_image.data.cbis_ddsm import CbisDdsm
from torch.utils.data import DataLoader

# Load dataset
dataset = CbisDdsm("/path/to/CBIS-DDSM", train=True)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate
for images, annotations in dataloader:
    pass
```

---

## Annotations

```python
from medical_image.utils.annotation import Annotation, GeometryType

# Calcification annotation
annotation = Annotation(
    annotation_type=GeometryType.BOUNDING_BOX,
    coordinates=[[100, 100, 200, 200]],
    classes=['calcification'],
    image_view='CC',
    abnormality_type='calcification',
    pathology='MALIGNANT',
    calcification_type='PLEOMORPHIC',
    calcification_distribution='CLUSTERED'
)

# Mass annotation
annotation = Annotation(
    annotation_type=GeometryType.POLYGON,
    coordinates=[[(100, 100), (200, 100), (200, 200)]],
    classes=['mass'],
    image_view='MLO',
    abnormality_type='mass',
    pathology='BENIGN',
    mass_shape='IRREGULAR',
    mass_margin='SPICULATED'
)
```

---

## Frequency Domain

```python
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.filters import Filters
from medical_image.data.in_memory_image import InMemoryImage
import torch

# FFT
freq_img = image.clone()
FrequencyOperations.fft(image, freq_img, device=None)

# Apply Butterworth filter
kernel_img = InMemoryImage(array=torch.zeros_like(freq_img.pixel_data))
Filters.butterworth_kernel(freq_img, kernel_img, D_0=21, W=32, n=3, device=None)
freq_img.pixel_data *= kernel_img.pixel_data

# Inverse FFT
output = image.clone()
FrequencyOperations.inverse_fft(freq_img, output, device=None)
```

---

## Morphology

```python
from medical_image.process.morphology import MorphologyOperations

# Morphological closing
MorphologyOperations.morphology_closing(input_img, output_img, kernel_size=5, device=None)

# Region filling
MorphologyOperations.region_fill(input_img, output_img, device=None)
```

---

## GPU Acceleration

### Device transfer

```python
# Move image data to GPU (auto-infers available device)
image.to("cuda")

# Move back to CPU
image.to("cpu")

# Check current device
print(image.device)
```

### DeviceContext -- managed GPU processing

```python
from medical_image.utils.device import DeviceContext

with DeviceContext(device="cuda", fallback="cpu", verbose=True) as ctx:
    image.to(ctx.device)
    output = image.clone()
    Filters.gaussian_filter(image, output, sigma=2.0, device=ctx.device)
    print(ctx.memory_stats())
# GPU cache is cleared automatically on exit
```

### Pin memory for faster transfers

```python
# Pin pixel_data to page-locked memory before GPU transfer
image.pin_memory()
image.to("cuda")
```

### Mixed precision

```python
from medical_image.utils.device import Precision, set_default_precision, get_dtype

set_default_precision(Precision.HALF)    # torch.float16
print(get_dtype())                       # torch.float16

# Options: Precision.FULL (float32), Precision.HALF (float16), Precision.BFLOAT16
```

### Batch processing across GPUs

```python
from medical_image.utils.device import AsyncGPUPipeline, MultiGPUAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm

# Async pipeline with overlapped I/O and compute
pipeline = AsyncGPUPipeline(device="cuda")
results = pipeline.process_images(image_list, algorithm)

# Data-parallel across multiple GPUs
multi = MultiGPUAlgorithm(KMeansAlgorithm, gpu_ids=[0, 1], k=3)
outputs = multi.apply_batch(image_list, output_list)
```

---

## Logging

```python
from medical_image.utils.logging import logger, configure_logging

# Enable console + file logging
configure_logging(level="DEBUG", log_file="processing.log")

logger.info("Processing image...")
logger.debug(f"Image shape: {image.pixel_data.shape}")
logger.error("Something went wrong")
```

---

## Error Handling

```python
from medical_image.data.dicom_image import DicomImage

try:
    image = DicomImage.from_file("image.dcm")
    image.load()
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Complete Pipeline Example

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.process.filters import Filters
from medical_image.process.threshold import Threshold
from medical_image.process.morphology import MorphologyOperations
from medical_image.utils.image_utils import ImageExporter

# 1. Load
image = DicomImage.from_file("mammogram.dcm")
image.load()

# 2. Preprocess
filtered = image.clone()
Filters.gaussian_filter(image, filtered, sigma=1.5)

# 3. Enhance
enhanced = filtered.clone()
Filters.gamma_correction(filtered, enhanced, gamma=1.2)

# 4. Segment
segmented = enhanced.clone()
Threshold.otsu_threshold(enhanced, segmented)

# 5. Post-process
final = segmented.clone()
MorphologyOperations.morphology_closing(segmented, final, kernel_size=3)

# 6. Save
ImageExporter.save_as(final, format="PNG")
```

---

## Common Parameters

### device=None

All filter, threshold, morphology, and frequency functions accept `device=None`.
When set to `None` (the default), the device is automatically inferred from the
input image's `pixel_data` tensor. Pass an explicit string like `"cuda"` or `"cpu"`
to override.

### Filter Sizes
- **Small:** 3x3 (minimal smoothing)
- **Medium:** 5x5 (standard)
- **Large:** 7x7+ (heavy smoothing)

### Sigma Values
- **Small:** 0.5-1.0 (slight blur)
- **Medium:** 1.5-3.0 (moderate blur)
- **Large:** >3.0 (heavy blur)

### Gamma Values
- **< 1:** Brighten image
- **= 1:** No change
- **> 1:** Darken image

---

## Useful Constants

```python
from medical_image.utils.annotation import GeometryType

# Annotation types
GeometryType.BOUNDING_BOX
GeometryType.POLYGON
GeometryType.MASK

# Pathology labels
'BENIGN'
'BENIGN_WITHOUT_CALLBACK'
'MALIGNANT'

# Image views
'CC'  # Craniocaudal
'MLO' # Mediolateral oblique

# Abnormality types
'calcification'
'mass'
```

---

## Testing

```bash
# Run all tests
pytest medical_image/tests/

# Run with coverage
pytest --cov=medical_image medical_image/tests/

# Run a specific test file
pytest medical_image/tests/test_dicom.py

# Verbose output
pytest -v medical_image/tests/
```

---

## Performance Tips

1. **Use GPU when available** -- `image.to("cuda")`
2. **Process in batches** -- use `gaussian_filter_batch` or `MultiGPUAlgorithm`
3. **Use patch-based processing for very large images**
4. **Pin memory before GPU transfer** -- `image.pin_memory()`
5. **Use `DeviceContext` for automatic cache management**
6. **Use lazy loading for large datasets**

---

## Documentation Links

- [Full Documentation](README.md)
- [API Reference](api_reference.md)
- [User Guide](user_guide.md)
- [Algorithm Reference](algorithms.md)
- [Dataset Guide](datasets.md)
- [Contributing](contributing.md)