# Quick Reference Guide

A quick reference for common operations in the Medical Image Standard Library.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Loading Images

```python
from medical_image.data.dicom_image import DicomImage

# Load DICOM
image = DicomImage("path/to/image.dcm")
image.load()

# Display info
image.display_info()

# Visualize
image.plot()

# Save as PNG
image.to_png()
```

---

## Common Filters

```python
from medical_image.process.filters import Filters

# Gaussian blur
Filters.gaussian_filter(input_img, output_img, sigma=2.0)

# Median filter (noise reduction)
Filters.median_filter(input_img, output_img, size=5)

# Difference of Gaussian
Filters.difference_of_gaussian(input_img, output_img, sigma_1=2.0, sigma_2=1.7)

# Gamma correction
Filters.gamma_correction(input_img, output_img, gamma=1.25)

# Contrast adjustment
Filters.ContrastAdjust(input_img, output_img, contrast=50, brightness=30)
```

---

## Thresholding

```python
from medical_image.process.threshold import Threshold

# Otsu's method (automatic)
Threshold.otsu_threshold(input_img, output_img)

# Sauvola's method (local adaptive)
Threshold.sauvola_threshold(input_img, output_img, window_size=15, k=0.5)

# Variance-based binarization
Threshold.binarize(input_img, output_img, alpha=0.5)
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
    # Access patch data
    data = patch.pixel_data
    position = patch.grid_id()
    
# Reconstruct image
reconstructed = patch_grid.reconstruct()
```

---

## FEBDS Algorithm

```python
from medical_image.algorithms.FEBDS import FebdsAlgorithm

# Method 1: DoG (fast)
febds = FebdsAlgorithm(method="dog")
febds.apply(input_img, output_img)

# Method 2: LoG (balanced)
febds = FebdsAlgorithm(method="log")
febds.apply(input_img, output_img)

# Method 3: FFT (high quality)
febds = FebdsAlgorithm(method="fft")
febds.apply(input_img, output_img)
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
    # Training loop
    pass
```

---

## Annotations

```python
from medical_image.utils.annotation import Annotation, AnnotationType

# Calcification annotation
annotation = Annotation(
    annotation_type=AnnotationType.BOUNDING_BOX,
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
    annotation_type=AnnotationType.POLYGON,
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
import numpy as np

# FFT
freq_img = DicomImage("freq.dcm")
FrequencyOperations.fft(input_img, freq_img)

# Apply Butterworth filter
kernel = Filters.butterworth_kernel(freq_img, D_0=21, W=32, n=3)
freq_img.pixel_data = np.multiply(freq_img.pixel_data, kernel)

# Inverse FFT
FrequencyOperations.inverse_fft(freq_img, output_img)
```

---

## Morphology

```python
from medical_image.process.morphology import MorphologyOperations

# Morphological closing
MorphologyOperations.morphology_closing(input_img, output_img, kernel_size=5)

# Region filling
MorphologyOperations.region_fill(input_img, output_img)
```

---

## GPU Acceleration

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Images automatically use GPU when available
print(f"Device: {image.device}")

# Manual device transfer
image.pixel_data = image.pixel_data.to('cuda')
image.pixel_data = image.pixel_data.to('cpu')

# Clear GPU cache
torch.cuda.empty_cache()
```

---

## Error Handling

```python
from medical_image.utils.ErrorHandler import ErrorMessages

try:
    image = DicomImage("image.dcm")
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

# 1. Load
image = DicomImage("mammogram.dcm")
image.load()

# 2. Preprocess
filtered = DicomImage("filtered.dcm")
Filters.gaussian_filter(image, filtered, sigma=1.5)

# 3. Enhance
enhanced = DicomImage("enhanced.dcm")
Filters.gamma_correction(filtered, enhanced, gamma=1.2)

# 4. Segment
segmented = DicomImage("segmented.dcm")
Threshold.otsu_threshold(enhanced, segmented)

# 5. Post-process
final = DicomImage("final.dcm")
MorphologyOperations.morphology_closing(segmented, final, kernel_size=3)

# 6. Save
final.to_png()
```

---

## Common Parameters

### Filter Sizes
- **Small:** 3×3 (minimal smoothing)
- **Medium:** 5×5 (standard)
- **Large:** 7×7+ (heavy smoothing)

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
# Annotation types
AnnotationType.BOUNDING_BOX
AnnotationType.POLYGON
AnnotationType.MASK

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

## Performance Tips

1. **Use GPU when available**
2. **Process in batches for large datasets**
3. **Use patch-based processing for very large images**
4. **Clear GPU cache regularly**
5. **Use lazy loading for large datasets**

---

## Debugging

```python
from log_manager import logger

# Enable logging
logger.info("Processing image...")
logger.debug(f"Image shape: {image.pixel_data.shape}")
logger.error("Error occurred!")

# Check tensor properties
print(f"Shape: {tensor.shape}")
print(f"Device: {tensor.device}")
print(f"Dtype: {tensor.dtype}")
print(f"Min/Max: {tensor.min()}/{tensor.max()}")
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=medical_image

# Run specific test
pytest medical_image/tests/test_filters.py

# Verbose output
pytest -v
```

---

## Documentation Links

- [Full Documentation](README.md)
- [API Reference](api_reference.md)
- [User Guide](user_guide.md)
- [Algorithm Reference](algorithms.md)
- [Dataset Guide](datasets.md)
- [Contributing](contributing.md)
