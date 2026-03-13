# User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Working with Images](#working-with-images)
3. [Image Processing](#image-processing)
4. [Working with Datasets](#working-with-datasets)
5. [Advanced Topics](#advanced-topics)
6. [Best Practices](#best-practices)

---

## Getting Started

### Installation

```bash
# Install from source
git clone https://github.com/LATIS-DocumentAI-Group/medical-image-std.git
cd medical-image-std
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```python
import medical_image
from medical_image.data.dicom_image import DicomImage
print("Installation successful!")
```

---

## Working with Images

### Loading DICOM Images

```python
from medical_image.data.dicom_image import DicomImage

# Load a DICOM file
image = DicomImage("/path/to/mammogram.dcm")
image.load()

# Access image properties
print(f"Dimensions: {image.width} x {image.height}")
print(f"Device: {image.device}")
print(f"Pixel data shape: {image.pixel_data.shape}")
```

### Displaying Images

```python
from medical_image.utils.image_utils import ImageVisualizer

# Display image information
image.display_info()

# Visualize the image
ImageVisualizer.show(image, cmap='gray')

# Use custom colormap and title
ImageVisualizer.show(image, cmap='hot', title='Mammogram')
```

### Saving Images

```python
from medical_image.utils.image_utils import ImageExporter

# Save as PNG
ImageExporter.save_as(image)  # Saves as '<original_name>.png'

# Save as JPEG
ImageExporter.save_as(image, format="JPEG")

# Save modified DICOM
image.save()
```

### Converting to NumPy

```python
from medical_image.utils.image_utils import TensorConverter
import numpy as np

# Convert to NumPy array
pixel_array = TensorConverter.to_numpy(image)
print(type(pixel_array))  # <class 'numpy.ndarray'>

# Perform NumPy operations
mean_intensity = np.mean(pixel_array)
std_intensity = np.std(pixel_array)
```

### Cloning Images

```python
# Create a lightweight copy for use as an output target
output = image.clone()

# The clone has its own pixel_data tensor, independent of the original
```

---

## Image Processing

### Applying Filters

#### Gaussian Filter

```python
from medical_image.process.filters import Filters
from medical_image.data.dicom_image import DicomImage
from medical_image.utils.image_utils import ImageExporter

# Load image
image = DicomImage("input.dcm")
image.load()

# Create output via clone
output = image.clone()

# Apply Gaussian filter
Filters.gaussian_filter(image, output, sigma=2.0)

# Save result
ImageExporter.save_as(output)
```

#### Median Filter (Noise Reduction)

```python
output = image.clone()

# Apply median filter to remove salt-and-pepper noise
Filters.median_filter(image, output, size=5)
```

#### Difference of Gaussian (Edge Detection)

```python
output = image.clone()

# DoG filter for edge enhancement
Filters.difference_of_gaussian(
    image,
    output,
    low_sigma=2.0,
    high_sigma=3.2,
)
```

#### Laplacian of Gaussian

```python
output = image.clone()

# LoG filter for blob detection
Filters.laplacian_of_gaussian(image, output, sigma=2.0)
```

### Brightness and Contrast Adjustment

```python
output = image.clone()

# Gamma correction
Filters.gamma_correction(image, output, gamma=1.25)

# Contrast and brightness adjustment
Filters.contrast_adjust(
    image,
    output,
    contrast=50,
    brightness=30,
)
```

### Batch Filtering

Batch variants operate on raw `(B, C, H, W)` tensors for efficient GPU processing:

```python
import torch
from medical_image.process.filters import Filters

# Stack several images into a batch tensor
batch = torch.stack([img1.pixel_data.unsqueeze(0),
                     img2.pixel_data.unsqueeze(0)])  # (B, 1, H, W)

# Apply Gaussian filter to the whole batch at once
filtered_batch = Filters.gaussian_filter_batch(batch, sigma=2.0, device="cuda")
```

### Thresholding

#### Otsu's Method (Automatic)

```python
from medical_image.process.threshold import Threshold
from medical_image.utils.image_utils import ImageVisualizer

output = image.clone()

# Automatic thresholding
Threshold.otsu_threshold(image, output)
ImageVisualizer.show(output, cmap='binary')
```

#### Sauvola's Method (Local Adaptive)

```python
output = image.clone()

# Local adaptive thresholding
Threshold.sauvola_threshold(
    image,
    output,
    window_size=15,
    k=0.5,
    r=128,
)
```

#### Variance-based Binarization

```python
output = image.clone()

# Binarize using local/global variance
Threshold.binarize(image, output, alpha=0.5)
```

### Frequency Domain Processing

```python
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.filters import Filters
import torch

# Transform to frequency domain
freq_image = image.clone()
FrequencyOperations.fft(image, freq_image)

# Apply Butterworth band-pass filter
kernel_image = image.clone()
Filters.butterworth_kernel(freq_image, kernel_image, D_0=21, W=32, n=3)
freq_image.pixel_data = torch.multiply(freq_image.pixel_data, kernel_image.pixel_data)

# Transform back to spatial domain
output = image.clone()
FrequencyOperations.inverse_fft(freq_image, output)
```

### Morphological Operations

```python
from medical_image.process.morphology import MorphologyOperations

output = image.clone()

# Morphological closing (fill small holes)
MorphologyOperations.morphology_closing(image, output, kernel_size=5)

# Region filling
MorphologyOperations.region_fill(image, output)
```

### Computing Metrics

```python
from medical_image.process.metrics import Metrics
from medical_image.data.dicom_image import DicomImage

# Calculate entropy
ent = Metrics.entropy(image, decimals=4)
print(f"Image entropy: {ent} bits")

# Calculate mutual information between two images
image1 = DicomImage("image1.dcm")
image2 = DicomImage("image2.dcm")
image1.load()
image2.load()

mi = Metrics.mutual_information(image1, image2)
print(f"Mutual information: {mi}")

# Calculate local variance
variance_image = image1.clone()
Metrics.local_variance(image1, variance_image, kernel=5)
```

---

## Working with Regions of Interest

### Bounding Box ROI

```python
from medical_image.data.region_of_interest import RegionOfInterest

# Define bounding box [x_min, y_min, x_max, y_max]
bbox = [100, 100, 400, 400]

# Create ROI
roi = RegionOfInterest(image, bbox)

# Extract ROI
cropped = roi.load()
```

### Polygon ROI

```python
# Define polygon as list of (x, y) coordinates
polygon = [
    (100, 100),
    (200, 100),
    (200, 200),
    (150, 250),
    (100, 200)
]

# Create and extract polygon ROI
roi = RegionOfInterest(image, polygon)
cropped = roi.load()
```

### Mask ROI

```python
import numpy as np

# Create binary mask
mask = np.zeros((image.height, image.width), dtype=bool)
mask[100:400, 100:400] = True

# Create and extract mask ROI
roi = RegionOfInterest(image, mask)
cropped = roi.load()
```

---

## Working with Patches

### Creating a Patch Grid

```python
from medical_image.data.patch import PatchGrid

# Divide image into 256x256 patches
patch_grid = PatchGrid(image, patch_size=(256, 256))

print(f"Number of patches: {len(patch_grid.patches)}")
print(f"Grid dimensions: {len(patch_grid.grid)} x {len(patch_grid.grid[0])}")
```

### Processing Individual Patches

```python
# Iterate through all patches
for patch in patch_grid.patches:
    print(f"Patch at grid position {patch.grid_id()}")
    print(f"Pixel position: {patch.pixel_position()}")
    print(f"Size: {patch.width}x{patch.height}")
    print(f"Is padded: {patch.is_padded}")

    # Process patch
    # patch.pixel_data contains the patch tensor
```

### Accessing Patches by Grid Position

```python
# Access specific patch in grid
patch = patch_grid.grid[0][0]  # Top-left patch
patch = patch_grid.grid[-1][-1]  # Bottom-right patch
```

### Reconstructing from Patches

```python
# Modify patches (example: apply filter to each)
for patch in patch_grid.patches:
    # Apply some processing to patch.pixel_data
    pass

# Reconstruct full image
reconstructed = patch_grid.reconstruct()

# Store into an output image
output = image.clone()
output.pixel_data = reconstructed
```

---

## Working with Datasets

### Creating a Custom Dataset

```python
from medical_image.data.medical_dataset import MedicalDataset
from medical_image.data.dicom_image import DicomImage

class MyMedicalDataset(MedicalDataset):
    def __init__(self, base_path, transform=None, train=True):
        super().__init__(base_path, file_format='.dcm', transform=transform, train=train)

        # Load image paths
        import os
        self.images_path = [
            os.path.join(base_path, f)
            for f in os.listdir(base_path)
            if f.endswith('.dcm')
        ]

    def load_batch(self, batch_size):
        # Implement batch loading logic
        pass

    def destroy_batch(self):
        # Free memory
        self.current_image = None

    def apply_transform(self, transform, pixel_data, label):
        # Apply transformations
        if transform:
            pixel_data = transform(pixel_data)
        return pixel_data, label
```

### Using with PyTorch DataLoader

```python
from torch.utils.data import DataLoader

# Create dataset
dataset = MyMedicalDataset("/path/to/dicom/files", train=True)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# Iterate through batches
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: {images.shape}")
    # Train your model here
```

### CBIS-DDSM Dataset

```python
from medical_image.data.cbis_ddsm import CbisDdsm

# Load CBIS-DDSM dataset
dataset = CbisDdsm(
    base_path="/path/to/CBIS-DDSM",
    train=True
)

# Access samples
image, annotation = dataset[0]
print(f"Abnormality type: {annotation.abnormality_type}")
print(f"Pathology: {annotation.pathology}")
```

---

## Using Algorithms

### FEBDS Algorithm for Microcalcification Detection

```python
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.data.dicom_image import DicomImage
from medical_image.utils.image_utils import ImageExporter

# Load mammogram
mammogram = DicomImage("mammogram.dcm")
mammogram.load()

# Create output image
segmentation = mammogram.clone()

# Method 1: Difference of Gaussian
febds_dog = FebdsAlgorithm(method="dog")
febds_dog(mammogram, segmentation)
ImageExporter.save_as(segmentation)

# Method 2: Laplacian of Gaussian
febds_log = FebdsAlgorithm(method="log")
febds_log(mammogram, segmentation)

# Method 3: FFT with Butterworth filter
febds_fft = FebdsAlgorithm(method="fft")
febds_fft(mammogram, segmentation)
```

### Segmentation Algorithms Overview

#### K-Means Clustering

```python
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.data.dicom_image import DicomImage
from medical_image.utils.image_utils import ImageVisualizer

img = DicomImage("sample_image.dcm")
img.load()

# Create KMeans segmentation with 3 clusters
kmeans = KMeansAlgorithm(k=3, max_iter=100)
kmeans_output = img.clone()

kmeans(img, kmeans_output)

# Access results
print(f"Centroids: {kmeans.centroids}")
print(f"Converged: {kmeans.converged} in {kmeans.n_iter} iterations")
for s in kmeans.stats:
    print(s)

ImageVisualizer.show(kmeans_output, cmap='viridis', title='K-Means Mask')
```

#### Fuzzy C-Means (FCM)

```python
from medical_image.algorithms.fcm import FCMAlgorithm

fcm = FCMAlgorithm(c=3, m=2.0)
fcm_output = img.clone()

fcm(img, fcm_output)

# Access results
print(f"Centroids: {fcm.centroids}")
print(f"Labels shape: {fcm.labels.shape}")
for s in fcm.stats:
    print(s)
```

#### Possibilistic Fuzzy C-Means (PFCM)

```python
from medical_image.algorithms.pfcm import PFCMAlgorithm

# PFCM parameters:
# eta: Possibilistic typicality fuzziness factor.
# a, b: Controls balancing of typicalities inside centroid update routines.
pfcm = PFCMAlgorithm(c=3, m=2.0, eta=2.0, a=1.0, b=4.0)
pfcm_output = img.clone()

pfcm(img, pfcm_output)
```

#### Top-Hat Morphological Transform

```python
from medical_image.algorithms.top_hat import TopHatAlgorithm

# radius specifies the structure element size spanning over targeted elements
tophat = TopHatAlgorithm(radius=10)
tophat_output = img.clone()

tophat(img, tophat_output)
```

### Batch Processing with Algorithms

```python
from medical_image.algorithms.kmeans import KMeansAlgorithm

kmeans = KMeansAlgorithm(k=3)

images = [img1, img2, img3]
outputs = [i.clone() for i in images]

# Process all images in a batch
results = kmeans.apply_batch(images, outputs)
```

### Creating Custom Algorithms

```python
from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image

class MyCustomAlgorithm(Algorithm):
    def __init__(self, param1, param2, device=None):
        super().__init__(device=device)
        self.param1 = param1
        self.param2 = param2

    def apply(self, image: Image, output: Image) -> Image:
        # Implement your algorithm
        # Access input: image.pixel_data
        # Write output: output.pixel_data = ...

        # Example: Simple threshold
        threshold = self.param1
        output.pixel_data = (image.pixel_data > threshold).float()
        return output

# Use custom algorithm (callable interface handles precision)
algo = MyCustomAlgorithm(param1=100, param2=0.5, device="cpu")
output = image.clone()
algo(image, output)
```

---

## Advanced Topics

### Working with Annotations

```python
from medical_image.utils.annotation import Annotation, GeometryType

# Create annotation for calcification
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

# Attach to image
image.label = annotation

# Access annotation properties
print(annotation.abnormality_type)
print(annotation.pathology)
```

### GPU Acceleration

#### Moving Images to GPU

```python
# Move an image to GPU -- all subsequent operations follow automatically
image = DicomImage("image.dcm")
image.load()
image.to("cuda")

print(f"Image device: {image.device}")  # cuda:0

# Move back to CPU
image.to("cpu")
```

#### Automatic Device Inference with `resolve_device`

Filters and processing functions use `resolve_device` to automatically pick
the device from the input image. You only need to pass an explicit `device`
argument when you want to override this behaviour:

```python
from medical_image.utils.device import resolve_device

# Infers device from the image's pixel_data tensor
device = resolve_device(image)
print(device)  # e.g. cuda:0 if image was moved to GPU

# Explicit override
device = resolve_device(image, explicit="cpu")
```

#### `DeviceContext` for Memory Management

`DeviceContext` is a context manager that clears GPU cache on entry and exit
and provides automatic CPU fallback when CUDA is unavailable or an OOM error
occurs:

```python
from medical_image.utils.device import DeviceContext

with DeviceContext("cuda", verbose=True) as ctx:
    image.to(ctx.device)
    output = image.clone()
    Filters.gaussian_filter(image, output, sigma=2.0)
    print(ctx.memory_stats())
# GPU cache is automatically cleared on exit
```

#### `@gpu_safe` for OOM Fallback

The `@gpu_safe` decorator catches `torch.cuda.OutOfMemoryError` and
automatically retries the decorated function on CPU:

```python
from medical_image.utils.device import gpu_safe

@gpu_safe
def my_heavy_operation(image, output, device=None):
    Filters.gaussian_filter(image, output, sigma=5.0, device=device)
    Threshold.otsu_threshold(output, output, device=device)
    return output

# Will attempt on GPU first; falls back to CPU on OOM
result = my_heavy_operation(image, output, device="cuda")
```

#### Mixed Precision with the `Precision` Enum

Algorithms accept a `precision` parameter. When running on GPU with
`Precision.HALF` or `Precision.BFLOAT16`, the `__call__` method
automatically wraps execution in `torch.cuda.amp.autocast`:

```python
from medical_image.utils.device import Precision
from medical_image.algorithms.kmeans import KMeansAlgorithm

kmeans = KMeansAlgorithm(k=3, device="cuda")

# Override precision for faster inference at the cost of some accuracy
from medical_image.algorithms.algorithm import Algorithm
kmeans.precision = Precision.HALF

output = image.clone()
kmeans(image, output)
```

#### Pinned Memory for Faster Transfers

Pin an image's tensor to page-locked memory before transferring to GPU.
This can significantly speed up CPU-to-GPU copies:

```python
image.pin_memory()
image.to("cuda")  # faster transfer from pinned memory
```

### Memory Management for Large Datasets

```python
# Process large dataset in batches
batch_size = 10

for i in range(0, len(dataset), batch_size):
    # Load batch
    batch = dataset.load_batch(batch_size)

    # Process batch
    for image in batch:
        # Process image
        pass

    # Free memory
    dataset.destroy_batch()
    torch.cuda.empty_cache()  # Clear GPU cache
```

### Pipeline Example: Complete Workflow

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.process.filters import Filters
from medical_image.process.threshold import Threshold
from medical_image.process.morphology import MorphologyOperations
from medical_image.data.region_of_interest import RegionOfInterest
from medical_image.utils.image_utils import ImageExporter

# 1. Load image
image = DicomImage("mammogram.dcm")
image.load()

# 2. Extract ROI
roi_coords = [500, 500, 1500, 1500]
roi = RegionOfInterest(image, roi_coords)
roi_image = roi.load()

# 3. Preprocessing
filtered = roi_image.clone()
Filters.gaussian_filter(roi_image, filtered, sigma=1.5)

# 4. Enhancement
enhanced = filtered.clone()
Filters.gamma_correction(filtered, enhanced, gamma=1.2)

# 5. Segmentation
segmented = enhanced.clone()
Threshold.otsu_threshold(enhanced, segmented)

# 6. Post-processing
final = segmented.clone()
MorphologyOperations.morphology_closing(segmented, final, kernel_size=3)
MorphologyOperations.region_fill(final, final)

# 7. Save results
ImageExporter.save_as(final)
```

---

## Best Practices

### 1. Memory Management

```python
import torch

# Always load images when needed
image = DicomImage("large_image.dcm")
image.load()

# Process
# ...

# Clear when done
del image
torch.cuda.empty_cache()
```

### 2. Error Handling

```python
from medical_image.utils.ErrorHandler import ErrorMessages

try:
    image = DicomImage("image.dcm")
    image.load()
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error loading image: {e}")
```

### 3. Type Checking

```python
from medical_image.data.image import Image
import torch

def process_image(img: Image) -> torch.Tensor:
    """Process image and return result."""
    if not isinstance(img.pixel_data, torch.Tensor):
        raise ValueError("Pixel data must be a tensor")

    # Process
    result = img.pixel_data * 2
    return result
```

### 4. Logging

```python
from medical_image.utils.logging import logger, configure_logging

# Enable console and file logging
configure_logging(log_file="processing.log")

logger.info("Loading image...")
image = DicomImage("image.dcm")
image.load()
logger.info(f"Image loaded: {image.width}x{image.height}")
```

### 5. Testing

```python
import pytest
from medical_image.data.dicom_image import DicomImage

def test_image_loading():
    image = DicomImage("test_data/sample.dcm")
    image.load()

    assert image.width > 0
    assert image.height > 0
    assert image.pixel_data is not None

def test_filter_application():
    from medical_image.process.filters import Filters

    input_img = DicomImage("test_data/sample.dcm")
    input_img.load()

    output_img = input_img.clone()
    Filters.gaussian_filter(input_img, output_img, sigma=2.0)

    assert output_img.pixel_data is not None
```

### 6. Performance Optimization

```python
import torch
from medical_image.data.patch import PatchGrid
from medical_image.utils.device import DeviceContext

# Use DeviceContext for GPU-aware processing
with DeviceContext("cuda") as ctx:
    image.to(ctx.device)
    # All operations automatically use the GPU

# Process in batches for large datasets
# Use patch-based processing for very large images
patch_grid = PatchGrid(large_image, patch_size=(512, 512))

# Process patches in parallel (if needed)
from concurrent.futures import ThreadPoolExecutor

def process_patch(patch):
    # Process single patch
    return patch

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_patch, patch_grid.patches)
```

---

## Troubleshooting

### Common Issues

**Issue: Out of memory errors**
```python
# Solution: Use smaller batch sizes or patch-based processing
from medical_image.data.patch import PatchGrid
patch_grid = PatchGrid(image, patch_size=(256, 256))
```

**Issue: CUDA out of memory**
```python
# Solution 1: Use DeviceContext for automatic OOM handling
from medical_image.utils.device import DeviceContext

with DeviceContext("cuda") as ctx:
    image.to(ctx.device)
    output = image.clone()
    Filters.gaussian_filter(image, output, sigma=2.0)
# If OOM occurs inside the context, it automatically falls back to CPU
```

```python
# Solution 2: Use @gpu_safe on your processing functions
from medical_image.utils.device import gpu_safe

@gpu_safe
def process(image, output, device=None):
    Filters.gaussian_filter(image, output, sigma=2.0, device=device)
    return output

# Tries GPU first, falls back to CPU on OOM
result = process(image, output, device="cuda")
```

```python
# Solution 3: Manually clear cache and move to CPU
import torch
torch.cuda.empty_cache()
image.to("cpu")
```

**Issue: File not found**
```python
# Solution: Check file path and extension
import os
if not os.path.exists(file_path):
    print(f"File does not exist: {file_path}")
```

---

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed documentation
- Check [Algorithm Reference](algorithms.md) for algorithm details
- See [Dataset Guide](datasets.md) for dataset-specific information
- Read [Contributing](contributing.md) to contribute to the project