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
pip install -r requirements.txt
pip install -e .
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
# Display image information
image.display_info()

# Visualize the image
image.plot(cmap='gray')

# Use custom colormap
image.plot(cmap='hot')
```

### Saving Images

```python
# Save as PNG
image.to_png()  # Saves as 'mammogram.png'

# Save modified DICOM
image.save()  # Saves as 'mammogram_modified.dcm'
```

### Converting to NumPy

```python
import numpy as np

# Convert to NumPy array
pixel_array = image.to_numpy()
print(type(pixel_array))  # <class 'numpy.ndarray'>

# Perform NumPy operations
mean_intensity = np.mean(pixel_array)
std_intensity = np.std(pixel_array)
```

---

## Image Processing

### Applying Filters

#### Gaussian Filter

```python
from medical_image.process.filters import Filters
from medical_image.data.dicom_image import DicomImage

# Load image
input_image = DicomImage("input.dcm")
input_image.load()

# Create output image object
output_image = DicomImage("output.dcm")

# Apply Gaussian filter
Filters.gaussian_filter(input_image, output_image, sigma=2.0)

# Save result
output_image.to_png()
```

#### Median Filter (Noise Reduction)

```python
# Apply median filter to remove salt-and-pepper noise
Filters.median_filter(input_image, output_image, size=5)
```

#### Difference of Gaussian (Edge Detection)

```python
# DoG filter for edge enhancement
Filters.difference_of_gaussian(
    input_image, 
    output_image, 
    sigma_1=2.0, 
    sigma_2=1.7
)
```

#### Laplacian of Gaussian

```python
# LoG filter for blob detection
Filters.laplacian_of_gaussian(input_image, output_image, sigma=2.0)
```

### Brightness and Contrast Adjustment

```python
# Gamma correction
Filters.gamma_correction(input_image, output_image, gamma=1.25)

# Contrast and brightness adjustment
Filters.ContrastAdjust(
    input_image, 
    output_image, 
    contrast=50, 
    brightness=30
)
```

### Thresholding

#### Otsu's Method (Automatic)

```python
from medical_image.process.threshold import Threshold

# Automatic thresholding
Threshold.otsu_threshold(input_image, output_image)
output_image.plot(cmap='binary')
```

#### Sauvola's Method (Local Adaptive)

```python
# Local adaptive thresholding
Threshold.sauvola_threshold(
    input_image, 
    output_image,
    window_size=15,
    k=0.5,
    r=128
)
```

#### Variance-based Binarization

```python
# Binarize using local/global variance
Threshold.binarize(input_image, output_image, alpha=0.5)
```

### Frequency Domain Processing

```python
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.filters import Filters
import numpy as np

# Transform to frequency domain
freq_image = DicomImage("freq.dcm")
FrequencyOperations.fft(input_image, freq_image)

# Apply Butterworth band-pass filter
kernel = Filters.butterworth_kernel(freq_image, D_0=21, W=32, n=3)
freq_image.pixel_data = np.multiply(freq_image.pixel_data, kernel)

# Transform back to spatial domain
FrequencyOperations.inverse_fft(freq_image, output_image)
```

### Morphological Operations

```python
from medical_image.process.morphology import MorphologyOperations

# Morphological closing (fill small holes)
MorphologyOperations.morphology_closing(input_image, output_image, kernel_size=5)

# Region filling
MorphologyOperations.region_fill(input_image, output_image)
```

### Computing Metrics

```python
from medical_image.process.metrics import Metrics

# Calculate entropy
ent = Metrics.entropy(input_image, decimals=4)
print(f"Image entropy: {ent} bits")

# Calculate mutual information between two images
image1 = DicomImage("image1.dcm")
image2 = DicomImage("image2.dcm")
image1.load()
image2.load()

mi = Metrics.mutual_information(image1, image2)
print(f"Mutual information: {mi}")

# Calculate local variance
variance_image = DicomImage("variance.dcm")
Metrics.local_variance(input_image, variance_image, kernel=5)
```

---

## Working with Regions of Interest

### Bounding Box ROI

```python
from medical_image.data.region_of_interest import RegionOfInterest

# Define bounding box [x_min, y_min, x_max, y_max]
bbox = [100, 100, 400, 400]

# Create ROI
roi = RegionOfInterest(input_image, bbox)

# Extract ROI
cropped = roi.load()
cropped.plot()
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
roi = RegionOfInterest(input_image, polygon)
cropped = roi.load()
```

### Mask ROI

```python
import numpy as np

# Create binary mask
mask = np.zeros((input_image.height, input_image.width), dtype=bool)
mask[100:400, 100:400] = True

# Create and extract mask ROI
roi = RegionOfInterest(input_image, mask)
cropped = roi.load()
```

---

## Working with Patches

### Creating a Patch Grid

```python
from medical_image.data.patch import PatchGrid

# Divide image into 256x256 patches
patch_grid = PatchGrid(input_image, patch_size=(256, 256))

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

# Create new image with reconstructed data
output_image = DicomImage("reconstructed.dcm")
output_image.pixel_data = reconstructed
output_image.to_png()
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

# Load mammogram
mammogram = DicomImage("mammogram.dcm")
mammogram.load()

# Create output image
segmentation = DicomImage("segmentation.dcm")

# Method 1: Difference of Gaussian
febds_dog = FebdsAlgorithm(method="dog")
febds_dog.apply(mammogram, segmentation)
segmentation.to_png()

# Method 2: Laplacian of Gaussian
febds_log = FebdsAlgorithm(method="log")
febds_log.apply(mammogram, segmentation)

# Method 3: FFT with Butterworth filter
febds_fft = FebdsAlgorithm(method="fft")
febds_fft.apply(mammogram, segmentation)
```

### Creating Custom Algorithms

```python
from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image

class MyCustomAlgorithm(Algorithm):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def apply(self, image: Image, output: Image):
        # Implement your algorithm
        # Access input: image.pixel_data
        # Write output: output.pixel_data = ...
        
        # Example: Simple threshold
        threshold = self.param1
        output.pixel_data = (image.pixel_data > threshold).float()

# Use custom algorithm
algo = MyCustomAlgorithm(param1=100, param2=0.5)
algo.apply(input_image, output_image)
```

---

## Advanced Topics

### Working with Annotations

```python
from medical_image.utils.annotation import Annotation, AnnotationType

# Create annotation for calcification
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

# Attach to image
image.label = annotation

# Access annotation properties
print(annotation.abnormality_type)
print(annotation.pathology)
```

### GPU Acceleration

```python
import torch

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Images automatically use GPU when available
image = DicomImage("image.dcm")
image.load()
print(f"Image device: {image.device}")

# Manual device transfer (if needed)
image.pixel_data = image.pixel_data.to('cuda')

# Move back to CPU
image.pixel_data = image.pixel_data.to('cpu')
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

# 1. Load image
image = DicomImage("mammogram.dcm")
image.load()

# 2. Extract ROI
roi_coords = [500, 500, 1500, 1500]
roi = RegionOfInterest(image, roi_coords)
roi_image = roi.load()

# 3. Preprocessing
filtered = DicomImage("filtered.dcm")
Filters.gaussian_filter(roi_image, filtered, sigma=1.5)

# 4. Enhancement
enhanced = DicomImage("enhanced.dcm")
Filters.gamma_correction(filtered, enhanced, gamma=1.2)

# 5. Segmentation
segmented = DicomImage("segmented.dcm")
Threshold.otsu_threshold(enhanced, segmented)

# 6. Post-processing
final = DicomImage("final.dcm")
MorphologyOperations.morphology_closing(segmented, final, kernel_size=3)
MorphologyOperations.region_fill(final, final)

# 7. Save results
final.to_png()
```

---

## Best Practices

### 1. Memory Management

```python
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
from log_manager import logger

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
    
    output_img = DicomImage("output.dcm")
    Filters.gaussian_filter(input_img, output_img, sigma=2.0)
    
    assert output_img.pixel_data is not None
```

### 6. Performance Optimization

```python
# Use GPU when available
import torch

if torch.cuda.is_available():
    print("Using GPU acceleration")
else:
    print("Using CPU")

# Process in batches for large datasets
# Use patch-based processing for very large images
patch_grid = PatchGrid(large_image, patch_size=(512, 512))

# Process patches in parallel (if needed)
from concurrent.futures import ThreadPoolExecutor

def process_patch(patch):
    # Process single patch
    return processed_patch

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_patch, patch_grid.patches)
```

---

## Troubleshooting

### Common Issues

**Issue: Out of memory errors**
```python
# Solution: Use smaller batch sizes or patch-based processing
patch_grid = PatchGrid(image, patch_size=(256, 256))
```

**Issue: CUDA out of memory**
```python
# Solution: Clear cache and move to CPU
torch.cuda.empty_cache()
image.pixel_data = image.pixel_data.to('cpu')
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
