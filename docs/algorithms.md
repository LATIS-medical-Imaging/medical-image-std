# Algorithm Reference

## Overview

This document provides detailed information about the image processing algorithms implemented in the Medical Image Standard Library.

---

## Table of Contents

1. [FEBDS Algorithm](#febds-algorithm)
2. [Filtering Algorithms](#filtering-algorithms)
3. [Thresholding Algorithms](#thresholding-algorithms)
4. [Morphological Algorithms](#morphological-algorithms)
5. [Frequency Domain Algorithms](#frequency-domain-algorithms)

---

## FEBDS Algorithm

### Frequency-Enhanced Band-pass Detection System

**Purpose:** Detection of microcalcifications in mammography images.

**Reference:** Based on the paper "Mammograms calcifications segmentation based on band-pass Fourier filtering and adaptive statistical thresholding"

### Algorithm Overview

FEBDS is a multi-stage algorithm designed to enhance and detect small calcifications in mammography images. It combines frequency domain filtering with adaptive thresholding and morphological post-processing.

### Methods

The algorithm supports three different frequency enhancement methods:

#### 1. Difference of Gaussian (DoG)

**Mathematical Definition:**
```
DoG(x, y) = G(x, y, σ₁) - G(x, y, σ₂)
```

Where:
- `G(x, y, σ)` is a Gaussian function with standard deviation σ
- `σ₁ > σ₂` (typically σ₁ = 2.0, σ₂ = 1.7)

**Properties:**
- Approximates the Laplacian of Gaussian
- Enhances edges and blob-like structures
- Computationally efficient

**Usage:**
```python
from medical_image.algorithms.FEBDS import FebdsAlgorithm

febds = FebdsAlgorithm(method="dog")
febds.apply(input_image, output_image)
```

#### 2. Laplacian of Gaussian (LoG)

**Mathematical Definition:**
```
LoG(x, y) = -1/(πσ⁴) * [1 - (x² + y²)/(2σ²)] * exp(-(x² + y²)/(2σ²))
```

**Properties:**
- Detects regions of rapid intensity change
- Blob detection
- Rotationally symmetric
- Zero-crossing detector

**Usage:**
```python
febds = FebdsAlgorithm(method="log")
febds.apply(input_image, output_image)
```

#### 3. FFT with Butterworth Band-pass Filter

**Mathematical Definition:**

Butterworth band-pass filter in frequency domain:
```
H(u, v) = 1 / [1 + ((D(u,v) * W) / (D(u,v)² - D₀²))^(2n)]
```

Where:
- `D(u, v)` = distance from frequency point (u, v) to origin
- `D₀` = center frequency (default: 21)
- `W` = bandwidth (default: 32)
- `n` = filter order (default: 3)

**Properties:**
- Maximally flat frequency response
- Selects specific frequency band
- Effective for periodic noise removal
- Preserves calcification frequencies

**Usage:**
```python
febds = FebdsAlgorithm(method="fft")
febds.apply(input_image, output_image)
```

### Pipeline Stages

All three methods follow the same processing pipeline after frequency enhancement:

#### Stage 1: Frequency Enhancement
- **DoG/LoG:** Spatial domain filtering
- **FFT:** Frequency domain filtering with Butterworth kernel

#### Stage 2: Median Filtering
- **Kernel size:** 5×5
- **Purpose:** Remove salt-and-pepper noise while preserving edges

#### Stage 3: Gamma Correction
- **Gamma value:** 1.25
- **Purpose:** Brightness adjustment and contrast enhancement
- **Formula:** `output = input^(1/γ)`

#### Stage 4: Thresholding
- **DoG/LoG:** Otsu's automatic thresholding
- **FFT:** Variance-based adaptive thresholding (α = 1)

#### Stage 5: Morphological Closing
- **Purpose:** Fill small gaps and smooth contours
- **Operation:** Dilation followed by erosion

#### Stage 6: Region Filling
- **Purpose:** Fill holes in detected regions
- **Result:** Solid segmentation masks

### Performance Characteristics

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| DoG | Fast | Good | General purpose, quick screening |
| LoG | Medium | Very Good | Precise blob detection |
| FFT | Slow | Excellent | High-quality segmentation, research |

### Example: Complete FEBDS Workflow

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.algorithms.FEBDS import FebdsAlgorithm
import matplotlib.pyplot as plt

# Load mammogram
mammogram = DicomImage("mammogram.dcm")
mammogram.load()

# Apply FEBDS with different methods
methods = ["dog", "log", "fft"]
results = []

for method in methods:
    output = DicomImage(f"output_{method}.dcm")
    febds = FebdsAlgorithm(method=method)
    febds.apply(mammogram, output)
    results.append(output)

# Visualize results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(mammogram.to_numpy(), cmap='gray')
axes[0].set_title('Original')

for i, (method, result) in enumerate(zip(methods, results)):
    axes[i+1].imshow(result.to_numpy(), cmap='binary')
    axes[i+1].set_title(f'FEBDS ({method.upper()})')

plt.show()
```

---

## Filtering Algorithms

### Gaussian Filter

**Purpose:** Smooth images and reduce noise

**Mathematical Definition:**
```
G(x, y) = (1/(2πσ²)) * exp(-(x² + y²)/(2σ²))
```

**Parameters:**
- `sigma`: Standard deviation (controls blur amount)
  - Small σ (0.5-1.0): Slight smoothing
  - Medium σ (1.5-3.0): Moderate smoothing
  - Large σ (>3.0): Heavy smoothing

**Properties:**
- Removes high-frequency noise
- Preserves edges better than box filter
- Separable (can be applied in 1D twice)

**Use Cases:**
- Preprocessing before edge detection
- Noise reduction
- Image pyramid construction

### Median Filter

**Purpose:** Remove salt-and-pepper noise

**Algorithm:**
1. For each pixel, consider neighborhood window
2. Sort all pixel values in window
3. Replace center pixel with median value

**Parameters:**
- `size`: Window size (must be odd)
  - 3×3: Minimal smoothing
  - 5×5: Standard noise removal
  - 7×7+: Heavy noise removal

**Properties:**
- Non-linear filter
- Preserves edges
- Effective against impulse noise
- Does not blur edges like Gaussian

**Use Cases:**
- Removing isolated noise pixels
- Preprocessing for segmentation
- Preserving sharp boundaries

### Difference of Gaussian (DoG)

**Purpose:** Edge and blob detection

**Algorithm:**
```
DoG = Gaussian(σ₁) - Gaussian(σ₂)
```

**Parameters:**
- `sigma_1`: Larger standard deviation
- `sigma_2`: Smaller standard deviation
- Ratio σ₁/σ₂ typically 1.6 (scale-space theory)

**Properties:**
- Approximates Laplacian of Gaussian
- Multi-scale feature detection
- Computationally efficient
- Used in SIFT algorithm

**Use Cases:**
- Blob detection
- Interest point detection
- Microcalcification enhancement

### Laplacian of Gaussian (LoG)

**Purpose:** Blob detection and edge enhancement

**Properties:**
- Second derivative operator
- Isotropic (rotation invariant)
- Zero-crossings indicate edges
- Sensitive to noise (requires smoothing)

**Parameters:**
- `sigma`: Gaussian smoothing parameter
  - Controls scale of detected features

**Use Cases:**
- Blob detection at specific scales
- Edge detection
- Feature extraction

### Butterworth Filter

**Purpose:** Frequency-selective filtering

**Types:**
- **Low-pass:** Remove high frequencies (smoothing)
- **High-pass:** Remove low frequencies (sharpening)
- **Band-pass:** Keep specific frequency range

**Parameters:**
- `D_0`: Cutoff/center frequency
- `W`: Bandwidth (for band-pass)
- `n`: Filter order (controls transition sharpness)

**Properties:**
- Smooth frequency response
- No ringing artifacts
- Adjustable transition steepness

**Use Cases:**
- Periodic noise removal
- Frequency band selection
- Texture analysis

### Gamma Correction

**Purpose:** Brightness and contrast adjustment

**Mathematical Definition:**
```
output = input^(1/γ)
```

**Parameters:**
- `gamma`: Correction factor
  - γ < 1: Brightens image
  - γ = 1: No change
  - γ > 1: Darkens image

**Properties:**
- Non-linear transformation
- Preserves relative contrasts
- Compensates for display characteristics

**Use Cases:**
- Display calibration
- Contrast enhancement
- Preprocessing for visualization

### Contrast Adjustment

**Purpose:** Adjust image contrast and brightness

**Mathematical Definition:**
```
output = α * input + β
```

Where:
- `α = contrast/2047 + 1`
- `β = brightness - contrast`

**Parameters:**
- `contrast`: Contrast adjustment value
- `brightness`: Brightness adjustment value

**Use Cases:**
- Image enhancement
- Normalization
- Preprocessing

---

## Thresholding Algorithms

### Otsu's Method

**Purpose:** Automatic global thresholding

**Algorithm:**
1. Compute histogram of image
2. For each possible threshold:
   - Calculate between-class variance
3. Select threshold that maximizes variance

**Mathematical Definition:**
```
σ²_between = w₀ * w₁ * (μ₀ - μ₁)²
```

Where:
- `w₀, w₁`: Class weights
- `μ₀, μ₁`: Class means

**Properties:**
- Fully automatic (no parameters)
- Optimal for bimodal histograms
- Global threshold
- Fast computation

**Use Cases:**
- Binary segmentation
- Foreground/background separation
- Document image processing

**Limitations:**
- Assumes bimodal distribution
- Single global threshold
- Sensitive to lighting variations

### Sauvola's Method

**Purpose:** Local adaptive thresholding

**Mathematical Definition:**
```
T(x, y) = m(x, y) * [1 + k * (s(x, y)/R - 1)]
```

Where:
- `m(x, y)`: Local mean
- `s(x, y)`: Local standard deviation
- `k`: Weighting factor (default: 0.5)
- `R`: Dynamic range (default: 128)

**Parameters:**
- `window_size`: Local neighborhood size
- `k`: Sensitivity to local variation
- `r`: Dynamic range parameter

**Properties:**
- Adaptive to local statistics
- Handles varying illumination
- Preserves fine details
- Computationally intensive

**Use Cases:**
- Document image binarization
- Uneven illumination
- Text extraction

### Variance-based Binarization

**Purpose:** Adaptive thresholding using variance

**Algorithm:**
1. Compute local variance in windows
2. Compute global variance
3. Threshold based on variance ratio

**Mathematical Definition:**
```
Binary(x, y) = 1 if σ²_local(x, y) > α * σ²_global else 0
```

**Parameters:**
- `alpha`: Scaling factor (typically 0.5-1.5)

**Properties:**
- Texture-aware
- Adaptive to local complexity
- Good for structured patterns

**Use Cases:**
- Microcalcification detection
- Texture segmentation
- Pattern recognition

---

## Morphological Algorithms

### Morphological Closing

**Purpose:** Fill small holes and gaps

**Algorithm:**
```
Closing = Erosion(Dilation(Image))
```

**Operations:**
1. **Dilation:** Expand bright regions
2. **Erosion:** Shrink bright regions

**Properties:**
- Fills small holes
- Smooths contours
- Connects nearby objects
- Preserves object size

**Parameters:**
- `kernel_size`: Structuring element size

**Use Cases:**
- Noise removal
- Gap filling
- Contour smoothing

### Region Filling

**Purpose:** Fill holes in binary regions

**Algorithm:**
1. Identify connected components
2. Fill interior holes
3. Preserve boundaries

**Properties:**
- Creates solid regions
- Removes internal holes
- Preserves external boundaries

**Use Cases:**
- Segmentation post-processing
- Mask creation
- Object completion

---

## Frequency Domain Algorithms

### Fast Fourier Transform (FFT)

**Purpose:** Transform image to frequency domain

**Mathematical Definition:**
```
F(u, v) = Σ Σ f(x, y) * exp(-2πi(ux/M + vy/N))
```

**Properties:**
- Decomposes image into frequency components
- Enables frequency-selective filtering
- Reveals periodic patterns
- Computationally efficient (O(N log N))

**Use Cases:**
- Frequency analysis
- Periodic noise removal
- Texture analysis
- Compression

### Inverse FFT

**Purpose:** Transform back to spatial domain

**Algorithm:**
Inverse of FFT operation

**Use Cases:**
- Reconstructing filtered images
- Completing frequency domain processing

---

## Algorithm Selection Guide

### For Noise Reduction:
- **Salt-and-pepper noise:** Median filter
- **Gaussian noise:** Gaussian filter
- **Periodic noise:** FFT + Butterworth filter

### For Edge Detection:
- **General edges:** DoG or LoG
- **Fine edges:** LoG with small σ
- **Coarse edges:** DoG with large σ

### For Segmentation:
- **Uniform illumination:** Otsu's method
- **Varying illumination:** Sauvola's method
- **Textured regions:** Variance-based binarization

### For Microcalcification Detection:
- **Quick screening:** FEBDS with DoG
- **High accuracy:** FEBDS with FFT
- **Balanced:** FEBDS with LoG

---

## Performance Optimization Tips

1. **Use appropriate data types:**
   - Float32 for processing
   - Uint8 for storage

2. **Leverage GPU acceleration:**
   - Ensure tensors are on GPU
   - Batch operations when possible

3. **Choose efficient algorithms:**
   - DoG faster than LoG
   - Median filter slower than Gaussian

4. **Optimize parameters:**
   - Smaller kernels = faster processing
   - Larger kernels = better quality

5. **Pipeline optimization:**
   - Combine operations when possible
   - Minimize data transfers

---

## References

1. **FEBDS Algorithm:**
   - "Mammograms calcifications segmentation based on band-pass Fourier filtering and adaptive statistical thresholding"
   - https://www.researchgate.net/publication/306253912

2. **Otsu's Method:**
   - https://en.wikipedia.org/wiki/Otsu%27s_method

3. **Butterworth Filter:**
   - https://en.wikipedia.org/wiki/Butterworth_filter

4. **Gamma Correction:**
   - https://en.wikipedia.org/wiki/Gamma_correction

5. **Entropy and Information Theory:**
   - https://en.wikipedia.org/wiki/Entropy_(information_theory)
