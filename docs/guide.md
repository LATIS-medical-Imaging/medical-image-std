# Medical Image Standard Library -- Complete API Guide

> **Package:** `medical-image-std`
> **Version:** 0.5.0
> **Python:** >= 3.11
> **License:** MIT
> **Last Updated:** 2026-04-15

A PyTorch-based framework for medical image loading, processing, segmentation, and annotation. Built around a central **Image** abstraction with lazy loading, GPU acceleration, and COCO-compatible dataset export.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Architecture Overview](#2-architecture-overview)
3. [Image (Core Abstraction)](#3-image-core-abstraction)
   - [DicomImage](#31-dicomimage)
   - [PNGImage](#32-pngimage)
   - [InMemoryImage](#33-inmemoryimage)
4. [Patch-Based Processing](#4-patch-based-processing)
   - [PatchGrid](#41-patchgrid)
   - [Patch](#42-patch)
5. [Region of Interest](#5-region-of-interest)
6. [Annotations](#6-annotations)
   - [GeometryType](#61-geometrytype-enum)
   - [Annotation](#62-annotation)
7. [Image Processing](#7-image-processing)
   - [Filters](#71-filters)
   - [Threshold](#72-threshold)
   - [MorphologyOperations](#73-morphologyoperations)
   - [FrequencyOperations](#74-frequencyoperations)
   - [Metrics](#75-metrics)
   - [MammographyPreprocessing](#76-mammographypreprocessing)
8. [Algorithms](#8-algorithms)
   - [Algorithm (Base)](#81-algorithm-base-class)
   - [TopHatAlgorithm](#82-tophatalgorithm)
   - [KMeansAlgorithm](#83-kmeansalgorithm)
   - [FCMAlgorithm](#84-fcmalgorithm)
   - [PFCMAlgorithm](#85-pfcmalgorithm)
   - [FebdsAlgorithm](#86-febdsalgorithm)
   - [BreastMaskAlgorithm](#87-breastmaskalgorithm)
   - [DicomWindowAlgorithm](#88-dicomwindowalgorithm)
   - [GrailWindowAlgorithm](#89-grailwindowalgorithm)
   - [BitDepthNormAlgorithm](#810-bitdepthnormalgorithm)
9. [Datasets](#9-datasets)
   - [BaseDataset](#91-basedataset)
   - [INbreastDataset](#92-inbreastdataset)
   - [CBISDDSMDataset](#93-cbisddsmdataset)
   - [COCO JSON Export / Import](#94-coco-json-export--import)
10. [GPU & Device Management](#10-gpu--device-management)
11. [Utilities](#11-utilities)
12. [Usage Examples with Visualizations](#12-usage-examples-with-visualizations)
13. [Running Tests](#13-running-tests)

---

## 1. Installation

### From PyPI (recommended)

```bash
pip install medical-image-std
```

### From source (development)

```bash
git clone https://github.com/LATIS-DocumentAI-Group/medical-image-std.git
cd medical-image-std

# Using uv (recommended)
uv venv && uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

### Verify

```python
import medical_image
print(medical_image.__version__) 
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Tensor backend, GPU acceleration |
| `pydicom` | DICOM file parsing |
| `numpy` | Array operations |
| `Pillow` | PNG/JPEG I/O |
| `scikit-image` | Morphological structuring elements |
| `pandas` | Dataset CSV parsing |
| `matplotlib` | Visualization |

---

## 2. Architecture Overview

The framework is organized into four layers. Everything revolves around the **Image** abstract class.

```mermaid
flowchart TB
    subgraph Data_Layer
        IMG[Image]
        DICOM[DicomImage]
        PNG[PNGImage]
        MEM[InMemoryImage]
        PATCH[PatchGrid / Patch]
        ROI[RegionOfInterest]
        ANN[Annotation]
    end

    subgraph Processing_Layer
        FIL[Filters]
        THR[Threshold]
        MOR[MorphologyOperations]
        FRQ[FrequencyOperations]
        MET[Metrics]
        MAM[MammographyPreprocessing]
    end

    subgraph Algorithm_Layer
        ALG[Algorithm]
        TH[TopHat]
        KM[KMeans]
        FCM[FCM]
        PFCM[PFCM]
        FEBDS[FEBDS]
        BM[BreastMask]
        DW[DicomWindow]
        GW[GrailWindow]
        BDN[BitDepthNorm]
    end

    subgraph Dataset_Layer
        BD[BaseDataset]
        INB[INbreastDataset]
        CBIS[CBISDDSMDataset]
    end

    IMG --> DICOM
    IMG --> PNG
    IMG --> MEM
    IMG --> PATCH
    IMG --> ROI
    IMG o-- ANN

    FIL --> IMG
    THR --> IMG
    MOR --> IMG
    FRQ --> IMG
    MET --> IMG
    MAM --> IMG

    ALG --> TH
    ALG --> KM
    ALG --> FCM
    ALG --> PFCM
    ALG --> FEBDS
    ALG --> BM
    ALG --> DW
    ALG --> GW
    ALG --> BDN
    ALG --> IMG

    BD --> INB
    BD --> CBIS
    BD --> IMG
```
### Design Principles

| Principle | Description |
|-----------|-------------|
| **Lazy Loading** | `Image.__init__()` stores the path; `load()` reads pixels on demand |
| **PyTorch-First** | All pixel data is `torch.Tensor` -- GPU-ready from the start |
| **Device Agnostic** | Every operation accepts `device=None`; resolved automatically from the input image |
| **Static Processing** | Filters, thresholds, morphology are stateless static methods -- composable and side-effect-free |
| **Fluent API** | `image.load().to("cuda").pin_memory()` -- chainable calls |
| **Aggregation** | An Image *optionally* holds a list of Annotations; images can exist without annotations |

---

## 3. Image (Core Abstraction)

**Module:** `medical_image.data.image`
**Class:** `Image` (Abstract Base Class)

The central type in the framework. Every processing operation, algorithm, dataset, and utility works with `Image`.

### Constructor

```python
Image(
    file_path: Optional[str] = None,
    array: Optional[Union[np.ndarray, torch.Tensor]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    source_image: Optional[Image] = None,
)
```

Four construction paths (mutually exclusive):

| Path | Trigger | Behavior |
|------|---------|----------|
| **File** | `file_path` provided | Stores path, pixel data loaded later via `load()` |
| **Array** | `array` provided | Wraps numpy/tensor as `pixel_data` immediately |
| **Clone** | `source_image` provided | Copies metadata, clones `pixel_data` tensor, shares annotations |
| **Empty** | No arguments (or `width`/`height` only) | Shell with no pixel data |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `width` | `Optional[int]` | Derived from `pixel_data.shape[-1]` when loaded; falls back to `_width` |
| `height` | `Optional[int]` | Derived from `pixel_data.shape[-2]` when loaded; falls back to `_height` |
| `device` | `torch.device` | From `pixel_data.device` when loaded; falls back to `_device` |
| `pixel_data` | `Optional[torch.Tensor]` | The image tensor (`None` until loaded) |
| `file_path` | `Optional[str]` | Path to the source file |
| `annotations` | `Optional[List[Annotation]]` | Attached annotations (`None` by default) |

### Methods

#### Loading & I/O

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `load()` | *abstract* | -- | Load pixel data from file (subclass-specific) |
| `save()` | *abstract* | -- | Save pixel data to file (subclass-specific) |
| `ensure_loaded()` | `() -> Image` | self | Raise `DicomDataNotLoadedError` if `pixel_data is None` |

#### Device Management

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `to(device)` | `(Union[str, torch.device]) -> Image` | self | Move pixel data to device (in-place, chainable) |
| `pin_memory()` | `() -> Image` | self | Pin tensor to page-locked memory for faster GPU transfers |

#### Construction

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `from_file(path)` | classmethod | `Image` | Construct from file path |
| `from_array(array)` | classmethod | `Image` | Construct from numpy array or torch tensor |
| `from_image(other)` | classmethod | `Image` | Construct by copying another image |
| `empty(width, height)` | classmethod | `Image` | Construct an empty shell |
| `clone()` | `() -> Image` | `Image` | Lightweight copy (clones tensor, not DICOM/PIL objects) |

#### Annotations

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_annotation(ann)` | `(Annotation) -> None` | -- | Append annotation; initializes list if `None` |
| `remove_annotation(idx)` | `(int) -> Annotation` | `Annotation` | Remove and return annotation at index |

#### JSON Serialization

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `to_json(file_path=None)` | `(Optional[str]) -> str` | JSON string | Serialize metadata + annotations; optionally write to file |
| `from_json(json_input)` | classmethod `(str) -> Image` | `Image` | Deserialize from JSON string or file path |

**`to_json` output schema:**

```json
{
    "file_path": "/path/to/image.dcm",
    "width": 2560,
    "height": 3328,
    "image_type": "DicomImage",
    "annotations": [ ... ]
}
```

#### Factory Function

```python
image_from_json(json_input: str) -> Image
```

Module-level function. Reads `image_type` from JSON and dispatches to the correct subclass (`DicomImage`, `PNGImage`, or `InMemoryImage`).

#### Display

| Method | Signature | Description |
|--------|-----------|-------------|
| `display_info()` | `() -> None` | Log file path, dimensions, device, annotation count |
| `__repr__()` | `() -> str` | One-line summary |

---

### 3.1 DicomImage

**Module:** `medical_image.data.dicom_image`

| | |
|---|---|
| **Inherits** | `Image` |
| **Constructor** | `DicomImage(file_path: str)` -- must be a `.dcm` file |
| **Extra Attribute** | `dicom_data: Optional[pydicom.Dataset]` |
| `load()` | Reads DICOM via `pydicom`, extracts pixel array as `torch.Tensor` |
| `save()` | Writes modified pixel data back to `{name}_modified.dcm` |

### 3.2 PNGImage

**Module:** `medical_image.data.png_image`

| | |
|---|---|
| **Inherits** | `Image` |
| **Constructor** | `PNGImage(file_path: str)` -- must be a `.png` file |
| **Extra Attribute** | `_pil_image: Optional[PIL.Image]` |
| `load()` | Opens via PIL, converts to float tensor |
| `save()` | Converts to uint8, saves as `{name}_modified.png` |

### 3.3 InMemoryImage

**Module:** `medical_image.data.in_memory_image`

| | |
|---|---|
| **Inherits** | `Image` |
| **Constructor** | Same as `Image` (all parameters) |
| `load()` | No-op |
| `save()` | No-op |
| **Use case** | Intermediate processing results, temporary images, test fixtures |

---

## 4. Patch-Based Processing

**Module:** `medical_image.data.patch`

Split large images into a grid of patches for memory-efficient processing, then reassemble.

### 4.1 PatchGrid

```python
PatchGrid(parent_image: Image, patch_size: Tuple[int, int])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `parent_image` | `Image` | Source image (must be loaded) |
| `patch_size` | `Tuple[int, int]` | `(height, width)` of each patch |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `parent` | `Image` | Source image |
| `patch_h`, `patch_w` | `int` | Patch dimensions |
| `patches` | `List[Patch]` | All patches (flat list) |
| `grid` | `List[List[Patch]]` | 2D grid of patches |
| `pad_bottom`, `pad_right` | `int` | Padding added to make image divisible |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `reconstruct()` | `() -> torch.Tensor` | `Tensor` | Reassemble full image from patches (removes padding) |

### 4.2 Patch

Single patch extracted from a `PatchGrid`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `parent` | `Image` | Source image |
| `row_idx`, `col_idx` | `int` | Grid position |
| `x`, `y` | `int` | Top-left pixel coordinates in original image |
| `pixel_data` | `torch.Tensor` | Patch tensor |
| `is_padded` | `bool` | True if patch contains padding pixels |
| `width`, `height` | `int` | Patch dimensions (computed from tensor) |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `grid_id()` | `() -> Tuple[int, int]` | `(row, col)` | Grid position |
| `pixel_position()` | `() -> Tuple[int, int]` | `(x, y)` | Pixel coordinates in parent image |
| `to_numpy()` | `() -> np.ndarray` | array | Convert to numpy |
| `load()` | `() -> Image` | `Image` | Convert patch to a standalone Image |

---

## 5. Region of Interest

**Module:** `medical_image.data.region_of_interest`
**Class:** `RegionOfInterest`

Extract a sub-region from an image using bounding box, polygon, or mask coordinates.

### Constructor

```python
RegionOfInterest(
    image: Image,
    coordinates: Union[List[int], List[Tuple[int, int]], np.ndarray],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `Image` | Source image |
| `coordinates` | varies | Bounding box `[x_min, y_min, x_max, y_max]`, polygon `[(x,y), ...]`, or 2D mask array |

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `from_center(image, cx, cy, half_size)` | classmethod | `RegionOfInterest` | Create square ROI from center + half-size |
| `load()` | `() -> Image` | `Image` | Crop the region from the image, return new Image |
| `normalize(image, divisor)` | static | `Image` | Normalize pixel values by dividing |

---

## 6. Annotations

**Module:** `medical_image.utils.annotation`

### 6.1 GeometryType Enum

| Member | Coordinate Format | Description |
|--------|-------------------|-------------|
| `RECTANGLE` | `[x_min, y_min, x_max, y_max]` | Axis-aligned bounding box |
| `ELLIPSE` | `[cx, cy, rx, ry]` | Center + radii |
| `POLYGON` | `[(x1,y1), (x2,y2), ...]` | Ordered vertices (>= 3) |
| `BOUNDING_BOX` | alias for `RECTANGLE` | Backward compatibility |

### 6.2 Annotation

```python
Annotation(
    shape: GeometryType,
    coordinates: Union[List[int], List[Tuple[int, int]]],
    label: str,
    metadata: Optional[dict] = None,
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `GeometryType` | Yes | Geometry type |
| `coordinates` | varies | Yes | Shape-specific coordinates |
| `label` | `str` | Yes | Label (e.g. `"mass"`, `"calcification"`) |
| `metadata` | `Optional[dict]` | No | Extra info (BI-RADS, pathology, etc.) |

**Computed Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `center` | `Tuple[float, float]` | Centroid, computed automatically in constructor |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_bounding_box()` | `() -> List[int]` | `[x_min, y_min, x_max, y_max]` | Enclosing bounding box for any shape |
| `get_roi(padding, roi_type, image_shape)` | `(int, str, Optional[Tuple]) -> dict` | `{"type": ..., "coordinates": ...}` | Padded ROI as bbox, rectangle, or ellipse |
| `to_dict()` | `() -> dict` | dict | Serialize to JSON-compatible dict |
| `from_dict(data)` | classmethod `(dict) -> Annotation` | `Annotation` | Deserialize from dict |

**`get_roi` parameters:**

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `padding` | `int` | `0` | Extra pixels on each side |
| `roi_type` | `str` | `"bbox"` | `"bbox"`, `"rectangle"`, `"ellipse"` |
| `image_shape` | `Optional[Tuple[int,int]]` | `None` | `(height, width)` for clamping |

**`to_dict` output schema:**

```json
{
    "shape": "RECTANGLE",
    "coordinates": [100, 150, 200, 250],
    "label": "mass",
    "center": [150.0, 200.0],
    "bounding_box": [100, 150, 200, 250],
    "metadata": {"birads": 4}
}
```

---

## 7. Image Processing

All processing methods are **static**, decorated with `@requires_loaded`, and accept an optional `device=None` parameter (auto-resolved from the input image). They follow the pattern:

```python
ClassName.method(input_image, output_image, ..., device=None)
```

### 7.1 Filters

**Module:** `medical_image.process.filters`

| Method | Signature | Description |
|--------|-----------|-------------|
| `convolution` | `(image, output, kernel: Tensor, device) -> Image` | Custom 2D convolution |
| `gaussian_filter` | `(image, output, sigma: float, device, truncate=4.0) -> Image` | Gaussian blur |
| `gaussian_filter_batch` | `(images: Tensor, sigma, device, truncate) -> Tensor` | Batch Gaussian (B,C,H,W input) |
| `median_filter` | `(image, output, size: int, device) -> Image` | Median filter (odd size) |
| `butterworth_kernel` | `(image, output, D_0=21, W=32, n=3, device) -> Image` | Butterworth band-pass |
| `difference_of_gaussian` | `(image, output, low_sigma, high_sigma, device, truncate) -> Image` | DoG edge enhancement |
| `laplacian_of_gaussian` | `(image, output, sigma, device) -> Image` | LoG edge detection |
| `gamma_correction` | `(image, output, gamma: float, device) -> Image` | Gamma correction |
| `contrast_adjust` | `(image, output, contrast, brightness, device) -> Image` | Contrast/brightness adjustment |

### 7.2 Threshold

**Module:** `medical_image.process.threshold`

| Method | Signature | Description |
|--------|-----------|-------------|
| `otsu_threshold` | `(image, output, device) -> Image` | Global Otsu binarization |
| `sauvola_threshold` | `(image, output, window_size=10, k=0.5, r=128, device) -> Image` | Adaptive local thresholding |
| `binarize` | `(image, output, alpha, device) -> Image` | Variance-based binarization |

### 7.3 MorphologyOperations

**Module:** `medical_image.process.morphology`

| Method | Signature | Description |
|--------|-----------|-------------|
| `morphology_closing` | `(image, output, kernel_size=7, device) -> Image` | Dilation then erosion |
| `region_fill` | `(image, output, device) -> Image` | Fill holes in binary image |
| `erosion` | `(image, output, radius=4, device) -> Image` | Grayscale erosion (disk SE) |
| `dilation` | `(image, output, radius=4, device) -> Image` | Grayscale dilation (disk SE) |
| `white_top_hat` | `(image, output, radius=4, device) -> Image` | Bright structure extraction (I - opening) |

### 7.4 FrequencyOperations

**Module:** `medical_image.process.frequency`

| Method | Signature | Description |
|--------|-----------|-------------|
| `fft` | `(image, output, device) -> Image` | 2D Fast Fourier Transform |
| `inverse_fft` | `(image, output, device) -> Image` | Inverse 2D FFT |

### 7.5 Metrics

**Module:** `medical_image.process.metrics`

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `entropy` | `(image, decimals=4, device) -> float` | `float` | Shannon entropy |
| `joint_entropy` | `(image1, image2, decimals, device) -> float` | `float` | Joint entropy of two images |
| `mutual_information` | `(image1, image2, decimals, device) -> float` | `float` | MI = H(A) + H(B) - H(A,B) |
| `local_variance` | `(image, output, kernel, device) -> Image` | `Image` | Local variance map |
| `variance` | `(image, output, device) -> Image` | `Image` | Global variance |

### 7.6 MammographyPreprocessing

**Module:** `medical_image.process.mammography`

Specialized methods for mammogram preprocessing.

| Method | Signature | Description |
|--------|-----------|-------------|
| `breast_mask` | `(image, output, device) -> Image` | Binary breast region mask (Otsu + largest CC) |
| `apply_breast_mask` | `(image, output, device) -> Image` | Multiply image by breast mask |
| `dicom_window` | `(image, output, window_center, window_width, device) -> Image` | DICOM WC/WW intensity mapping |
| `grail_window` | `(image, output, n_scales=3, n_orientations=6, delta=300, k_max=3, device) -> Image` | Automatic windowing via Gabor MI optimization |
| `normalize_bit_depth` | `(image, output, bits_stored=None, target_max=255.0, device) -> Image` | Normalize based on DICOM BitsStored tag |

---

## 8. Algorithms

### 8.1 Algorithm (Base Class)

**Module:** `medical_image.algorithms.algorithm`

```python
Algorithm(device: str = None, precision: Precision = Precision.FULL)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `None` (auto-detect) | `"cuda"` or `"cpu"` |
| `precision` | `Precision` | `FULL` | `FULL` (fp32), `HALF` (fp16), `BFLOAT16` |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `apply(image, output)` | *abstract* | `Image` | Apply algorithm to single image |
| `apply_batch(images, outputs)` | `(List, List) -> List[Image]` | `List[Image]` | Process batch (default: loops over `apply`) |
| `__call__(image, output)` | `(Image, Image) -> Image` | `Image` | Calls `apply` with optional autocast |

---

### 8.2 TopHatAlgorithm

```python
TopHatAlgorithm(radius: int = 4, device: str = "cpu")
```

White top-hat morphological enhancement. Extracts bright structures smaller than the structuring element (disk of given radius). Ideal first step for microcalcification detection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | `int` | `4` | Disk structuring element radius |

---

### 8.3 KMeansAlgorithm

```python
KMeansAlgorithm(k: int = 2, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42, device: str = "cpu")
```

K-Means clustering segmentation. Assigns pixels to `k` clusters, outputs quantized image and binary mask (brightest cluster).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `2` | Number of clusters |
| `max_iter` | `int` | `100` | Maximum iterations |
| `tol` | `float` | `1e-4` | Convergence tolerance |
| `random_state` | `int` | `42` | Reproducibility seed |

---

### 8.4 FCMAlgorithm

```python
FCMAlgorithm(c: int = 2, m: float = 2.0, max_iter: int = 100, tol: float = 1e-3, random_state: int = 42, device: str = "cpu")
```

Fuzzy C-Means clustering. Produces soft membership values per pixel per cluster.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c` | `int` | `2` | Number of clusters |
| `m` | `float` | `2.0` | Fuzziness exponent (> 1) |
| `max_iter` | `int` | `100` | Maximum iterations |
| `tol` | `float` | `1e-3` | Convergence tolerance |

---

### 8.5 PFCMAlgorithm

```python
PFCMAlgorithm(c: int = 2, m: float = 2.0, eta: float = 2.0, a: float = 1.0, b: float = 4.0, tau: float = 0.04, max_iter: int = 100, tol: float = 1e-3, fcm_max_iter: int = 100, random_state: int = 42, device: str = "cpu")
```

Possibilistic Fuzzy C-Means. Combines FCM membership with possibilistic typicality. Pixels with low typicality in all clusters are flagged as **atypical** -- useful for microcalcification detection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c` | `int` | `2` | Number of clusters |
| `m` | `float` | `2.0` | Fuzziness exponent |
| `eta` | `float` | `2.0` | Typicality exponent |
| `a`, `b` | `float` | `1.0`, `4.0` | Membership vs typicality weights |
| `tau` | `float` | `0.04` | Atypicality threshold |

---

### 8.6 FebdsAlgorithm

```python
FebdsAlgorithm(method: str, device: str = "cpu")
```

Fourier Enhancement and Band-pass Filtering for microcalcification segmentation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | required | Filtering method: `"dog"`, `"log"`, or `"fft"` |

---

### 8.7 BreastMaskAlgorithm

```python
BreastMaskAlgorithm(mask_only: bool = False, device: str = None)
```

Breast region extraction via Otsu + largest connected component.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_only` | `bool` | `False` | `True` = output binary mask; `False` = output masked image |

---

### 8.8 DicomWindowAlgorithm

```python
DicomWindowAlgorithm(window_center: Optional[float] = None, window_width: Optional[float] = None, device: str = None)
```

DICOM Window Center / Window Width intensity mapping. Auto-reads from DICOM header if not specified.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_center` | `Optional[float]` | `None` | Window center (auto from DICOM header) |
| `window_width` | `Optional[float]` | `None` | Window width (auto from DICOM header) |

---

### 8.9 GrailWindowAlgorithm

```python
GrailWindowAlgorithm(n_scales: int = 3, n_orientations: int = 6, delta: int = 300, k_max: int = 3, device: str = None)
```

Automatic intensity windowing via Gabor-filtered mutual information optimization (GRAIL method).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_scales` | `int` | `3` | Number of Gabor scales |
| `n_orientations` | `int` | `6` | Number of Gabor orientations |
| `delta` | `int` | `300` | Search grid spacing |
| `k_max` | `int` | `3` | Optimization iterations |

---

### 8.10 BitDepthNormAlgorithm

```python
BitDepthNormAlgorithm(bits_stored: Optional[int] = None, target_max: float = 255.0, device: str = None)
```

Normalize pixel values based on the DICOM BitsStored tag.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bits_stored` | `Optional[int]` | `None` | Explicit bit depth (auto-detect from header if None) |
| `target_max` | `float` | `255.0` | Output range upper bound |

---

## 9. Datasets

All datasets inherit from `BaseDataset`, which extends `torch.utils.data.Dataset`. They follow a **lazy loading** contract: images are loaded on-the-fly in `__getitem__`.

### 9.1 BaseDataset

**Module:** `medical_image.datasets.base_dataset`

```python
BaseDataset(
    root_dir: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    target_size: Optional[Tuple[int, int]] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | `str` | required | Root directory of dataset files |
| `transform` | `Optional[Callable]` | `None` | Transform applied to image tensors |
| `target_transform` | `Optional[Callable]` | `None` | Transform applied to mask tensors |
| `target_size` | `Optional[Tuple[int,int]]` | `None` | `(H, W)` to resize all outputs |

**Abstract methods (subclasses must implement):**

| Method | Signature | Description |
|--------|-----------|-------------|
| `_build_sample_list()` | `() -> None` | Scan directory, populate `self._samples` |
| `_load_sample(idx)` | `(int) -> Dict[str, Any]` | Load single sample |

**PyTorch interface:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__len__()` | `() -> int` | `int` | Number of samples |
| `__getitem__(idx)` | `(int) -> Dict` | dict | Load sample + apply transforms |

**Output contract (`__getitem__` return):**

```python
{
    "image": torch.Tensor,    # (C, H, W), float32
    "mask": torch.Tensor,     # (1, H, W) -- segmentation datasets
    # OR
    "label": int,             # classification datasets
    "metadata": dict,         # dataset-specific
}
```

**Other methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `download(source, dest, method, pct)` | classmethod | `str` | Download dataset from source |
| `to_coco_json(output_path, description)` | `(...) -> dict` | COCO dict | Export to COCO format |
| `from_coco_json(json_path)` | classmethod | `dict` | Load from COCO format |
| `_get_annotations(idx)` | `(int) -> List[Annotation]` | list | Override point for annotation retrieval |

---

### 9.2 INbreastDataset

**Module:** `medical_image.datasets.inbreast`

```python
INbreastDataset(
    root_dir: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    target_size: Optional[Tuple[int, int]] = None,
    point_radius: int = 3,
)
```

PyTorch Dataset for the INbreast mammography database. Pairs DICOM images with XML annotation files to generate binary segmentation masks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `point_radius` | `int` | `3` | Radius for rendering single-point ROIs as circles |

**Expected directory:**

```
root_dir/
  INbreast Release 1.0/
    AllDICOMs/       # *.dcm
    AllXML/          # *.xml (plist annotations)
    INbreast.csv     # metadata
```

**Sample metadata:**

| Key | Type | Description |
|-----|------|-------------|
| `case_id` | `str` | Numeric ID |
| `laterality` | `str` | From CSV |
| `view` | `str` | From CSV |
| `birads` | `int` | BI-RADS score |
| `file_name` | `str` | DICOM filename |

---

### 9.3 CBISDDSMDataset

**Module:** `medical_image.datasets.cbis_ddsm`

```python
CBISDDSMDataset(
    root_dir: str,
    mode: Literal["full_image", "patch"] = "full_image",
    patch_size: int = 512,
    stride: int = 256,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    target_size: Optional[Tuple[int, int]] = None,
    percentage: Optional[float] = None,
    seed: int = 42,
)
```

PyTorch Dataset for the CBIS-DDSM mammography database. Supports full-image and sliding-window patch modes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"full_image"` | `"full_image"` or `"patch"` |
| `patch_size` | `int` | `512` | Patch side length (patch mode) |
| `stride` | `int` | `256` | Sliding window stride (patch mode) |
| `percentage` | `Optional[float]` | `None` | Use random subset (0 - 1] |
| `seed` | `int` | `42` | Subset sampling seed |

**Extra methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_detailed_sample(idx)` | `(int) -> Dict` | dict | Rich output with per-ROI bounding boxes, crops, masks |
| `collate_fn(batch)` | static | dict | Custom collation for DataLoader |

---

### 9.4 COCO JSON Export / Import

#### Export

```python
dataset.to_coco_json(
    output_path: Optional[str] = None,
    description: str = "Medical Image Dataset",
) -> dict
```

Exports the entire dataset as COCO-format JSON. Each annotation includes a custom `center` field.

**COCO output schema:**

```json
{
    "info": {"description": "...", "version": "1.0", "date_created": "..."},
    "licenses": [],
    "categories": [{"id": 1, "name": "mass", "supercategory": "lesion"}],
    "images": [{"id": 1, "file_name": "img.dcm", "width": 2560, "height": 3328}],
    "annotations": [{
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [[x1, y1, x2, y2, ...]],
        "bbox": [x, y, width, height],
        "area": 10000.0,
        "iscrowd": 0,
        "center": [150.0, 200.0]
    }]
}
```

> **Note:** COCO `bbox` uses `[x, y, width, height]` format. The `center` field is a custom extension.

#### Import

```python
BaseDataset.from_coco_json(json_path: str) -> dict
```

Returns:

```python
{
    "images": List[dict],                       # COCO image entries
    "annotations": Dict[int, List[Annotation]], # image_id -> Annotation list
    "categories": Dict[int, str],               # category_id -> label name
}
```

---

## 10. GPU & Device Management

**Module:** `medical_image.utils.device`

### Functions

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `resolve_device(*images, explicit)` | | `torch.device` | Pick device: explicit > first loaded image > CPU |
| `set_default_precision(p)` | `(Precision)` | -- | Set global precision |
| `get_default_precision()` | | `Precision` | Get global precision |

### Precision Enum

| Value | Torch dtype | Use case |
|-------|-------------|----------|
| `Precision.FULL` | `torch.float32` | Default, maximum accuracy |
| `Precision.HALF` | `torch.float16` | Faster, less memory (NVIDIA GPUs) |
| `Precision.BFLOAT16` | `torch.bfloat16` | Good range, less precision (Ampere+) |

### DeviceContext

```python
with DeviceContext("cuda", verbose=True) as ctx:
    image.to(ctx.device)
    # Falls back to CPU if CUDA unavailable
```

| Method | Returns | Description |
|--------|---------|-------------|
| `memory_stats()` | `dict` | GPU memory usage |

### @gpu_safe Decorator

Catches CUDA OOM errors and retries the function on CPU automatically.

### AsyncGPUPipeline

```python
pipeline = AsyncGPUPipeline("cuda")
results = pipeline.process_images(images, algorithm)
```

Overlaps data transfer and GPU compute using CUDA streams.

### MultiGPUAlgorithm

```python
multi = MultiGPUAlgorithm(KMeansAlgorithm, gpu_ids=[0, 1], k=3)
results = multi.apply_batch(images, outputs)
```

Distributes batch round-robin across multiple GPUs.

---

## 11. Utilities

**Module:** `medical_image.utils.image_utils`

### TensorConverter

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_numpy(image)` | `(Image) -> np.ndarray` | Convert pixel data to CPU numpy |
| `ensure_tensor(image, device, dtype)` | `(Image, ...) -> None` | Move tensor to device/dtype |

### ImageExporter

| Method | Signature | Description |
|--------|-----------|-------------|
| `save_as(image, format="PNG")` | `(Image, str) -> None` | Save to PNG/JPG/TIFF (auto uint8) |

### ImageVisualizer

| Method | Signature | Description |
|--------|-----------|-------------|
| `show(image, cmap, title)` | `(Image, str, str) -> None` | Display with matplotlib |
| `compare(before, after, titles)` | `(Image, Image, Tuple) -> None` | Side-by-side comparison |

### MathematicalOperations

| Method | Signature | Description |
|--------|-----------|-------------|
| `abs(image, output)` | `(Image, Image) -> Image` | Element-wise absolute value |
| `euclidean_distance_sq(Z, V)` | `(Tensor, Tensor) -> Tensor` | Squared Euclidean distances |
| `normalize_12bit(image, output)` | `(Image, Image) -> Image` | Normalize 12-bit DICOM to [0, 1] |

---

## 12. Usage Examples with Visualizations

### 12.1 Load, process, and display a mammogram

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.filters import Filters
from medical_image.utils.image_utils import ImageVisualizer

# Load
image = DicomImage.from_file("mammogram.dcm")
image.load()

# Process
output = InMemoryImage(source_image=image)
Filters.gaussian_filter(image, output, sigma=2.0)

# Display
ImageVisualizer.compare(image, output, titles=("Original", "Gaussian Blurred"))
```

### 12.2 Full microcalcification detection pipeline

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.utils.image_utils import ImageVisualizer

image = DicomImage.from_file("mammogram.dcm")
image.load()

# Step 1: Top-hat enhancement
enhanced = InMemoryImage(source_image=image)
tophat = TopHatAlgorithm(radius=5, device="cpu")
tophat.apply(image, enhanced)

# Step 2: K-Means segmentation
segmented = InMemoryImage(source_image=enhanced)
kmeans = KMeansAlgorithm(k=3, device="cpu")
kmeans.apply(enhanced, segmented)

# Visualize the pipeline
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(image.pixel_data.cpu().numpy(), cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(enhanced.pixel_data.cpu().numpy(), cmap="gray")
axes[1].set_title("Top-Hat Enhanced")
axes[2].imshow(segmented.pixel_data.cpu().numpy(), cmap="gray")
axes[2].set_title("K-Means Segmented")
plt.tight_layout()
plt.savefig("pipeline_result.png", dpi=150)
plt.show()
```

### 12.3 Patch-based processing for large images

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.data.patch import PatchGrid
from medical_image.algorithms.top_hat import TopHatAlgorithm

image = DicomImage.from_file("large_mammogram.dcm")
image.load()

# Split into 512x512 patches
grid = PatchGrid(image, patch_size=(512, 512))
print(f"Split into {len(grid.patches)} patches")

# Process each patch
tophat = TopHatAlgorithm(radius=4)
for patch in grid.patches:
    patch_img = patch.load()
    out = InMemoryImage(source_image=patch_img)
    tophat.apply(patch_img, out)
    patch.pixel_data = out.pixel_data

# Reassemble
result_tensor = grid.reconstruct()
print(f"Reconstructed: {result_tensor.shape}")
```

### 12.4 Annotate images and visualize

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from medical_image.data.in_memory_image import InMemoryImage
from medical_image.utils.annotation import Annotation, GeometryType

# Create image with annotations
image = InMemoryImage.from_array(np.random.rand(512, 512).astype(np.float32))

image.add_annotation(
    Annotation(GeometryType.RECTANGLE, [80, 100, 180, 200], "mass",
               metadata={"birads": 4})
)
image.add_annotation(
    Annotation(GeometryType.POLYGON,
               [(300, 200), (330, 190), (350, 220), (340, 250), (310, 245)],
               "microcalcification")
)
image.add_annotation(
    Annotation(GeometryType.ELLIPSE, [400.0, 80.0, 30.0, 20.0], "lesion")
)

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image.pixel_data.numpy(), cmap="gray")

colors = {"mass": "red", "microcalcification": "cyan", "lesion": "yellow"}
for ann in image.annotations:
    bbox = ann.get_bounding_box()
    x, y, x2, y2 = bbox
    color = colors.get(ann.label, "white")

    rect = mpatches.Rectangle((x, y), x2 - x, y2 - y,
                               linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    cx, cy = ann.center
    ax.plot(cx, cy, "o", color=color, markersize=6)
    ax.text(x, y - 5, f"{ann.label}", color=color, fontsize=9,
            fontweight="bold", backgroundcolor="black")

ax.set_title("Annotated Medical Image")
plt.tight_layout()
plt.savefig("annotated_image.png", dpi=150)
plt.show()
```

### 12.5 ROI extraction with padding

```python
ann = Annotation(GeometryType.RECTANGLE, [100, 80, 200, 180], "mass")

# Tight bounding box
roi = ann.get_roi()
print(roi)  # {"type": "bbox", "coordinates": [100, 80, 200, 180]}

# Padded
roi = ann.get_roi(padding=30)
print(roi)  # {"type": "bbox", "coordinates": [70, 50, 230, 210]}

# Ellipse ROI
roi = ann.get_roi(padding=10, roi_type="ellipse")
print(roi)
# {"type": "ellipse", "coordinates": {"center": (150.0, 130.0), "radii": (60.0, 60.0)}}

# Clamped to image bounds
roi = ann.get_roi(padding=200, roi_type="bbox", image_shape=(512, 512))
print(roi)  # coordinates clamped to [0, 0, 400, 380]
```

### 12.6 Dataset loading with PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

dataset = CBISDDSMDataset(
    root_dir="/data/cbis-ddsm",
    mode="patch",
    patch_size=256,
    stride=128,
    target_size=(256, 256),
)

loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

for batch in loader:
    images = batch["image"]      # (16, 1, 256, 256)
    masks = batch["mask"]        # (16, 1, 256, 256)
    metadata = batch["metadata"]
    print(f"Batch: {images.shape}")
    break
```

### 12.7 COCO JSON export and visualization

```python
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Export dataset to COCO
dataset = CBISDDSMDataset(root_dir="/data/cbis-ddsm")
coco = dataset.to_coco_json(output_path="cbis_coco.json",
                             description="CBIS-DDSM Calcification Dataset")
print(f"Exported: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

# Load back and inspect
from medical_image.datasets.base_dataset import BaseDataset
result = BaseDataset.from_coco_json("cbis_coco.json")

# Plot annotations for first image
img_entry = coco["images"][0]
img_anns = [a for a in coco["annotations"] if a["image_id"] == img_entry["id"]]
cat_names = {c["id"]: c["name"] for c in coco["categories"]}

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, img_entry["width"])
ax.set_ylim(img_entry["height"], 0)

for ann in img_anns:
    x, y, w, h = ann["bbox"]
    cx, cy = ann["center"]
    label = cat_names[ann["category_id"]]

    rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                               edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.plot(cx, cy, "r+", markersize=10, markeredgewidth=2)
    ax.text(x, y - 3, f"{label}", color="red", fontsize=8)

ax.set_title(f"{img_entry['file_name']} -- {len(img_anns)} annotations")
plt.tight_layout()
plt.savefig("coco_annotations.png", dpi=150)
plt.show()
```

### 12.8 JSON round-trip for a single annotated image

```python
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.data.image import image_from_json
from medical_image.utils.annotation import Annotation, GeometryType

# Build
image = InMemoryImage(width=2560, height=3328)
image.add_annotation(
    Annotation(GeometryType.RECTANGLE, [500, 600, 700, 800], "mass",
               metadata={"birads": 4, "pathology": "malignant"})
)

# Save
image.to_json(file_path="annotated_mammogram.json")

# Load (factory auto-detects subclass)
restored = image_from_json("annotated_mammogram.json")
print(type(restored).__name__)           # InMemoryImage
print(len(restored.annotations))         # 1
print(restored.annotations[0].center)    # (600.0, 700.0)
print(restored.annotations[0].metadata)  # {"birads": 4, "pathology": "malignant"}
```

### 12.9 GPU-accelerated processing

```python
from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.filters import Filters
from medical_image.utils.device import DeviceContext

image = DicomImage.from_file("mammogram.dcm")
image.load()

# Automatic GPU with OOM fallback
with DeviceContext("cuda", verbose=True) as ctx:
    image.to(ctx.device)
    output = InMemoryImage(source_image=image)
    Filters.gaussian_filter(image, output, sigma=2.0, device=ctx.device)
    print(f"Processed on: {ctx.device}")
    print(f"Memory: {ctx.memory_stats()}")
```

### 12.10 Compare algorithms side-by-side

```python
import matplotlib.pyplot as plt
from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.algorithms.pfcm import PFCMAlgorithm

image = DicomImage.from_file("mammogram.dcm")
image.load()

# Enhance
enhanced = InMemoryImage(source_image=image)
TopHatAlgorithm(radius=5).apply(image, enhanced)

# Segment with three different algorithms
results = {}
for name, algo in [
    ("K-Means (k=3)", KMeansAlgorithm(k=3)),
    ("FCM (c=3)", FCMAlgorithm(c=3)),
    ("PFCM (c=2, tau=0.04)", PFCMAlgorithm(c=2, tau=0.04)),
]:
    out = InMemoryImage(source_image=enhanced)
    algo.apply(enhanced, out)
    results[name] = out

# Plot
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(enhanced.pixel_data.cpu().numpy(), cmap="gray")
axes[0].set_title("Top-Hat Enhanced")

for ax, (name, img) in zip(axes[1:], results.items()):
    ax.imshow(img.pixel_data.cpu().numpy(), cmap="gray")
    ax.set_title(name)

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("algorithm_comparison.png", dpi=150)
plt.show()
```

---

## 13. Running Tests

```bash
# All tests
uv run pytest medical_image/tests/ -v

# Specific test suites
uv run pytest medical_image/tests/test_dicom.py -v         # DICOM, filters, patches
uv run pytest medical_image/tests/test_mc_algorithms.py -v  # Algorithms, ROI, pipelines
uv run pytest medical_image/tests/test_mammography.py -v    # Mammography preprocessing
uv run pytest medical_image/tests/test_gpu.py -v            # GPU, device, precision
uv run pytest medical_image/tests/test_annotation.py -v     # Annotation class
uv run pytest medical_image/tests/test_image_json.py -v     # Image JSON serialization
uv run pytest medical_image/tests/test_coco_export.py -v    # COCO export/import

# Code formatting
uv run black --check .
```

---

**Last Updated:** 2026-04-15
**Version:** 0.5.0