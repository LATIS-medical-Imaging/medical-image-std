# Architecture Documentation

## Overview

This document provides detailed architectural information about the Medical Image Standard Library, including design patterns, class relationships, and workflow diagrams.

---

## Design Philosophy

### Core Principles

1. **Abstraction-First Design**
   - Define interfaces through abstract base classes
   - Concrete implementations extend abstractions
   - Ensures consistency across different image types

2. **Lazy Loading Pattern**
   - Object instantiation does not imply data loading
   - `__init__()`: Store metadata only
   - `load()`: Load actual pixel data
   - Benefits: Memory efficiency, faster initialization

3. **Device-Aware Processing**
   - All processing methods accept `device=None` by default
   - Device is resolved at call time via `resolve_device()`
   - Priority: explicit parameter > image device > CPU fallback

4. **Composition over Inheritance**
   - Algorithms compose processing methods
   - Lambda functions define processing pipelines
   - Flexible and extensible

5. **Separation of Concerns**
   - **Data**: Image structures and representations
   - **Process**: Processing operations
   - **Algorithms**: High-level workflows
   - **Utils**: Device management, logging, annotations

---

## Package Architecture

### Data Package (`medical_image.data`)

**Purpose**: Define data structures and abstractions for medical images

```mermaid
graph TB
    subgraph "Abstract Layer"
        Image[Image ABC]
    end

    subgraph "Concrete Implementations"
        DicomImage[DicomImage]
        PNGImage[PNGImage]
        InMemoryImage[InMemoryImage]
    end

    subgraph "Supporting Classes"
        Patch[Patch]
        PatchGrid[PatchGrid]
        ROI[RegionOfInterest]
        Annotation[Annotation]
    end

    Image --> DicomImage
    Image --> PNGImage
    Image --> InMemoryImage

    PatchGrid --> Patch
    Image -.references.- Patch
    Image -.operates on.- ROI
    Image -.labeled by.- Annotation

    style Image fill:#FFE4B5
    style InMemoryImage fill:#B0E0E6
```

### Datasets Package (`medical_image.datasets`)

**Purpose**: Production-grade PyTorch Dataset classes with lazy loading, automatic file pairing, and standardized output dictionaries.

```mermaid
graph TB
    subgraph "Abstract Layer"
        BaseDataset[BaseDataset ABC]
    end

    subgraph "Concrete Datasets"
        INbreast[INbreastDataset]
        CustomINbreast[CustomINbreastDataset]
        CBISDDSM[CBISDDSMDataset]
    end

    subgraph "Supporting Modules"
        Pairing[pairing.py<br/>File matching]
        MaskUtils[mask_utils.py<br/>Mask generation]
        Downloader[downloader.py<br/>Dataset download]
    end

    subgraph "Data Layer"
        DicomImage2[DicomImage]
    end

    BaseDataset --> INbreast
    BaseDataset --> CustomINbreast
    BaseDataset --> CBISDDSM

    INbreast --> Pairing
    INbreast --> MaskUtils
    INbreast --> DicomImage2
    CustomINbreast --> Pairing
    CustomINbreast --> MaskUtils
    CBISDDSM --> Pairing
    CBISDDSM --> MaskUtils
    CBISDDSM --> DicomImage2
    BaseDataset --> Downloader

    style BaseDataset fill:#FFE4B5
    style INbreast fill:#B0E0E6
    style CustomINbreast fill:#B0E0E6
    style CBISDDSM fill:#B0E0E6
```

**Key patterns:**
- **Template Method**: `BaseDataset.__getitem__` calls abstract `_load_sample`, then applies transforms and resizing
- **Lazy Loading**: Images loaded from disk on each access, never pre-loaded
- **Data Classes**: `INbreastSample`, `CBISDDSMSample`, etc. as typed sample containers
- **Strategy**: `CBISDDSMDataset` dispatches between `_load_full_image` and `_load_patch` based on mode

#### Image Class Hierarchy

```mermaid
classDiagram
    class Image {
        <<abstract>>
        -file_path: Optional~str~
        -_width: Optional~int~
        -_height: Optional~int~
        -_device: torch.device
        -pixel_data: Optional~torch.Tensor~

        +width property
        +height property
        +device property

        +__init__(file_path, array, width, height, source_image)
        +load()* abstract
        +save()* abstract
        +display_info()
        +to(device) Image
        +clone() Image
        +pin_memory() Image
        +ensure_loaded() Image
        +from_file(file_path)$ Image
        +from_array(array)$ Image
        +from_image(other)$ Image
        +empty(width, height)$ Image
    }

    class DicomImage {
        -dicom_data: pydicom.Dataset

        +__init__(file_path, ...)
        +load() override
        +save() override
    }

    class PNGImage {
        +__init__(file_path, ...)
        +load() override
        +save() override
    }

    class InMemoryImage {
        +__init__(file_path, array, width, height, source_image)
        +load() no-op
        +save() no-op
    }

    Image <|-- DicomImage
    Image <|-- PNGImage
    Image <|-- InMemoryImage

    note for Image "Lazy Loading:\n__init__: Store path + metadata\nload(): Load pixel data\ndevice: derived from pixel_data"
    note for InMemoryImage "No file I/O.\nUsed for intermediate results\nand in-memory construction."
```

#### Patch System Architecture

```mermaid
classDiagram
    class Patch {
        +parent: Image
        +row_idx: int
        +col_idx: int
        +x: int
        +y: int
        +pixel_data: torch.Tensor
        +is_padded: bool
        +height: int
        +width: int

        +grid_id() Tuple
        +pixel_position() Tuple
        +to_numpy() ndarray
    }

    class PatchGrid {
        +parent: Image
        +patch_h: int
        +patch_w: int
        +patches: List~Patch~
        +grid: List~List~Patch~~
        +pad_bottom: int
        +pad_right: int

        +__init__(parent_image, patch_size)
        -_split() private
        +reconstruct() Tensor
    }

    class Image {
        <<abstract>>
        +pixel_data: Tensor
    }

    PatchGrid "1" *-- "n" Patch : manages
    Patch --> Image : references parent
    PatchGrid --> Image : operates on

    note for PatchGrid "_split() is called\nautomatically in __init__"
```

---

### Process Package (`medical_image.process`)

**Purpose**: Provide static processing methods organized by category

```mermaid
graph LR
    subgraph "Process Package"
        Filters[Filters<br/>Static Methods]
        Threshold[Threshold<br/>Static Methods]
        Metrics[Metrics<br/>Static Methods]
        Morphology[Morphology<br/>Static Methods]
        Frequency[Frequency<br/>Static Methods]
    end

    Image[Image] --> Filters
    Image --> Threshold
    Image --> Metrics
    Image --> Morphology
    Image --> Frequency

    Filters --> Output[Processed Image]
    Threshold --> Output
    Metrics --> Output
    Morphology --> Output
    Frequency --> Output

    style Filters fill:#B0E0E6
    style Threshold fill:#B0E0E6
    style Metrics fill:#B0E0E6
    style Morphology fill:#B0E0E6
    style Frequency fill:#B0E0E6
```

#### Processing Method Pattern

All processing methods follow this pattern:

```python
@staticmethod
@requires_loaded
def method_name(image: Image, output: Image, param1, param2, ..., device=None):
    """
    Process image and store result in output.

    Args:
        image: Input image (read-only)
        output: Output image (modified)
        param1, param2: Processing parameters
        device: Target device (None = infer from image)
    """
    # 1. Resolve device
    device = resolve_device(image, explicit=device)

    # 2. Access input data on target device
    img = image.pixel_data.to(device).float()

    # 3. Apply processing
    result = process(img, param1, param2)

    # 4. Store in output
    output.pixel_data = result
```

---

### Algorithms Package (`medical_image.algorithms`)

**Purpose**: Define high-level processing workflows

```mermaid
classDiagram
    class Algorithm {
        <<abstract>>
        +device: str
        +precision: Precision
        +__init__(device=None, precision=Precision.FULL)
        +apply(image: Image, output: Image)* abstract
        +__call__(image: Image, output: Image) Image
        +apply_batch(images, outputs) List~Image~
    }

    class FebdsAlgorithm {
        +method: str
        +__init__(method: str, device: str)
        +apply(image: Image, output: Image) Image
    }

    class KMeansAlgorithm {
        +k: int
        +max_iter: int
        +tol: float
        +random_state: int
        +__init__(k: int, max_iter: int, tol: float, random_state: int, device: str)
        +apply(image: Image, output: Image) Image
    }

    class FCMAlgorithm {
        +c: int
        +m: float
        +max_iter: int
        +tol: float
        +__init__(c: int, m: float, max_iter: int, tol: float, random_state: int, device: str)
        +apply(image: Image, output: Image) Image
    }

    class PFCMAlgorithm {
        +c: int
        +m: float
        +eta: float
        +a: float
        +b: float
        +tau: float
        +__init__(c: int, m: float, eta: float, a: float, b: float, tau: float, max_iter: int, tol: float, device: str)
        +apply(image: Image, output: Image) Image
    }

    class TopHatAlgorithm {
        +radius: int
        +__init__(radius: int, device: str)
        +apply(image: Image, output: Image) Image
    }

    Algorithm <|-- FebdsAlgorithm
    Algorithm <|-- KMeansAlgorithm
    Algorithm <|-- FCMAlgorithm
    Algorithm <|-- PFCMAlgorithm
    Algorithm <|-- TopHatAlgorithm

    note for Algorithm "Template Method Pattern:\n__init__: Define steps as lambdas\napply: Execute sequence\n__call__: autocast wrapper"
```

#### Algorithm Pattern

```python
class MyAlgorithm(Algorithm):
    def __init__(self, param1, param2, device="cpu"):
        """Define processing steps as lambda functions."""
        super().__init__(device=device)

        # Step 1: Preprocessing
        self.preprocess = lambda img, out: Filters.gaussian_filter(
            image=img, output=out, sigma=param1, device=self.device
        )

        # Step 2: Enhancement
        self.enhance = lambda img, out: Filters.gamma_correction(
            image=img, output=out, gamma=param2, device=self.device
        )

        # Step 3: Segmentation
        self.segment = lambda img, out: Threshold.otsu_threshold(
            image=img, output=out, device=self.device
        )

    def apply(self, image: Image, output: Image) -> Image:
        """Execute the sequence of processing steps."""
        self.preprocess(image, output)
        self.enhance(output, output)
        self.segment(output, output)
        return output
```

The `__call__` method on `Algorithm` wraps `apply()` with `torch.cuda.amp.autocast` when `precision` is not `Precision.FULL` and the device is not CPU:

```python
def __call__(self, image, output):
    if self.precision != Precision.FULL and self.device != "cpu":
        with torch.cuda.amp.autocast(dtype=self.precision.value):
            self.apply(image, output)
    else:
        self.apply(image, output)
    return output
```

---

## Device Flow Architecture

### resolve_device Priority

```mermaid
flowchart TD
    Start([resolve_device called]) --> ExplicitCheck{explicit param provided?}

    ExplicitCheck -->|Yes| UseExplicit[Use explicit device]
    ExplicitCheck -->|No| ImageCheck{image.pixel_data loaded?}

    ImageCheck -->|Yes| UseImage[Use image.pixel_data.device]
    ImageCheck -->|No| UseCPU[Fallback to CPU]

    UseExplicit --> Done([Return torch.device])
    UseImage --> Done
    UseCPU --> Done

    style UseExplicit fill:#90EE90
    style UseImage fill:#FFE4B5
    style UseCPU fill:#B0E0E6
```

### Device Management Components

```mermaid
graph TB
    subgraph "Device Resolution"
        resolve_device["resolve_device(*images, explicit=None)"]
    end

    subgraph "Context Management"
        DeviceContext["DeviceContext(device, fallback, verbose)"]
    end

    subgraph "Safety"
        gpu_safe["@gpu_safe decorator"]
    end

    subgraph "Precision Control"
        PrecisionEnum["Precision enum: FULL / HALF / BFLOAT16"]
    end

    resolve_device --> ProcessMethods[Processing Methods]
    DeviceContext --> UserCode[User Pipeline Code]
    gpu_safe --> ProcessMethods
    PrecisionEnum --> Algorithm[Algorithm.__call__]

    style resolve_device fill:#90EE90
    style DeviceContext fill:#FFE4B5
    style gpu_safe fill:#DDA0DD
    style PrecisionEnum fill:#B0E0E6
```

All processing methods (`Filters`, `Threshold`, `Morphology`, `Frequency`) accept `device=None` and call `resolve_device(image, explicit=device)` internally. This means the caller can either:

- Pass an explicit device: `Filters.gaussian_filter(image, output, sigma=2.0, device="cuda:0")`
- Let it infer from the image: `Filters.gaussian_filter(image, output, sigma=2.0)`
- Get CPU fallback if nothing else is available

---

## Workflow Diagrams

### Complete Image Processing Workflow

```mermaid
sequenceDiagram
    actor User
    participant DicomImage
    participant Filters
    participant Threshold
    participant Output

    Note over User,Output: 1. Image Creation (No Loading)
    User->>DicomImage: image = DicomImage("path.dcm")
    Note over DicomImage: Only file_path stored<br/>pixel_data = None

    Note over User,Output: 2. Lazy Loading
    User->>DicomImage: image.load()
    DicomImage->>DicomImage: Validate file exists
    DicomImage->>DicomImage: Read DICOM metadata
    DicomImage->>DicomImage: Load pixel_array
    DicomImage->>DicomImage: Convert to torch.Tensor
    DicomImage->>DicomImage: Move to device (GPU/CPU)
    Note over DicomImage: pixel_data loaded

    Note over User,Output: 3. Clone for Output
    User->>DicomImage: output = image.clone()
    Note over Output: Lightweight clone:<br/>copies pixel_data tensor,<br/>skips heavy DICOM objects

    Note over User,Output: 4. Processing Pipeline
    User->>Filters: gaussian_filter(image, output, sigma=2.0)
    Filters->>Filters: resolve_device(image, explicit=None)
    Filters->>DicomImage: Access pixel_data
    Filters->>Filters: Apply convolution
    Filters->>Output: Set pixel_data

    User->>Threshold: otsu_threshold(output, output)
    Threshold->>Output: Access pixel_data
    Threshold->>Threshold: Calculate threshold
    Threshold->>Output: Update pixel_data

    Note over User,Output: 5. Save Results
    User->>Output: output.save()
```

### PatchGrid Detailed Workflow

```mermaid
sequenceDiagram
    actor User
    participant Image
    participant PatchGrid
    participant Patch

    User->>Image: image = DicomImage("large.dcm")
    User->>Image: image.load()
    Note over Image: pixel_data: [C, H, W]

    User->>PatchGrid: grid = PatchGrid(image, (256, 256))
    activate PatchGrid

    PatchGrid->>PatchGrid: __init__()
    PatchGrid->>PatchGrid: _split()

    Note over PatchGrid: Calculate padding
    PatchGrid->>PatchGrid: pad_bottom = patch_h - (H % patch_h)
    PatchGrid->>PatchGrid: pad_right = patch_w - (W % patch_w)

    alt Padding needed
        PatchGrid->>Image: Get pixel_data
        PatchGrid->>PatchGrid: Apply torch.nn.functional.pad()
        Note over PatchGrid: Padded image: [C, H', W']
    end

    Note over PatchGrid: Calculate grid dimensions
    PatchGrid->>PatchGrid: num_rows = H' // patch_h
    PatchGrid->>PatchGrid: num_cols = W' // patch_w

    loop For each row r
        loop For each column c
            PatchGrid->>PatchGrid: Calculate x, y coordinates
            PatchGrid->>PatchGrid: Extract patch tensor
            PatchGrid->>Patch: Create Patch(parent, r, c, x, y, tensor)
            Patch-->>PatchGrid: patch object
            PatchGrid->>PatchGrid: patches.append(patch)
            PatchGrid->>PatchGrid: grid[r][c] = patch
        end
    end

    deactivate PatchGrid
    Note over PatchGrid: Grid ready with all patches

    User->>PatchGrid: for patch in grid.patches
    PatchGrid-->>User: Iterate patches

    User->>User: Process each patch

    User->>PatchGrid: reconstructed = grid.reconstruct()
    PatchGrid->>PatchGrid: Concatenate patches
    PatchGrid->>PatchGrid: Remove padding
    PatchGrid-->>User: Full image tensor
```

### FEBDS Algorithm Execution Flow

```mermaid
flowchart TD
    Start([User calls algo image, output]) --> Init{Check method}

    Init -->|method == 'dog'| DoG[Difference of Gaussian<br/>low_sigma=1.7, high_sigma=2.0]
    Init -->|method == 'log'| LoG[Laplacian of Gaussian<br/>sigma=2.0]
    Init -->|method == 'fft'| FFT[FFT + Butterworth<br/>+ Inverse FFT]

    DoG --> Abs
    LoG --> Abs
    FFT --> Abs

    Abs[Absolute Value] --> Median[Median Filter<br/>size=5]

    Median --> Gamma[Gamma Correction<br/>gamma=1.25]

    Gamma --> ThresholdCheck{Check method}

    ThresholdCheck -->|method == 'fft'| Binarize[Variance-based<br/>Binarization alpha=1]
    ThresholdCheck -->|method != 'fft'| Otsu[Otsu's<br/>Threshold]

    Binarize --> Closing
    Otsu --> Closing

    Closing[Morphological<br/>Closing] --> Fill[Region<br/>Filling]

    Fill --> End([Result in output])

    style DoG fill:#FFE4B5
    style LoG fill:#FFE4B5
    style FFT fill:#FFE4B5
    style Abs fill:#E0E0E0
    style Median fill:#B0E0E6
    style Gamma fill:#B0E0E6
    style Binarize fill:#98FB98
    style Otsu fill:#98FB98
    style Closing fill:#DDA0DD
    style Fill fill:#DDA0DD
```

---

## Design Patterns

### 1. Abstract Factory Pattern

**Used in**: Image class hierarchy

```python
# Abstract factory
class Image(ABC):
    @abstractmethod
    def load(self): pass

    @abstractmethod
    def save(self): pass

# Concrete implementations
class DicomImage(Image):
    def load(self):
        # DICOM-specific loading
        pass

class PNGImage(Image):
    def load(self):
        # PNG-specific loading
        pass

class InMemoryImage(Image):
    def load(self):
        pass  # no-op: data already in memory

    def save(self):
        pass  # no-op: no backing file
```

### 2. Template Method Pattern

**Used in**: Algorithm class

```python
class Algorithm(ABC):
    def __init__(self, device=None, precision=Precision.FULL):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision

    @abstractmethod
    def apply(self, image, output):
        pass

    def __call__(self, image, output):
        # Wraps apply() with optional autocast
        ...

class FebdsAlgorithm(Algorithm):
    def __init__(self, method, device="cpu"):
        super().__init__(device=device)
        # Define steps as lambdas
        self.dog = lambda img, out: Filters.difference_of_gaussian(...)
        self.median = lambda img, out: Filters.median_filter(...)

    def apply(self, image, output):
        # Execute template
        self.dog(image, output)
        self.median(output, output)
        ...
```

### 3. Strategy / Template Method Pattern

**Used in**: Algorithms and FEBDS method selection

The `Algorithm` base class dictates a common interface via `apply()`. Subclasses like `KMeansAlgorithm` instantiate the steps of the process inside `__init__()`. Some algorithms (like `FebdsAlgorithm`) leverage internal strategy switching to select operations.

```python
class FebdsAlgorithm:
    def apply(self, image, output):
        # Strategy selection
        if self.method == "dog":
            self.dog(image, output)
        elif self.method == "log":
            self.log(image, output)
        elif self.method == "fft":
            self.fft(image, output)

class KMeansAlgorithm(Algorithm):
    def __init__(self, k=2, max_iter=100, tol=1e-4, random_state=42, device="cpu"):
        super().__init__(device=device)
        # Define steps using templates
        self.compute_distances = lambda Z, V: ...

    def apply(self, image, output):
        # Execute template strictly in order
        ...
```

### 4. Lazy Initialization Pattern

**Used in**: Image loading

```python
class Image:
    def __init__(self, file_path=None, array=None, ...):
        self.file_path = file_path
        self.pixel_data = None  # Not loaded yet
        self._device = torch.device("cpu")

    def load(self):
        # Load only when called
        self.pixel_data = load_from_file(self.file_path)

    def ensure_loaded(self):
        """Guard: raise if pixel_data is None."""
        if self.pixel_data is None:
            raise DicomDataNotLoadedError("Call .load() first")
        return self
```

---

## Memory Management

### Lazy Loading Benefits

```mermaid
graph LR
    subgraph "Without Lazy Loading"
        A1[Create Object] --> A2[Load Data<br/>Memory: HIGH]
        A2 --> A3[Process]
    end

    subgraph "With Lazy Loading"
        B1[Create Object<br/>Memory: LOW] --> B2[Load When Needed]
        B2 --> B3[Process]
        B3 --> B4[Clear Memory]
    end

    style A2 fill:#FFB6C6
    style B1 fill:#90EE90
    style B4 fill:#90EE90
```

### Memory Lifecycle

```python
# 1. Object creation - minimal memory
image = DicomImage("large_file.dcm")  # Only path stored

# 2. Load data - memory allocated
image.load()  # pixel_data loaded to device

# 3. Clone for output (lightweight, no deep copy)
output = image.clone()  # clones pixel_data tensor, skips heavy DICOM objects

# 4. Process
Filters.gaussian_filter(image, output, sigma=2.0)

# 5. Clear memory
del image
torch.cuda.empty_cache()  # Free GPU memory
```

### DeviceContext Manager

`DeviceContext` provides GPU-aware processing with automatic memory management:

```python
from medical_image import DeviceContext

with DeviceContext("cuda", verbose=True) as ctx:
    image.to(ctx.device)
    output = image.clone()
    algo = FebdsAlgorithm(method="dog", device=str(ctx.device))
    algo(image, output)
# GPU cache automatically cleared on exit
```

Features:
- Clears GPU cache on entry and exit
- Provides `memory_stats()` for tracking GPU usage
- Automatic CPU fallback when CUDA is unavailable
- Suppresses OOM exceptions and falls back to CPU

### pin_memory for Faster GPU Transfers

```python
image.load()
image.pin_memory()  # Pin to page-locked memory
image.to("cuda")    # Non-blocking transfer possible
```

### gpu_safe Decorator

The `@gpu_safe` decorator catches CUDA OOM errors and retries the operation on CPU:

```python
from medical_image import gpu_safe

@gpu_safe
def my_processing(image, output, device=None):
    Filters.gaussian_filter(image, output, sigma=2.0, device=device)
    return output
```

---

## Extension Points

### Adding New Image Format

```python
from medical_image.data.image import Image

class TIFFImage(Image):
    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)

    def load(self):
        from PIL import Image as PILImage
        img = PILImage.open(self.file_path)
        self.pixel_data = torch.from_numpy(np.array(img)).float()

    def save(self):
        # Implement TIFF saving
        pass
```

### Adding New Processing Method

```python
class Filters:
    @staticmethod
    @requires_loaded
    def bilateral_filter(image: Image, output: Image,
                        sigma_color: float, sigma_space: float,
                        device=None):
        """Add new filter to existing class."""
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()
        # Implementation
        output.pixel_data = result
```

### Adding New Algorithm

```python
from medical_image.algorithms.algorithm import Algorithm
from medical_image.utils.device import Precision

class MyCustomAlgorithm(Algorithm):
    def __init__(self, param1, param2, device="cpu", precision=Precision.FULL):
        super().__init__(device=device, precision=precision)
        # Define processing steps as lambdas
        self.step1 = lambda img, out: Filters.gaussian_filter(
            image=img, output=out, sigma=param1, device=self.device
        )
        self.step2 = lambda img, out: Threshold.otsu_threshold(
            image=img, output=out, device=self.device
        )

    def apply(self, image: Image, output: Image) -> Image:
        # Execute sequence
        self.step1(image, output)
        self.step2(output, output)
        return output
```

---

## Performance Considerations

### GPU Acceleration

```mermaid
graph TB
    Image[Image Object] --> Check{CUDA Available?}
    Check -->|Yes| GPU[device = 'cuda']
    Check -->|No| CPU[device = 'cpu']

    GPU --> TensorGPU[pixel_data on GPU]
    CPU --> TensorCPU[pixel_data on CPU]

    TensorGPU --> FastProcess[Fast Processing]
    TensorCPU --> SlowProcess[Slower Processing]

    style GPU fill:#90EE90
    style FastProcess fill:#90EE90
```

### Mixed Precision

The `Algorithm` base class supports mixed precision via the `Precision` enum:

```python
from medical_image.utils.device import Precision

algo = FebdsAlgorithm(method="dog", device="cuda")
algo.precision = Precision.HALF  # Use float16

# __call__ wraps apply() with autocast
algo(image, output)  # Runs under torch.cuda.amp.autocast
```

Available precision modes: `Precision.FULL` (float32), `Precision.HALF` (float16), `Precision.BFLOAT16` (bfloat16).

### Batch Processing

```python
algo = KMeansAlgorithm(k=3, device="cuda")
outputs = algo.apply_batch(images, output_images)
```

The default `apply_batch()` loops over `apply()`. Subclasses can override for truly batched GPU processing.

### Patch-based Processing for Large Images

```python
# For very large images
large_image = DicomImage("4096x4096.dcm")
large_image.load()

# Process in patches to manage memory
patch_grid = PatchGrid(large_image, patch_size=(512, 512))

for patch in patch_grid.patches:
    # Process each patch independently
    process_patch(patch.pixel_data)

# Reconstruct
result = patch_grid.reconstruct()
```

---

## Testing Architecture

### Test Organization

```
medical_image/tests/
├── __init__.py
├── test_dicom.py              # DICOM loading, filters vs scikit-image, morphology vs scipy, FEBDS pipeline, patches
├── test_mc_algorithms.py      # KMeans, FCM, PFCM, TopHat, full pipeline integration, ROI extraction
├── test_gpu.py                # Device inference, DeviceContext, Precision, pin_memory, all modules on CPU+CUDA, batch ops
└── dummy_data/                # Test data
    └── sample.dcm
```

### Test Pattern

```python
class TestFeature:
    """Test suite for a specific feature."""

    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = process(input_data)

        # Assert
        assert result is not None
        assert result.shape == expected_shape

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        with pytest.raises(ValueError):
            process(invalid_input)
```

---

## Summary

The Medical Image Standard Library follows a clean, extensible architecture:

- **Abstract base classes** define standard interfaces (`Image`, `Algorithm`)
- **Lazy loading** optimizes memory usage
- **Device-aware processing** via `resolve_device()` with explicit > image > CPU priority
- **Lambda composition** enables flexible algorithm pipelines
- **Mixed precision** support through `Precision` enum and `autocast`
- **Memory management** with `DeviceContext`, `pin_memory()`, `clone()`, and `@gpu_safe`
- **Patch system** handles large images efficiently
- **GPU acceleration** with automatic fallback

This architecture makes it easy to:
- Add new image formats (extend `Image` ABC)
- Implement new processing methods (add static methods with `device=None` + `resolve_device()`)
- Create custom algorithms (extend `Algorithm`, define steps as lambdas)
- Maintain and test code