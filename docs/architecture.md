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
   - Object instantiation ≠ data loading
   - `__init__()`: Store metadata only
   - `load()`: Load actual pixel data
   - Benefits: Memory efficiency, faster initialization

3. **Static Processing Methods**
   - Processing operations are stateless
   - Organized in classes for namespace management
   - Easy to test and reuse
   - No side effects

4. **Composition over Inheritance**
   - Algorithms compose processing methods
   - Lambda functions define processing pipelines
   - Flexible and extensible

5. **Separation of Concerns**
   - **Data**: Image structures and representations
   - **Process**: Processing operations
   - **Algorithms**: High-level workflows

---

## Package Architecture

### Data Package (`medical_image.data`)

**Purpose**: Define data structures and abstractions for medical images

```mermaid
graph TB
    subgraph "Abstract Layer"
        Image[Image ABC]
        MedicalDataset[MedicalDataset ABC]
    end
    
    subgraph "Concrete Implementations"
        DicomImage[DicomImage]
        PNGImage[PNGImage]
        CbisDdsm[CbisDdsm Dataset]
    end
    
    subgraph "Supporting Classes"
        Patch[Patch]
        PatchGrid[PatchGrid]
        ROI[RegionOfInterest]
        Annotation[Annotation]
    end
    
    Image --> DicomImage
    Image --> PNGImage
    MedicalDataset --> CbisDdsm
    
    PatchGrid --> Patch
    Image -.references.- Patch
    Image -.operates on.- ROI
    Image -.labeled by.- Annotation
    
    style Image fill:#FFE4B5
    style MedicalDataset fill:#FFE4B5
```

#### Image Class Hierarchy

```mermaid
classDiagram
    class Image {
        <<abstract>>
        -file_path: str
        -width: int
        -height: int
        -pixel_data: torch.Tensor
        -label: Annotation
        -device: str
        
        +__init__(file_path)
        +load()* abstract
        +save()* abstract
        +display_info()
        +to_png()
        +plot(cmap)
        +to_numpy()
    }
    
    class DicomImage {
        -dicom_data: pydicom.Dataset
        
        +__init__(file_path)
        +load() override
        +save() override
    }
    
    class PNGImage {
        +__init__(file_path)
        +load() override
        +save() override
    }
    
    Image <|-- DicomImage
    Image <|-- PNGImage
    
    note for Image "Lazy Loading:\n__init__: Store path only\nload(): Load pixel data"
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
def method_name(image_data: Image, output: Image, param1, param2, ...):
    """
    Process image_data and store result in output.
    
    Args:
        image_data: Input image (read-only)
        output: Output image (modified)
        param1, param2: Processing parameters
    """
    # 1. Access input data
    input_pixels = image_data.pixel_data
    
    # 2. Apply processing
    result = process(input_pixels, param1, param2)
    
    # 3. Store in output
    output.pixel_data = result
```

---

### Algorithms Package (`medical_image.algorithms`)

**Purpose**: Define high-level processing workflows

```mermaid
classDiagram
    class Algorithm {
        <<abstract>>
        +apply(image: Image, output: Image)* abstract
    }
    
    class FebdsAlgorithm {
        +method: str
        +dog: Callable
        +log: Callable
        +fft: Callable
        +median: Callable
        +gamma: Callable
        +binarize: Callable
        +otsu: Callable
        +morphology_closing: Callable
        +region_fill: Callable
        
        +__init__(method: str)
        +apply(image: Image, output: Image)
    }
    
    class CustomAlgorithm {
        +param1
        +param2
        +process1: Callable
        +process2: Callable
        
        +__init__(param1, param2)
        +apply(image: Image, output: Image)
    }
    
    Algorithm <|-- FebdsAlgorithm
    Algorithm <|-- CustomAlgorithm
    
    note for Algorithm "Template Method Pattern:\n__init__: Define steps as lambdas\napply: Execute sequence"
    
    note for FebdsAlgorithm "Lambda functions wrap\nstatic processing methods"
```

#### Algorithm Pattern

```python
class MyAlgorithm(Algorithm):
    def __init__(self, param1, param2):
        """Define processing steps as lambda functions."""
        super().__init__()
        
        # Step 1: Preprocessing
        self.preprocess = lambda img, out: Filters.gaussian_filter(
            img, out, sigma=param1
        )
        
        # Step 2: Enhancement
        self.enhance = lambda img, out: Filters.gamma_correction(
            img, out, gamma=param2
        )
        
        # Step 3: Segmentation
        self.segment = lambda img, out: Threshold.otsu_threshold(img, out)
    
    def apply(self, image: Image, output: Image):
        """Execute the sequence of processing steps."""
        # Execute lambda functions in order
        self.preprocess(image, output)
        self.enhance(output, output)
        self.segment(output, output)
```

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
    
    Note over User,Output: 3. Processing Pipeline
    User->>Filters: gaussian_filter(image, output, sigma=2.0)
    Filters->>DicomImage: Access pixel_data
    Filters->>Filters: Apply convolution
    Filters->>Output: Set pixel_data
    
    User->>Threshold: otsu_threshold(output, output)
    Threshold->>Output: Access pixel_data
    Threshold->>Threshold: Calculate threshold
    Threshold->>Output: Update pixel_data
    
    Note over User,Output: 4. Saving Results
    User->>Output: to_png()
    Output->>Output: Convert tensor to numpy
    Output->>Output: Save as PNG file
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
    Start([User calls apply]) --> Init{Check method}
    
    Init -->|method == 'dog'| DoG[Difference of Gaussian<br/>σ1=2.0, σ2=1.7]
    Init -->|method == 'log'| LoG[Laplacian of Gaussian<br/>σ=2.0]
    Init -->|method == 'fft'| FFT[FFT + Butterworth<br/>+ Inverse FFT]
    
    DoG --> Median
    LoG --> Median
    FFT --> Median
    
    Median[Median Filter<br/>size=5] --> Gamma[Gamma Correction<br/>γ=1.25]
    
    Gamma --> ThresholdCheck{Check method}
    
    ThresholdCheck -->|method == 'fft'| Binarize[Variance-based<br/>Binarization α=1]
    ThresholdCheck -->|method != 'fft'| Otsu[Otsu's<br/>Threshold]
    
    Binarize --> Closing
    Otsu --> Closing
    
    Closing[Morphological<br/>Closing] --> Fill[Region<br/>Filling]
    
    Fill --> End([Result in output])
    
    style DoG fill:#FFE4B5
    style LoG fill:#FFE4B5
    style FFT fill:#FFE4B5
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

# Concrete factories
class DicomImage(Image):
    def load(self):
        # DICOM-specific loading
        pass

class PNGImage(Image):
    def load(self):
        # PNG-specific loading
        pass
```

### 2. Template Method Pattern

**Used in**: Algorithm class

```python
class Algorithm(ABC):
    # Template method
    @abstractmethod
    def apply(self, image, output):
        pass

class FebdsAlgorithm(Algorithm):
    def __init__(self, method):
        # Define steps
        self.step1 = lambda: ...
        self.step2 = lambda: ...
    
    def apply(self, image, output):
        # Execute template
        self.step1()
        self.step2()
```

### 3. Strategy Pattern

**Used in**: FEBDS method selection

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
```

### 4. Lazy Initialization Pattern

**Used in**: Image loading

```python
class Image:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pixel_data = None  # Not loaded yet
    
    def load(self):
        # Load only when called
        self.pixel_data = load_from_file(self.file_path)
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
image.load()  # pixel_data loaded to GPU/CPU

# 3. Process - working memory
Filters.gaussian_filter(image, output, sigma=2.0)

# 4. Clear memory
del image
torch.cuda.empty_cache()  # Free GPU memory
```

---

## Extension Points

### Adding New Image Format

```python
from medical_image.data.image import Image

class TIFFImage(Image):
    def __init__(self, file_path):
        super().__init__(file_path)
        # TIFF-specific initialization
    
    def load(self):
        # Implement TIFF loading
        from PIL import Image as PILImage
        img = PILImage.open(self.file_path)
        self.pixel_data = torch.from_numpy(np.array(img))
        self.width, self.height = img.size
    
    def save(self):
        # Implement TIFF saving
        pass
```

### Adding New Processing Method

```python
class Filters:
    @staticmethod
    def bilateral_filter(image_data: Image, output: Image, 
                        sigma_color: float, sigma_space: float):
        """Add new filter to existing class."""
        # Implementation
        pass
```

### Adding New Algorithm

```python
from medical_image.algorithms.algorithm import Algorithm

class MyCustomAlgorithm(Algorithm):
    def __init__(self, param1, param2):
        super().__init__()
        # Define processing steps as lambdas
        self.step1 = lambda img, out: Filters.gaussian_filter(
            img, out, sigma=param1
        )
        self.step2 = lambda img, out: Threshold.otsu_threshold(img, out)
    
    def apply(self, image: Image, output: Image):
        # Execute sequence
        self.step1(image, output)
        self.step2(output, output)
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
├── test_dicom.py          # DICOM image tests (run by CI)
├── test_dataset.py        # Dataset tests
├── test_filters.py        # Filter tests
├── test_threshold.py      # Threshold tests
├── test_patch.py          # Patch system tests
└── dummy_data/            # Test data
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

- **Abstract base classes** define standard interfaces
- **Lazy loading** optimizes memory usage
- **Static methods** provide stateless processing
- **Lambda composition** enables flexible algorithms
- **Patch system** handles large images efficiently
- **GPU acceleration** improves performance

This architecture makes it easy to:
- Add new image formats
- Implement new processing methods
- Create custom algorithms
- Maintain and test code
