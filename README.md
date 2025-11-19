# Medical Image Standard Library

A standardized Python library for medical image processing, providing abstract interfaces and extensible implementations for DICOM and other medical image formats.

---

## Purpose

Provide **standardized abstractions** for medical image processing workflows:

- **Abstract base classes** defining core interfaces (`Image`, `Algorithm`)
- **Lazy loading pattern** for memory-efficient image handling
- **Static processing methods** for filters, thresholding, and morphology
- **Extensible algorithm framework** using lambda composition
- **Patch-based processing** for large images
- **GPU acceleration** via PyTorch tensors

---

##  Architecture Overview

### Core Design Principles

1. **Abstraction-First**: Define interfaces through abstract base classes
2. **Lazy Loading**: `__init__()` stores metadata, `load()` loads data
3. **Static Methods**: Stateless processing operations
4. **Composition**: Build algorithms from lambda functions
5. **Extensibility**: Easy to add formats, algorithms, and operations

### Package Structure

```
medical_image/
├── data/                # Abstract Image, Patch, PatchGrid, ROI
├── process/             # Static methods: Filters, Threshold, Metrics
├── algorithms/          # Abstract Algorithm, FEBDS implementation
└── utils/               # Error handling, annotations
```

**📖 Detailed Architecture**: See [docs/architecture.md](docs/architecture.md)

---

## Installation

### Requirements
- Python 3.11 or 3.12
- Linux OS
- CUDA GPU (optional)

### Install

```bash
git clone https://github.com/LATIS-DocumentAI-Group/medical-image-std.git
cd medical-image-std
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1. Load Image (Lazy Loading)

```python
from medical_image.data.dicom_image import DicomImage

# Create object (no data loaded yet)
image = DicomImage("mammogram.dcm")

# Load data when needed
image.load()  # ← Lazy loading

# Display and visualize
image.display_info()
image.plot()
```

### 2. Apply Processing

```python
from medical_image.process.filters import Filters
from medical_image.process.threshold import Threshold

# Create output image
output = DicomImage("output.dcm")

# Apply filters (static methods)
Filters.gaussian_filter(image, output, sigma=2.0)
Threshold.otsu_threshold(output, output)

# Save result
output.to_png()
```

### 3. Use Algorithms

```python
from medical_image.algorithms.FEBDS import FebdsAlgorithm

# Create algorithm (lambda functions defined in __init__)
febds = FebdsAlgorithm(method="dog")

# Apply algorithm sequence
febds.apply(image, output)
```

### 4. Patch-based Processing

```python
from medical_image.data.patch import PatchGrid

# Create patch grid (calls _split() automatically)
patch_grid = PatchGrid(image, patch_size=(256, 256))

# Process each patch
for patch in patch_grid.patches:
    # Process patch.pixel_data
    pass

# Reconstruct
reconstructed = patch_grid.reconstruct()
```

---

## Key Concepts

### Lazy Loading Pattern
- **Object Creation**: `image = DicomImage("path.dcm")` → Only stores path
- **Data Loading**: `image.load()` → Loads pixel data to memory
- **Memory Efficient**: Load only when needed, clear when done

### Static Processing Methods
All processing operations are static methods:
- **Filters**: `Filters.gaussian_filter()`, `Filters.median_filter()`, etc.
- **Threshold**: `Threshold.otsu_threshold()`, `Threshold.sauvola_threshold()`, etc.
- **Metrics**: `Metrics.entropy()`, `Metrics.mutual_information()`, etc.

### Algorithm Framework
Algorithms define processing pipelines:
- **`__init__`**: Define steps as lambda functions
- **`apply`**: Execute sequence of lambdas

Example:
```python
class MyAlgorithm(Algorithm):
    def __init__(self):
        self.step1 = lambda img, out: Filters.gaussian_filter(img, out, sigma=2.0)
        self.step2 = lambda img, out: Threshold.otsu_threshold(img, out)
    
    def apply(self, image, output):
        self.step1(image, output)
        self.step2(output, output)
```

### PatchGrid System
- **Automatic splitting**: `_split()` called in `__init__()`
- **Automatic padding**: Handles non-divisible dimensions
- **Easy reconstruction**: `reconstruct()` removes padding

---

## Documentation

| Document | Description |
|----------|-------------|
| **[INDEX](docs/INDEX.md)** | Documentation navigation and overview |
| **[Architecture](docs/architecture.md)** | Design patterns, diagrams, workflows |
| **[API Reference](docs/api_reference.md)** | Complete API documentation |
| **[User Guide](docs/user_guide.md)** | Tutorials and examples |
| **[Algorithms](docs/algorithms.md)** | Algorithm theory and implementation |
| **[Datasets](docs/datasets.md)** | CBIS-DDSM and custom datasets |
| **[Contributing](docs/contributing.md)** | Development guide and CI requirements |
| **[Quick Reference](docs/quick_reference.md)** | Code snippets cheat sheet |

---

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run CI tests
pytest medical_image/tests/test_dicom.py

# Check formatting
black --check .
```

### CI Requirements

All code must pass CI before merging:
- ✅ Tests pass: `pytest medical_image/tests/test_dicom.py`
- ✅ Black formatting: `black --check .`

**Pre-push validation**:
```bash
pytest medical_image/tests/test_dicom.py && black --check .
```

**📖 CI Details**: See [docs/contributing.md](docs/contributing.md#ci-requirements)

---

## Development

### Code Formatting

```bash
# Format code
black medical_image/

# Check formatting (CI requirement)
black --check .
```

### Adding Features

- **New Image Format**: Extend `Image` abstract class
- **New Processing Method**: Add static method to appropriate class
- **New Algorithm**: Extend `Algorithm` abstract class

**📖 Extension Guide**: See [docs/architecture.md](docs/architecture.md#extension-points)

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Follow code standards (Black formatting)
4. Write tests following existing structure
5. Ensure CI passes locally
6. Submit pull request

**📖 Full Guide**: See [docs/contributing.md](docs/contributing.md)

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Links

- **Repository**: https://github.com/LATIS-DocumentAI-Group/medical-image-std
- **Documentation**: [docs/INDEX.md](docs/INDEX.md)

---

## Version

**Current**: 0.2.8.dev1

---

## Quick Navigation

**Getting Started** → [Installation](#-installation) → [Quick Start](#-quick-start)  
**Learn More** → [Documentation](#-documentation) → [Architecture](docs/architecture.md)  
**Contribute** → [Contributing Guide](docs/contributing.md) → [CI Requirements](docs/contributing.md#ci-requirements)
