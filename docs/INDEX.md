# Documentation Index

Welcome to the Medical Image Standard Library documentation!

## Documentation Structure

### Getting Started

1. **[README.md](../README.md)** - Start here!
   - Project overview and architecture
   - Installation instructions
   - Quick start guide
   - Class diagrams and sequence diagrams
   - CI/CD workflow explanation

### Core Documentation

2. **[Architecture](architecture.md)** - Deep dive into design
   - Design philosophy and principles
   - Package architecture details
   - Design patterns used
   - Workflow diagrams
   - Extension points
   - Performance considerations

3. **[API Reference](api_reference.md)** - Complete API documentation
   - Data module (Image, Patch, ROI, InMemoryImage, etc.)
   - Process module (Filters, Threshold, Metrics, etc.)
   - Algorithms module
   - Utils module (including device and logging)
   - Type hints and error handling

4. **[User Guide](user_guide.md)** - Practical tutorials
   - Working with images (lazy loading)
   - Image processing operations
   - Patch-based processing
   - Working with datasets
   - Using algorithms
   - GPU acceleration
   - Best practices
   - Troubleshooting

### Specialized Topics

5. **[Annotation & COCO Export API](annotation_api.md)** - Swagger-style API reference
   - Annotation class (center, bounding box, ROI)
   - Image-annotation integration (to_json / from_json)
   - COCO JSON export / import
   - JSON schemas and visualization examples
   - Installation and usage guide

6. **[Algorithm Reference](algorithms.md)** - Algorithm details
   - FEBDS algorithm (with paper reference)
   - Segmentation algorithms (K-Means, FCM, PFCM, Top-Hat)
   - Filtering algorithms
   - Thresholding algorithms
   - Morphological algorithms
   - Frequency domain algorithms
   - Algorithm selection guide

7. **[Dataset Guide](datasets.md)** - Working with medical datasets
   - CBIS-DDSM dataset
   - Creating custom datasets
   - Annotation types and formats
   - Data loading strategies
   - PyTorch integration

### Development

8. **[Contributing Guide](contributing.md)** - For contributors
   - Development setup
   - Code standards (Black formatting)
   - Testing requirements
   - CI validation requirements
   - Pull request process
   - Issue reporting

9. **[Quick Reference](quick_reference.md)** - Code snippets
   - Common operations cheat sheet
   - Quick code examples
   - Useful constants
   - Performance tips

---

## Quick Navigation

### I want to...

**Get started quickly**
-> [README.md](../README.md) -> [Quick Reference](quick_reference.md)

**Understand the architecture**
-> [Architecture](architecture.md)

**Learn how to use the library**
-> [User Guide](user_guide.md)

**Look up specific APIs**
-> [API Reference](api_reference.md)

**Understand algorithms**
-> [Algorithm Reference](algorithms.md)

**Work with annotations & COCO export**
-> [Annotation & COCO Export API](annotation_api.md)

**Work with datasets**
-> [Dataset Guide](datasets.md)

**Contribute code**
-> [Contributing Guide](contributing.md)

---

## Reading Paths

### For New Users

1. [README.md](../README.md) - Overview and installation
2. [Quick Reference](quick_reference.md) - Common operations
3. [User Guide](user_guide.md) - Detailed tutorials

### For Developers

1. [README.md](../README.md) - Architecture overview
2. [Architecture](architecture.md) - Design details
3. [API Reference](api_reference.md) - Complete API
4. [User Guide](user_guide.md) - GPU Acceleration
5. [Contributing Guide](contributing.md) - Development workflow

### For Researchers

1. [Algorithm Reference](algorithms.md) - Algorithm theory
2. [User Guide](user_guide.md) - Implementation examples
3. [Dataset Guide](datasets.md) - Working with CBIS-DDSM

---

## Key Concepts

### Lazy Loading Pattern
Images follow lazy loading: `__init__()` stores path, `load()` loads data.
See: [README.md](../README.md#lazy-loading-pattern)

### Static Processing Methods
All processing operations are static methods in classes.
See: [Architecture](architecture.md#process-package-medicalimageprocess)

### Algorithm Framework
Algorithms define processing pipelines using lambda functions.
See: [Architecture](architecture.md#algorithms-package-medicalimagealgorithms)

### Patch-based Processing
`PatchGrid` splits images into patches with automatic padding.
See: [User Guide](user_guide.md#working-with-patches)

### GPU Acceleration
The library supports GPU-accelerated processing through a device management
layer. `DeviceContext` provides a context manager for selecting compute devices
(CPU or CUDA). The `device.py` utility handles device detection, selection, and
mixed precision support, allowing algorithms to transparently run on available
hardware.
See: [User Guide](user_guide.md#gpu-acceleration)

---

## Diagrams

The documentation includes several types of diagrams:

### Class Diagrams
- [Data Package Architecture](../README.md#data-package-architecture)
- [Process Package Architecture](../README.md#process-package-architecture)
- [Algorithm Package Architecture](../README.md#algorithm-package-architecture)

### Sequence Diagrams
- [Image Loading Workflow](../README.md#image-loading-and-processing-workflow)
- [PatchGrid Splitting](../README.md#patchgrid-splitting-workflow)
- [Algorithm Application](../README.md#algorithm-application-workflow-febds)

### Workflow Diagrams
- [CI Pipeline](../README.md#github-ci-workflow)
- [FEBDS Algorithm Flow](architecture.md#febds-algorithm-execution-flow)

---

## CI/CD Information

### GitHub Actions Workflow

The project uses automated CI/CD:
- **Trigger**: Push to `master` branch
- **Tests**: Python 3.11 and 3.12
- **Checks**: pytest + Black formatting
- **Test suites**: `test_gpu.py` (GPU/device tests), `test_mc_algorithms.py` (algorithm tests), `test_dicom.py` (DICOM loading tests)

See: [README.md - GitHub CI Workflow](../README.md#github-ci-workflow)

### Before Pushing Code

```bash
# Ensure tests pass
pytest medical_image/tests/

# Ensure formatting is correct
black --check .

# Or run both
pytest medical_image/tests/ && black --check .
```

See: [Contributing Guide - CI Requirements](contributing.md#ci-requirements)

---

## Package Structure

```
medical-image-std/
├── README.md                    # Main documentation
├── docs/                        # Documentation folder
│   ├── INDEX.md                 # This file
│   ├── architecture.md          # Architecture details
│   ├── api_reference.md         # API documentation
│   ├── annotation_api.md        # Annotation & COCO API (Swagger-style)
│   ├── user_guide.md            # User tutorials
│   ├── algorithms.md            # Algorithm reference
│   ├── datasets.md              # Dataset guide
│   ├── contributing.md          # Contribution guide
│   └── quick_reference.md       # Quick reference
├── medical_image/               # Source code
│   ├── data/                    # Data structures
│   │   ├── dicom_image.py       # DICOM image handling
│   │   ├── image.py             # Base image class
│   │   ├── in_memory_image.py   # In-memory image support
│   │   ├── patch.py             # Patch and PatchGrid
│   │   ├── png_image.py         # PNG image handling
│   │   └── region_of_interest.py
│   ├── process/                 # Processing methods
│   ├── algorithms/              # Algorithm framework
│   ├── utils/                   # Utilities
│   │   ├── annotation.py        # Annotation helpers
│   │   ├── device.py            # GPU/device management
│   │   ├── image_utils.py       # Image utility functions
│   │   └── logging.py           # Logging configuration
│   └── tests/                   # Unit tests
│       ├── test_dicom.py        # DICOM tests
│       ├── test_gpu.py          # GPU and device tests
│       └── test_mc_algorithms.py # Algorithm tests
└── pyproject.toml               # Package configuration
```

---

## External Resources

- **Repository**: https://github.com/LATIS-DocumentAI-Group/medical-image-std
- **CBIS-DDSM Dataset**: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
- **PyTorch Documentation**: https://pytorch.org/docs/stable/
- **FEBDS Paper**: See `paper.pdf` in repository

---

## Documentation Standards

All documentation follows:
- **Markdown format** with GitHub Flavored Markdown
- **Mermaid diagrams** for visualizations
- **Code examples** with syntax highlighting
- **Clear structure** with table of contents
- **Cross-references** between documents

---

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Create an issue with bug report template
- **Features**: Create an issue with feature request template
- **Contributing**: See [Contributing Guide](contributing.md)

---

## License

MIT License - See LICENSE file for details

---

**Last Updated**: 2026-04-15
**Version**: 0.5.0