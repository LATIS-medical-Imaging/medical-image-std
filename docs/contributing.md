# Contributing to Medical Image Standard Library

Thank you for your interest in contributing to the Medical Image Standard Library! This document provides guidelines and best practices for contributing to the project.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Standards](#code-standards)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Pull Request Process](#pull-request-process)
7. [Issue Reporting](#issue-reporting)

---

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- Git
- Linux operating system
- CUDA-compatible GPU (optional)

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/medical-image-std.git
cd medical-image-std

# Add upstream remote
git remote add upstream https://github.com/LATIS-DocumentAI-Group/medical-image-std.git
```

---

## Development Setup

### Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Install Dependencies

```bash
# Install all dependencies including dev tools
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Install Development Tools

```bash
# Install additional development tools
pip install black pytest pytest-cov flake8 mypy
```

---

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length:** 100 characters maximum
- **Indentation:** 4 spaces
- **Quotes:** Double quotes for strings
- **Imports:** Organized and sorted

### Code Formatting

We use **Black** for automatic code formatting:

```bash
# Format all Python files
black medical_image/

# Check formatting without making changes
black --check medical_image/
```

### Type Hints

All functions should include type hints:

```python
from typing import Optional, List, Tuple
import torch

def process_image(
    image: Image,
    output: Image,
    sigma: float = 2.0
) -> torch.Tensor:
    """
    Process an image with Gaussian filter.
    
    Args:
        image: Input image
        output: Output image object
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Processed image tensor
    """
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string
        
    Examples:
        >>> result = example_function(5, "test")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    
    return True
```

### Naming Conventions

- **Classes:** PascalCase (e.g., `DicomImage`, `FebdsAlgorithm`)
- **Functions/Methods:** snake_case (e.g., `load_image`, `apply_filter`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `MAX_SIZE`, `DEFAULT_SIGMA`)
- **Private methods:** Prefix with underscore (e.g., `_internal_method`)

---

## Testing

### Test Structure Guidelines

**IMPORTANT**: All unit tests must follow the existing test structure and organization:

1. **Location**: Place tests in `medical_image/tests/`
2. **Naming**: Test files should be named `test_*.py`
3. **Structure**: Follow the same pattern as existing tests
4. **Organization**: Group related tests in classes

### Writing Tests

All new features should include tests following this structure:

```python
# medical_image/tests/test_filters.py
import pytest
import torch
from medical_image.data.dicom_image import DicomImage
from medical_image.process.filters import Filters

class TestFilters:
    """Test suite for filter operations."""
    
    def test_gaussian_filter(self):
        """Test Gaussian filter application."""
        # Create test image
        input_img = DicomImage("tests/dummy_data/sample.dcm")
        input_img.load()
        
        output_img = DicomImage("tests/dummy_data/output.dcm")
        
        # Apply filter
        Filters.gaussian_filter(input_img, output_img, sigma=2.0)
        
        # Assertions
        assert output_img.pixel_data is not None
        assert output_img.pixel_data.shape == input_img.pixel_data.shape
        assert torch.is_tensor(output_img.pixel_data)
    
    def test_gaussian_filter_invalid_sigma(self):
        """Test Gaussian filter with invalid sigma."""
        input_img = DicomImage("tests/dummy_data/sample.dcm")
        input_img.load()
        
        output_img = DicomImage("tests/dummy_data/output.dcm")
        
        # Should raise error for negative sigma
        with pytest.raises(ValueError):
            Filters.gaussian_filter(input_img, output_img, sigma=-1.0)
```

### CI Requirements

**All code must pass CI checks before merging:**

#### 1. Tests Must Pass
```bash
# Run the same tests that CI runs
pytest medical_image/tests/test_dicom.py

# Ensure all your new tests pass
pytest medical_image/tests/test_your_feature.py
```

#### 2. Black Formatting Must Pass
```bash
# Format your code with Black
black medical_image/

# Check formatting (this is what CI runs)
black --check .

# If check fails, run black without --check to auto-format
black .
```

#### 3. CI Validation Checklist

Before pushing, ensure:
- [ ] All tests pass locally: `pytest`
- [ ] Code is formatted with Black: `black --check .`
- [ ] New tests follow existing structure
- [ ] Tests are in `medical_image/tests/`
- [ ] Test files are named `test_*.py`

### Running Tests

```bash
# Run all tests
pytest

# Run tests that CI runs
pytest medical_image/tests/test_dicom.py

# Run with coverage
pytest --cov=medical_image --cov-report=html

# Run specific test file
pytest medical_image/tests/test_filters.py

# Run specific test
pytest medical_image/tests/test_filters.py::TestFilters::test_gaussian_filter

# Run with verbose output
pytest -v
```

### Test Coverage

Aim for at least 80% code coverage:

```bash
# Generate coverage report
pytest --cov=medical_image --cov-report=term-missing

# View HTML coverage report
pytest --cov=medical_image --cov-report=html
firefox htmlcov/index.html
```

### GitHub CI Workflow

The CI pipeline runs automatically on push to `master`:

1. **Matrix Testing**: Tests run on Python 3.11 and 3.12
2. **Test Execution**: `pytest medical_image/tests/test_dicom.py`
3. **Format Check**: `black --check .`
4. **Result**: Build fails if either tests fail OR formatting is incorrect

**To ensure CI passes:**
```bash
# Run this before pushing
pytest medical_image/tests/test_dicom.py && black --check .
```

---

## Documentation

### Code Documentation

- All public classes, methods, and functions must have docstrings
- Include type hints for all parameters and return values
- Provide examples in docstrings when helpful

### User Documentation

When adding new features, update relevant documentation:

- `docs/api_reference.md`: API documentation
- `docs/user_guide.md`: Usage examples
- `docs/algorithms.md`: Algorithm descriptions
- `docs/datasets.md`: Dataset information

### Documentation Format

Use Markdown with proper formatting:

```markdown
## Function Name

**Purpose:** Brief description

**Parameters:**
- `param1` (type): Description
- `param2` (type): Description

**Returns:**
- `type`: Description

**Example:**
\`\`\`python
result = function_name(param1, param2)
\`\`\`
```

---

## Pull Request Process

### Before Submitting

1. **Update your fork:**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes:**
   - Write code following style guidelines
   - Add tests for new features
   - Update documentation

4. **Run tests:**
   ```bash
   pytest
   black --check medical_image/
   ```

5. **Commit changes:**
   ```bash
   git add .
   git commit -m "Add feature: description of your feature"
   ```

### Commit Message Guidelines

Follow conventional commits format:

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(filters): add bilateral filter implementation

Implement bilateral filter for edge-preserving smoothing.
Includes tests and documentation.

Closes #123
```

```
fix(dicom): handle missing DICOM metadata

Add error handling for DICOM files with missing metadata fields.
```

### Submitting Pull Request

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request on GitHub:**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill in PR template

3. **PR Description should include:**
   - What changes were made
   - Why the changes were necessary
   - How to test the changes
   - Related issues (if any)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Code coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

### Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, PR will be merged

---

## Issue Reporting

### Before Creating an Issue

1. Search existing issues to avoid duplicates
2. Check if it's already fixed in the latest version
3. Gather relevant information (error messages, system info)

### Bug Reports

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Linux (Ubuntu 22.04)
- Python version: 3.11
- Library version: 0.2.8.dev1
- CUDA version (if applicable): 12.8

## Error Messages
```
Paste error messages here
```

## Additional Context
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches you've considered

## Additional Context
Any other relevant information
```

---

## Development Workflow

### Typical Workflow

1. **Find or create an issue**
2. **Discuss approach** (for major changes)
3. **Fork and clone** repository
4. **Create feature branch**
5. **Implement changes**
6. **Write tests**
7. **Update documentation**
8. **Run tests and formatting**
9. **Commit and push**
10. **Create pull request**
11. **Address review feedback**
12. **Merge**

### Branch Naming

- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation updates
- `refactor/description`: Code refactoring

---

## Code Review Guidelines

### For Contributors

- Be open to feedback
- Respond to comments promptly
- Make requested changes
- Ask questions if unclear

### For Reviewers

- Be respectful and constructive
- Explain reasoning for requested changes
- Acknowledge good work
- Focus on code quality and maintainability

---

## Additional Guidelines

### Adding New Algorithms

1. **Create algorithm class:**
   ```python
   from medical_image.algorithms.algorithm import Algorithm
   
   class NewAlgorithm(Algorithm):
       def __init__(self, param1, param2):
           super().__init__()
           self.param1 = param1
           self.param2 = param2
       
       def apply(self, image: Image, output: Image):
           # Implementation
           pass
   ```

2. **Add tests:**
   ```python
   def test_new_algorithm():
       algo = NewAlgorithm(param1=1.0, param2=2.0)
       # Test implementation
   ```

3. **Document in `docs/algorithms.md`**

### Adding New Filters

1. **Add static method to `Filters` class**
2. **Include docstring with mathematical definition**
3. **Add tests**
4. **Update documentation**

### Adding Dataset Support

1. **Extend `MedicalDataset` class**
2. **Implement required abstract methods**
3. **Add dataset-specific documentation**
4. **Include example usage**

---

## Getting Help

- **Questions:** Open a discussion on GitHub
- **Bugs:** Create an issue
- **Chat:** Join our community (if available)
- **Email:** Contact maintainers

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the Medical Image Standard Library!
