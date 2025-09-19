# Contributing to Traffic AI

Thank you for your interest in contributing to Traffic AI! We welcome contributions from the community and are pleased to have you join us.

## 🤝 Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## 🚀 How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include screenshots if applicable**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Describe the current behavior and explain the enhanced behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install dependencies**: `pip install -r requirements-dev.txt`
3. **Make your changes** following our coding standards
4. **Add tests** for your changes
5. **Run the test suite**: `pytest`
6. **Run code quality checks**: `black . && flake8 && mypy src/`
7. **Update documentation** if needed
8. **Commit your changes** with a clear commit message
9. **Push to your fork** and submit a pull request

## 📋 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended for development)

### Local Development Environment

1. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/traffic-ai.git
   cd traffic-ai
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/
   ```

## 🎨 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Quotes**: Use double quotes for strings
- **Imports**: Use `isort` for import sorting
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use automated code formatting tools:

```bash
# Format code
black .

# Sort imports  
isort .

# Check style
flake8

# Type checking
mypy src/
```

### Pre-commit Hooks

We recommend using pre-commit hooks to automatically format code:

```bash
pip install pre-commit
pre-commit install
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_detection.py

# Run with coverage
pytest --cov=src/traffic_ai tests/

# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)
- Use pytest fixtures for common setup
- Mock external dependencies

Example test structure:
```python
def test_vehicle_detection_with_valid_input():
    # Arrange
    detector = YOLODetector(device="cpu")
    test_frame = create_test_frame()
    
    # Act
    result = detector.detect_frame(test_frame)
    
    # Assert
    assert isinstance(result, DetectionResult)
    assert len(result.detections) >= 0
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def detect_vehicles(self, frame: np.ndarray) -> DetectionResult:
    """
    Detect vehicles in the given frame.
    
    Args:
        frame: Input image as numpy array with shape (H, W, C)
        
    Returns:
        DetectionResult containing all vehicle detections
        
    Raises:
        ValueError: If frame is empty or invalid format
        RuntimeError: If model is not loaded
        
    Example:
        >>> detector = YOLODetector()
        >>> result = detector.detect_vehicles(frame)
        >>> print(f"Found {len(result.detections)} vehicles")
    """
```

### README Updates

When adding new features, please update the README with:
- Installation instructions (if dependencies change)
- Usage examples
- API documentation
- Configuration options

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/). Version numbers follow the pattern `MAJOR.MINOR.PATCH`:

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## 📦 Release Process

1. Update version in `src/traffic_ai/__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Create release commit: `git commit -m "Release v1.2.3"`
4. Tag the release: `git tag -a v1.2.3 -m "Release v1.2.3"`
5. Push tags: `git push origin main --tags`

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

### Core Features
- New detection algorithms
- Advanced tracking methods
- Traffic prediction models
- Performance optimizations

### Integration & Deployment
- Docker improvements
- Cloud deployment scripts
- CI/CD enhancements
- Monitoring and logging

### Documentation & Examples
- Tutorial notebooks
- Video tutorials
- API documentation
- Real-world examples

### Testing & Quality
- Test coverage improvements
- Performance benchmarks
- Integration tests
- Load testing

## 💬 Community

- **Discord**: [Traffic AI Community](https://discord.gg/traffic-ai)
- **GitHub Discussions**: [Project Discussions](https://github.com/lumina2006/traffic-ai/discussions)
- **Email**: contact@traffic-ai.com

## 📄 License

By contributing to Traffic AI, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README contributors section
- Release notes
- Project website (coming soon)

Thank you for contributing to Traffic AI! 🚗🤖