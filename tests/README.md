# Elliott Wave Bot Test Suite

## Overview
Comprehensive test suite for the Elliott Wave Bot using pytest.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Fixtures and configuration
├── test_data_loader.py         # Data loading tests
├── test_wave_detector.py       # Wave detection tests
├── test_fibonacci.py           # Fibonacci analysis tests
├── test_strategy.py            # Trading strategy tests
├── test_backtester.py          # Backtesting tests
├── test_visualizer.py          # Visualization tests
└── test_integration.py         # Full workflow tests
```

## Running Tests

### Run all tests
```powershell
pytest
```

### Run with verbose output
```powershell
pytest -v
```

### Run specific test file
```powershell
pytest tests/test_wave_detector.py
```

### Run specific test
```powershell
pytest tests/test_wave_detector.py::TestWaveDetector::test_detect_waves_returns_list
```

### Run tests with coverage
```powershell
pytest --cov=src --cov-report=html
```

### Run only unit tests
```powershell
pytest -m unit
```

### Run only integration tests
```powershell
pytest -m integration
```

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows
- **Slow Tests**: Tests that take longer (marked with `@pytest.mark.slow`)
- **Network Tests**: Tests requiring internet (marked with `@pytest.mark.requires_network`)

## Fixtures

The `conftest.py` file provides several useful fixtures:

- `sample_data`: Generic OHLCV data for testing
- `wave_pattern_data`: Data with clear Elliott Wave patterns
- `config_dict`: Test configuration dictionary

## Writing New Tests

1. Create test file: `test_<module_name>.py`
2. Create test class: `class Test<ClassName>:`
3. Write test methods: `def test_<functionality>():`
4. Use fixtures for common data
5. Use markers for categorization

Example:
```python
import pytest
from src.module import MyClass

class TestMyClass:
    def test_initialization(self):
        """Test that class initializes correctly."""
        obj = MyClass()
        assert obj is not None
    
    @pytest.mark.slow
    def test_complex_operation(self, sample_data):
        """Test time-consuming operation."""
        obj = MyClass()
        result = obj.process(sample_data)
        assert result is not None
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure:
- Tests don't require manual interaction
- Network tests are marked appropriately
- Tests clean up any created files
- Tests are idempotent (can run multiple times)

## Troubleshooting

### Import Errors
If you see import errors, ensure you're running pytest from the project root:
```powershell
cd c:\Users\patri\Documents\ElliotWave\elliotBot
pytest
```

### Missing Dependencies
Install test dependencies:
```powershell
pip install pytest pytest-cov pytest-mock
```

### Slow Tests
Skip slow tests during development:
```powershell
pytest -m "not slow"
```
