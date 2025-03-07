# Crypto_ParadoxOS Test Suite Documentation

This document provides detailed instructions for running and extending the test suite for the Crypto_ParadoxOS system.

## Test Organization

The test suite is organized into several modules, each focusing on a specific component of the system:

1. **Core Functionality Tests** (`test_core_functionality.py`)
   - Tests for basic transformation rules and the paradox resolver
   - Covers numerical, matrix, and text-based paradox resolution

2. **API Tests** (`test_api.py`)
   - Tests for the API layer, including job management
   - Validates rule registration and execution flow

3. **Optimizer Tests** (`test_optimizer.py`)
   - Tests for the resource allocation optimizer
   - Validates stakeholder preference resolution and constraint satisfaction

4. **Visualization Tests** (`test_visualization.py`)
   - Tests for visualization utilities and chart creation
   - Validates data preparation for different visualization types

5. **Integration Tests** (`test_paradox.py`)
   - End-to-end tests for common paradox types
   - Validates the complete resolution pipeline

## Running Tests

### Running All Tests

To run the complete test suite:

```bash
python run_tests.py
```

### Running Specific Test Modules

To run tests for a specific component:

```bash
python run_tests.py test_core_functionality
python run_tests.py test_api
python run_tests.py test_optimizer
python run_tests.py test_visualization
python run_tests.py test_paradox
```

### Running Individual Test Cases

To run a specific test case using the unittest module:

```bash
python -m unittest test_core_functionality.TestTransformationRules.test_fixed_point_iteration_numeric
```

### Using Verbose Output

For more detailed test output, use the `-v` flag:

```bash
python -m unittest test_core_functionality -v
```

## Writing New Tests

### Test Structure

Each test module follows a similar structure:

1. Import dependencies
2. Define test classes (one per component/feature)
3. Implement test methods with assertions

Example:

```python
import unittest
import numpy as np
from transformation_rules import fixed_point_iteration

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup code that runs before each test
        pass
        
    def test_specific_functionality(self):
        # Arrange
        input_value = 0.5
        expected_output = 2.0
        
        # Act
        result = fixed_point_iteration(input_value)
        
        # Assert
        self.assertAlmostEqual(result, expected_output, places=4)
```

### Test Naming Conventions

- Test module names should start with `test_`
- Test class names should start with `Test`
- Test method names should start with `test_`
- Test method names should clearly describe what's being tested

### Test Coverage Guidelines

When writing new tests, ensure that they cover:

1. **Happy Path** - Normal, expected functionality
2. **Edge Cases** - Extreme inputs, boundary conditions
3. **Error Cases** - Invalid inputs, error handling
4. **Complex Scenarios** - Multi-step operations, interactions between components

## Mocking and Test Utilities

For components that require complex setup or have external dependencies, use mocks:

```python
class MockParadoxState:
    def __init__(self):
        self.value = 42
        
    def get_history(self):
        return [1, 2, 42]
```

## Continuous Improvement

As the system evolves:

1. Add tests for new features
2. Update existing tests for changed functionality
3. Add regression tests for fixed bugs
4. Refactor tests for improved clarity and maintenance

## Common Test Assertions

- `assertEqual(a, b)` - Verify equality
- `assertAlmostEqual(a, b, places=7)` - Verify floating-point equality
- `assertTrue(x)` - Verify a condition is true
- `assertFalse(x)` - Verify a condition is false
- `assertRaises(ErrorType, callable, *args)` - Verify an error is raised
- `assertIn(item, container)` - Verify an item is in a container
- `assertIsInstance(obj, cls)` - Verify an object is an instance of a class

## Troubleshooting Tests

If tests are failing:

1. Check if the error is in the test or the implementation
2. Use print statements or logging for debugging
3. Check if test dependencies are correctly set up
4. Isolate the failing test with a specific test run
5. Review recent changes that might have affected the test

## Testing Goals

The goal of this test suite is to ensure:

1. **Correctness** - The system produces expected results
2. **Robustness** - The system handles unexpected inputs gracefully
3. **Performance** - The system meets performance requirements
4. **Regression Prevention** - Changes don't break existing functionality