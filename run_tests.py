#!/usr/bin/env python
import unittest
import sys

# Import test modules
import test_core_functionality
import test_optimizer
import test_api
import test_visualization


def run_all_tests():
    """Run all test suites."""
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite and add tests
    test_suite = unittest.TestSuite()
    
    # Add all tests from modules
    test_suite.addTests(loader.loadTestsFromModule(test_core_functionality))
    test_suite.addTests(loader.loadTestsFromModule(test_optimizer))
    test_suite.addTests(loader.loadTestsFromModule(test_api))
    test_suite.addTests(loader.loadTestsFromModule(test_visualization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_name):
    """Run a specific test module."""
    if test_name == "core":
        suite = unittest.TestLoader().loadTestsFromModule(test_core_functionality)
    elif test_name == "optimizer":
        suite = unittest.TestLoader().loadTestsFromModule(test_optimizer)
    elif test_name == "api":
        suite = unittest.TestLoader().loadTestsFromModule(test_api)
    elif test_name == "visualization":
        suite = unittest.TestLoader().loadTestsFromModule(test_visualization)
    else:
        print(f"Unknown test module: {test_name}")
        print("Available modules: core, optimizer, api, visualization")
        return 1
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        sys.exit(run_specific_test(sys.argv[1]))
    else:
        # Run all tests
        sys.exit(run_all_tests())