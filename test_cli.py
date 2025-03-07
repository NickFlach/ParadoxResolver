#!/usr/bin/env python3
"""
Test script for the Crypto_ParadoxOS CLI
This script validates that the CLI works as expected on both platforms.
"""

import os
import sys
import platform
import subprocess
import time

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def run_test(command, expected_output=None, should_fail=False):
    """Run a test command and validate the output."""
    print(f"\n> Testing command: {command}")
    
    try:
        # Run the command and capture output
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), capture_output=True, text=True)
        
        # Print the output for debugging
        if result.stdout:
            print(f"  Output: {result.stdout[:100]}...")
        else:
            print("  No output received.")
        
        if result.stderr:
            print(f"  Errors: {result.stderr[:100]}...")
        
        # Check if the command executed as expected
        if should_fail:
            if result.returncode != 0:
                print("  ✅ Test passed (command failed as expected)")
                return True
            else:
                print("  ❌ Test failed (command succeeded but was expected to fail)")
                return False
        else:
            if result.returncode == 0:
                if expected_output and expected_output not in result.stdout:
                    print(f"  ❌ Test failed (expected output not found)")
                    return False
                print("  ✅ Test passed")
                return True
            else:
                print(f"  ❌ Test failed (exit code {result.returncode})")
                return False
    
    except Exception as e:
        print(f"  ❌ Test failed with exception: {str(e)}")
        return False

def main():
    """Run a series of tests on the CLI."""
    print_header("Crypto_ParadoxOS CLI Testing")
    
    # Determine whether we're testing Windows or Linux
    system = platform.system()
    print(f"Testing on {system} platform")
    
    if system == "Windows":
        cli_command = "paradox_cli_windows.exe"
    else:
        cli_command = "./paradox_cli_linux"
    
    # Track test results
    total_tests = 0
    passed_tests = 0
    
    # Test help message
    total_tests += 1
    if run_test(f"{cli_command} --help", expected_output="usage"):
        passed_tests += 1
    
    # Test basic paradox resolution
    total_tests += 1
    if run_test(f"{cli_command} resolve --input 'x = 1/x' --type numerical", 
                expected_output="Final state"):
        passed_tests += 1
    
    # Test invalid input handling
    total_tests += 1
    if run_test(f"{cli_command} resolve", should_fail=True):
        passed_tests += 1
    
    # Test evolutionary engine
    total_tests += 1
    if run_test(f"{cli_command} evolve --generations 2 --population 5", 
                expected_output="Evolution completed"):
        passed_tests += 1
    
    # Test meta-resolver
    total_tests += 1
    if run_test(f"{cli_command} meta-resolve --input '0.5' --type numerical",
                expected_output="Meta-resolution completed"):
        passed_tests += 1
    
    # Print summary
    print_header("Test Summary")
    print(f"Passed {passed_tests} out of {total_tests} tests")
    
    if passed_tests == total_tests:
        print("\n✅ All tests passed! The CLI is working correctly on this platform.")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())