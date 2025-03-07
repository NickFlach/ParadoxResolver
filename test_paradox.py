#!/usr/bin/env python3
"""
Test script for Crypto_ParadoxOS

This script tests the core functionality with predefined examples.
"""

import numpy as np
from crypto_paradox_os import ParadoxResolver, get_standard_rules

# Test with numerical paradox
def test_numerical():
    print("\n=== Testing Numerical Paradox Resolution ===")
    resolver = ParadoxResolver(
        transformation_rules=get_standard_rules(),
        max_iterations=20,
        convergence_threshold=0.001
    )
    
    # Fixed-point equation x = 1/x (should converge to 1 or -1)
    initial_state = 0.5
    print(f"Initial state: {initial_state}")
    
    # Resolve the paradox
    result, steps, converged = resolver.resolve(initial_state)
    
    # Show results
    print(f"Converged: {converged}")
    print(f"Final state: {result}")
    print(f"Steps taken: {len(steps)-1}")
    
    # Show history
    if len(steps) <= 10:
        print("\nResolution steps:")
        for i, step in enumerate(steps):
            print(f"Step {i}: {step}")

# Test with matrix paradox
def test_matrix():
    print("\n=== Testing Matrix Paradox Resolution ===")
    resolver = ParadoxResolver(
        transformation_rules=get_standard_rules(),
        max_iterations=20,
        convergence_threshold=0.001
    )
    
    # Unstable matrix with eigenvalues > 1
    initial_state = np.array([
        [1.1, 0.3, 0.1],
        [0.2, 0.9, 0.4],
        [0.1, 0.2, 1.2]
    ])
    print(f"Initial state:\n{initial_state}")
    
    # Resolve the paradox
    result, steps, converged = resolver.resolve(initial_state)
    
    # Show results
    print(f"Converged: {converged}")
    print(f"Final state:\n{result}")
    print(f"Steps taken: {len(steps)-1}")

# Run the tests
if __name__ == "__main__":
    print("Crypto_ParadoxOS Test Suite")
    print("===========================")
    test_numerical()
    test_matrix()
    print("\n=== All Tests Complete ===")