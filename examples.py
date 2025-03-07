#!/usr/bin/env python3
"""
Crypto_ParadoxOS Example Paradoxes

This module provides a collection of example paradoxes that can be used
to demonstrate the capabilities of Crypto_ParadoxOS.
"""

from typing import Dict, Any

def get_example_paradoxes() -> Dict[str, Dict[str, Any]]:
    """
    Return a dictionary of example paradoxes that users can select from.
    
    Each example includes a description, type, and the paradox itself.
    
    Returns:
        Dictionary of example paradoxes
    """
    examples = {
        "fixed_point": {
            "name": "Fixed-Point Equation",
            "description": "A self-referential equation where x equals the reciprocal of itself",
            "type": "numerical",
            "value": "x = 1/x",
            "initial_value": 0.5
        },
        "golden_ratio": {
            "name": "Golden Ratio Convergence",
            "description": "An iterative process that converges to the golden ratio",
            "type": "numerical",
            "value": 1.0,
            "initial_value": 1.0,
            "notes": "Converges to φ ≈ 1.618... using the recurrence relation x_n+1 = 1 + 1/x_n"
        },
        "matrix_eigenvalue": {
            "name": "Matrix Eigenvalue Paradox",
            "description": "A matrix with seemingly contradictory eigenstructure",
            "type": "matrix",
            "value": [[0.7, 0.3], [0.4, 0.6]],
            "notes": "A stochastic matrix that represents a Markov process with steady state"
        },
        "voting_paradox": {
            "name": "Condorcet Voting Paradox",
            "description": "Collective preferences that form a cycle, violating transitivity",
            "type": "matrix",
            "value": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            "notes": "A preference matrix where A > B, B > C, but C > A"
        },
        "liars_paradox": {
            "name": "Liar's Paradox",
            "description": "The classic self-referential statement 'This statement is false'",
            "type": "text",
            "value": "This statement is false",
            "notes": "A self-referential statement that creates a logical contradiction"
        },
        "barber_paradox": {
            "name": "Barber Paradox",
            "description": "The barber shaves all those who do not shave themselves",
            "type": "text",
            "value": "The barber shaves all and only those who do not shave themselves",
            "notes": "A self-referential scenario: who shaves the barber?"
        },
        "probability_reversal": {
            "name": "Probability Reversal",
            "description": "A scenario where adding information decreases probability",
            "type": "numerical",
            "value": 0.7,
            "initial_value": 0.7,
            "notes": "Demonstrates how Bayesian updates can lead to counterintuitive results"
        },
        "arrow_impossibility": {
            "name": "Arrow's Impossibility Theorem",
            "description": "A social choice scenario demonstrating the impossibility of perfect voting",
            "type": "matrix",
            "value": [
                [1, 2, 3], 
                [2, 3, 1], 
                [3, 1, 2]
            ],
            "notes": "Preference rankings of 3 voters over 3 alternatives, showing cyclic majority preferences"
        },
        "quantum_superposition": {
            "name": "Quantum Superposition Paradox",
            "description": "A simplified representation of quantum state that is both 0 and 1",
            "type": "numerical",
            "value": 0.5,
            "notes": "Represents a qubit in equal superposition between states"
        },
        "recursive_structure": {
            "name": "Recursive Structure",
            "description": "A nested structure that contains itself",
            "type": "text",
            "value": "This sentence refers to [self], where [self] is this entire sentence",
            "notes": "Demonstrates self-reference through recursive structure"
        }
    }
    
    return examples

def print_example_descriptions():
    """Print descriptions of all available examples."""
    examples = get_example_paradoxes()
    
    print("Available Paradox Examples:")
    print("=========================")
    
    for key, example in examples.items():
        print(f"\n{example['name']} (key: '{key}')")
        print(f"  Description: {example['description']}")
        print(f"  Type: {example['type']}")
        value_str = str(example['value'])
        if len(value_str) > 50:
            value_str = value_str[:47] + "..."
        print(f"  Value: {value_str}")
        if "notes" in example:
            print(f"  Notes: {example['notes']}")

def get_example_command(key: str) -> str:
    """
    Get a command-line invocation for the specified example.
    
    Args:
        key: The example key
        
    Returns:
        A string with the command to run the example
    """
    examples = get_example_paradoxes()
    
    if key not in examples:
        return f"Error: Example '{key}' not found"
    
    example = examples[key]
    
    if example['type'] == 'numerical':
        cmd = f"paradox_cli.py resolve --input \"{example['value']}\" --type numerical"
        if 'initial_value' in example:
            cmd += f" --initial-value {example['initial_value']}"
    
    elif example['type'] == 'matrix':
        # Format matrix for command line
        matrix_str = str(example['value']).replace(' ', '')
        cmd = f"paradox_cli.py resolve --input \"{matrix_str}\" --type matrix"
    
    elif example['type'] == 'text':
        cmd = f"paradox_cli.py resolve --input \"{example['value']}\" --type text"
    
    else:
        return f"Error: Unknown example type '{example['type']}'"
    
    return cmd

if __name__ == "__main__":
    print_example_descriptions()
    
    print("\nExample Commands:")
    print("================")
    
    # Print example commands for a few interesting cases
    for key in ['fixed_point', 'matrix_eigenvalue', 'liars_paradox']:
        print(f"\n{get_example_command(key)}")