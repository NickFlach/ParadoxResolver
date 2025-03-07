#!/usr/bin/env python3
"""
Crypto_ParadoxOS Utility Functions

This module contains utility functions for the Crypto_ParadoxOS system.
"""

import re
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union

def format_paradox_input(paradox_input: Any, input_type: str, initial_value: Optional[float] = None) -> Any:
    """
    Format and prepare the paradox input for processing.
    
    Args:
        paradox_input: Raw input from the user
        input_type: Type of input ("Text", "Numerical", "Matrix")
        initial_value: Initial value for numerical paradoxes
        
    Returns:
        Formatted input ready for the resolver
    """
    if input_type.lower() == "numerical":
        # Handle numerical paradoxes
        if isinstance(paradox_input, (int, float)):
            return float(paradox_input)
        
        # Handle equation strings like "x = 1/x"
        elif isinstance(paradox_input, str) and "=" in paradox_input:
            # Just return the string for now, the resolver will handle it
            return paradox_input
        
        # Default to initial value if provided
        elif initial_value is not None:
            return float(initial_value)
        
        else:
            return 0.5  # Default starting value
    
    elif input_type.lower() == "matrix":
        # Ensure we have a proper numpy array
        if isinstance(paradox_input, np.ndarray):
            return paradox_input
        
        # Convert list to numpy array
        elif isinstance(paradox_input, list):
            return np.array(paradox_input)
        
        # Handle string representation of matrix
        elif isinstance(paradox_input, str):
            # Simple conversion (a full implementation would be more robust)
            try:
                # Try to evaluate as a Python expression
                matrix = eval(paradox_input)
                return np.array(matrix)
            except Exception:
                # Default to a simple 2x2 identity matrix
                return np.eye(2)
        
        # Default matrix
        else:
            return np.eye(2)
    
    elif input_type.lower() == "text":
        # Just return the text as is
        if isinstance(paradox_input, str):
            return paradox_input
        
        # Convert to string if not a string
        else:
            return str(paradox_input)
    
    # Default case - just return input as-is
    return paradox_input

def validate_input(formatted_input: Any, input_type: str) -> Tuple[bool, str]:
    """
    Validate the formatted input before processing.
    
    Args:
        formatted_input: Input that has been formatted
        input_type: Type of input ("Text", "Numerical", "Matrix")
        
    Returns:
        Tuple of (is_valid, validation_message)
    """
    if input_type.lower() == "numerical":
        # Check if it's a number
        if isinstance(formatted_input, (int, float)):
            return True, "Valid numerical input"
        
        # Check if it's an equation string
        elif isinstance(formatted_input, str) and "=" in formatted_input:
            # Simple validation (could be more sophisticated)
            if "x" in formatted_input:
                return True, "Valid equation input"
            else:
                return False, "Equation must contain the variable 'x'"
        
        # Not a valid numerical input
        else:
            return False, "Invalid numerical input format"
    
    elif input_type.lower() == "matrix":
        # Check if it's a numpy array
        if isinstance(formatted_input, np.ndarray):
            # Check dimensions - we need at least a 2x2 matrix
            if len(formatted_input.shape) == 2 and formatted_input.shape[0] > 1 and formatted_input.shape[1] > 1:
                return True, "Valid matrix input"
            else:
                return False, "Matrix must be at least 2x2"
        
        # Not a valid matrix
        else:
            return False, "Invalid matrix format"
    
    elif input_type.lower() == "text":
        # Check if it's a string with meaningful content
        if isinstance(formatted_input, str) and len(formatted_input.strip()) > 0:
            return True, "Valid text input"
        else:
            return False, "Text input cannot be empty"
    
    # Unknown input type
    return False, f"Unknown input type: {input_type}"

def extract_numeric_values(text: str) -> List[float]:
    """
    Extract numerical values from a text string.
    
    Args:
        text: Text string that may contain numbers
        
    Returns:
        List of extracted numerical values
    """
    # Match floating point numbers with optional sign
    number_pattern = r'[-+]?\d*\.?\d+'
    matches = re.findall(number_pattern, text)
    
    # Convert matched strings to float
    return [float(match) for match in matches]

def detect_paradox_type(input_data: Any) -> str:
    """
    Attempt to detect the type of paradox based on the input.
    
    Args:
        input_data: The paradox input
        
    Returns:
        String describing the detected paradox type
    """
    if isinstance(input_data, (int, float)):
        return "numerical"
    
    elif isinstance(input_data, np.ndarray):
        return "matrix"
    
    elif isinstance(input_data, str):
        # Check if it looks like an equation
        if "=" in input_data and any(c in input_data for c in "xyz"):
            return "numerical"
        
        # Check if it contains numbers
        elif re.search(r'\d', input_data):
            return "numerical"
        
        # Otherwise assume it's text
        else:
            return "text"
    
    elif isinstance(input_data, list):
        # Check if it's a list of numbers or nested list (matrix)
        if all(isinstance(x, (int, float)) for x in input_data):
            return "numerical"
        
        elif all(isinstance(x, list) for x in input_data):
            return "matrix"
    
    # Default case
    return "unknown"

def estimate_resolution_complexity(input_data: Any) -> int:
    """
    Estimate the computational complexity of resolving the given paradox.
    
    Args:
        input_data: The paradox input
        
    Returns:
        Integer from 1-10 indicating estimated complexity
    """
    # For numerical values
    if isinstance(input_data, (int, float)):
        # Simple heuristics based on value
        if input_data == 0:
            return 7  # Division by zero is challenging
        elif abs(input_data) < 0.001 or abs(input_data) > 1000:
            return 5  # Extreme values can be harder
        else:
            return 3  # Standard numerical values
    
    # For matrices
    elif isinstance(input_data, np.ndarray):
        # Complexity based on size and properties
        size = np.prod(input_data.shape)
        if size > 25:  # 5x5 or larger
            return 8
        elif size > 9:  # 3x3 or larger
            return 6
        else:
            return 4
    
    # For text
    elif isinstance(input_data, str):
        # Complexity based on length and type
        if "=" in input_data:  # Equations
            return 5
        elif len(input_data) > 200:  # Long text
            return 7
        else:
            return 4
    
    # Default complexity
    return 5

def format_resolution_output(final_state: Any, original_input_type: str) -> str:
    """
    Format the final resolution state for display.
    
    Args:
        final_state: The resolved state
        original_input_type: Type of the original input
        
    Returns:
        Formatted string representation
    """
    if original_input_type.lower() == "numerical":
        if isinstance(final_state, (int, float)):
            return f"{final_state:.6f}"
        else:
            return str(final_state)
    
    elif original_input_type.lower() == "matrix":
        if isinstance(final_state, np.ndarray):
            # Format matrix with clean precision
            return np.array2string(final_state, precision=4, suppress_small=True)
        else:
            return str(final_state)
    
    elif original_input_type.lower() == "text":
        return str(final_state)
    
    # Default format
    return str(final_state)