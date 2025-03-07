import numpy as np
import re
from typing import Any, Tuple, Union, Dict, List, Optional

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
    if input_type == "Text":
        # Return string as is
        return paradox_input.strip()
    
    elif input_type == "Numerical":
        # Check if it's an equation or just a value
        if isinstance(paradox_input, (int, float)):
            return float(paradox_input)
        
        if "=" in paradox_input:
            # Return the equation string
            return paradox_input.strip()
        else:
            # Try to evaluate as a numerical expression
            try:
                return float(eval(paradox_input))
            except:
                # If evaluation fails, return the initial value
                return float(initial_value) if initial_value is not None else 0.5
    
    elif input_type == "Matrix":
        # Ensure it's a numpy array
        if isinstance(paradox_input, np.ndarray):
            return paradox_input
        else:
            # Convert to numpy array if possible
            try:
                return np.array(paradox_input, dtype=float)
            except:
                # Return identity matrix as fallback
                dimension = 2  # Default dimension
                return np.eye(dimension)
    
    # Default return
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
    if input_type == "Text":
        if not isinstance(formatted_input, str):
            return False, "Input must be a text string."
        
        if len(formatted_input) < 3:
            return False, "Input text is too short to be a paradox."
        
        return True, "Valid text input."
    
    elif input_type == "Numerical":
        if isinstance(formatted_input, (int, float)):
            return True, "Valid numerical input."
        
        if isinstance(formatted_input, str):
            if "=" in formatted_input:
                # Check if it's a well-formed equation
                parts = formatted_input.split("=")
                if len(parts) != 2:
                    return False, "Equation should have exactly one equals sign."
                
                # Check if both sides have some content
                if not parts[0].strip() or not parts[1].strip():
                    return False, "Both sides of the equation must have content."
                
                return True, "Valid equation input."
            else:
                # Should be a numerical expression
                return False, "Invalid numerical expression. Use an equation with '=' or a numeric value."
        
        return False, "Numerical input must be a number or an equation."
    
    elif input_type == "Matrix":
        if not isinstance(formatted_input, np.ndarray):
            return False, "Matrix input must be a numpy array."
        
        if len(formatted_input.shape) != 2:
            return False, "Input must be a 2D matrix."
        
        if formatted_input.shape[0] != formatted_input.shape[1]:
            return False, "Matrix must be square (same number of rows and columns)."
        
        return True, "Valid matrix input."
    
    return False, "Unrecognized input type."

def extract_numeric_values(text: str) -> List[float]:
    """
    Extract numerical values from a text string.
    
    Args:
        text: Text string that may contain numbers
        
    Returns:
        List of extracted numerical values
    """
    # Find all numbers in the text
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    # Convert matches to floats
    values = [float(match) for match in matches]
    
    return values

def detect_paradox_type(input_data: Any) -> str:
    """
    Attempt to detect the type of paradox based on the input.
    
    Args:
        input_data: The paradox input
        
    Returns:
        String describing the detected paradox type
    """
    if isinstance(input_data, str):
        # Check for common text paradox patterns
        lower_text = input_data.lower()
        
        if any(term in lower_text for term in ["this statement", "is false", "lying", "liar"]):
            return "Liar Paradox"
        
        if "all" in lower_text and "not all" in lower_text:
            return "Universal Quantification Paradox"
        
        if any(term in lower_text for term in ["infinite", "recursion", "recursive"]):
            return "Infinite Recursion Paradox"
        
        if "sorites" in lower_text or ("heap" in lower_text and "grain" in lower_text):
            return "Sorites Paradox"
        
        if "zeno" in lower_text or ("achilles" in lower_text and "tortoise" in lower_text):
            return "Zeno's Paradox"
        
        # Default text paradox
        return "Linguistic Paradox"
    
    elif isinstance(input_data, (int, float)):
        return "Numerical Paradox"
    
    elif isinstance(input_data, np.ndarray):
        if len(input_data.shape) == 2 and input_data.shape[0] == input_data.shape[1]:
            # Check for specific matrix properties
            try:
                eigenvalues = np.linalg.eigvals(input_data)
                if np.any(np.abs(eigenvalues) > 1):
                    return "Unstable Dynamical System"
                else:
                    return "Stable Matrix Paradox"
            except:
                return "Matrix Paradox"
    
    return "Unknown Paradox Type"

def estimate_resolution_complexity(input_data: Any) -> int:
    """
    Estimate the computational complexity of resolving the given paradox.
    
    Args:
        input_data: The paradox input
        
    Returns:
        Integer from 1-10 indicating estimated complexity
    """
    if isinstance(input_data, str):
        # Estimate based on text length and complexity
        complexity = min(10, 1 + (len(input_data) // 50) + input_data.count("if") + input_data.count("="))
    
    elif isinstance(input_data, (int, float)):
        # Simple numerical paradoxes are typically less complex
        complexity = 3
    
    elif isinstance(input_data, np.ndarray):
        # Matrix complexity based on size and properties
        n = input_data.shape[0]
        complexity = min(10, 1 + n + (3 if np.count_nonzero(input_data) / input_data.size > 0.5 else 0))
    
    else:
        # Default complexity
        complexity = 5
    
    return max(1, min(complexity, 10))

def format_resolution_output(final_state: Any, original_input_type: str) -> str:
    """
    Format the final resolution state for display.
    
    Args:
        final_state: The resolved state
        original_input_type: Type of the original input
        
    Returns:
        Formatted string representation
    """
    if isinstance(final_state, (int, float)):
        return f"{final_state:.6f}"
    
    elif isinstance(final_state, np.ndarray):
        if len(final_state.shape) == 2:
            # Format 2D matrix
            result = "Matrix:\n"
            for row in final_state:
                result += "[" + " ".join([f"{val:.4f}" for val in row]) + "]\n"
            return result
        else:
            # Format other array types
            return np.array2string(final_state, precision=4, suppress_small=True)
    
    elif isinstance(final_state, dict):
        # Format dictionary results
        return "\n".join([f"{k}: {v}" for k, v in final_state.items()])
    
    # Default string representation
    return str(final_state)
