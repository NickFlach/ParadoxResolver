import numpy as np
import re
from typing import Any, Dict, Callable, Union, List

def get_available_rules() -> Dict[str, Callable]:
    """
    Returns a dictionary of all available transformation rules.
    
    Each rule is a function that takes a state and returns a transformed state.
    """
    return {
        "Fixed-Point Iteration": fixed_point_iteration,
        "Contradiction Resolution": contradiction_resolution,
        "Self-Reference Unwinding": self_reference_unwinding,
        "Eigenvalue Stabilization": eigenvalue_stabilization,
        "Fuzzy Logic Transformation": fuzzy_logic_transformation,
        "Duality Inversion": duality_inversion,
        "Bayesian Update": bayesian_update,
        "Recursive Normalization": recursive_normalization
    }

def fixed_point_iteration(state: Any) -> Any:
    """
    Apply fixed-point iteration to reach stable points for recursive equations.
    
    Works with numerical values and equations of the form x = f(x)
    """
    if isinstance(state, (int, float)):
        # For simple numerical values, apply a dampened averaging transformation
        # This helps achieve convergence for many recursive equations
        dampening_factor = 0.7
        inverse_value = 1.0 / state if state != 0 else 1.0
        return state * (1 - dampening_factor) + dampening_factor * inverse_value
    
    elif isinstance(state, str) and '=' in state:
        # For equation strings, we'll do a symbolic transformation
        # This is a simplified approach; a proper implementation would use sympy
        parts = state.split('=')
        if len(parts) == 2 and 'x' in parts[1]:
            # If it's a form like "x = 1/x", modify it to approach fixed point
            # Here we're simulating a mathematical transformation
            if "1/x" in parts[1]:
                return "x = (x + 1/x)/2"  # Averaging x and 1/x for better convergence
            elif "x^2" in parts[1] or "x**2" in parts[1]:
                return "x = (x + x^2/2)/2"  # Modified iteration for quadratic terms
            else:
                return state  # Return unchanged if we don't have a specific rule
        return state
    
    elif isinstance(state, np.ndarray):
        # For matrices, apply a specialized fixed-point iteration
        # This could be a form of power iteration or other matrix fixed-point algorithm
        if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
            # Square matrix case
            identity = np.eye(state.shape[0])
            transformed = 0.5 * (state + identity)  # Move towards identity matrix
            # Normalize to prevent explosion
            if np.max(np.abs(transformed)) > 0:
                transformed = transformed / np.max(np.abs(transformed))
            return transformed
    
    # Default: return state unchanged
    return state

def contradiction_resolution(state: Any) -> Any:
    """
    Transform logical contradictions into consistent states.
    
    Works with text-based paradoxes and boolean expressions.
    """
    if isinstance(state, str):
        # Handle classic text paradoxes
        if "this statement is false" in state.lower():
            # Transform to a non-contradictory statement about self-reference
            return "This statement refers to itself and has an undefined truth value."
        
        if "i am lying" in state.lower() or "all statements are false" in state.lower():
            # Transform to avoid the direct contradiction
            return "Some statements can be false while others are true."
        
        # Detect and resolve direct contradictions like "A and not A"
        contradiction_pattern = r'\b(\w+)\b.+\bnot\b.+\b\1\b'
        if re.search(contradiction_pattern, state.lower()):
            return "This contains mutually exclusive propositions that can be resolved through context-dependent evaluation."
    
    elif isinstance(state, dict) and 'value' in state and 'negation' in state:
        # For structured logical representations
        # If we have a direct contradiction where value == negation
        if state['value'] == state['negation']:
            # Introduce uncertainty/fuzziness
            return {
                'value': 0.5,
                'negation': 0.5,
                'uncertain': True
            }
    
    # Return state unchanged if not a recognized contradiction
    return state

def self_reference_unwinding(state: Any) -> Any:
    """
    Resolves self-referential statements or structures by unwinding one level.
    """
    if isinstance(state, str):
        # Handle self-referential text statements
        if "this statement" in state.lower() or "itself" in state.lower():
            # Unwind by making the self-reference explicit
            return "A statement about statements can be evaluated in a meta-linguistic framework."
        
        # For nested self-reference, simplify
        if state.count("refers to") > 1:
            return "Multiple levels of self-reference can be reduced to a single level plus context."
    
    elif isinstance(state, dict) and 'refers_to' in state:
        # For structured self-reference
        if state.get('refers_to') == 'self':
            # Unwind one level
            return {
                'content': state.get('content', ''),
                'refers_to': 'content',
                'level': state.get('level', 1) - 1
            }
    
    elif isinstance(state, list) and state and state[0] == state:
        # A list that contains itself as the first element
        # Create a new list without the self-reference
        return state[1:] if len(state) > 1 else []
    
    return state

def eigenvalue_stabilization(state: Any) -> Any:
    """
    Stabilizes matrix-based paradoxes using eigenvalue/eigenvector techniques.
    """
    if isinstance(state, np.ndarray) and len(state.shape) == 2:
        # Only apply to square matrices
        if state.shape[0] == state.shape[1]:
            try:
                # Calculate eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(state)
                
                # Find the dominant eigenvalue
                dominant_idx = np.argmax(np.abs(eigenvalues))
                dominant_value = eigenvalues[dominant_idx]
                
                # Normalize the eigenvalue for stability
                normalized_eigenvalues = eigenvalues / max(1, np.max(np.abs(eigenvalues)))
                
                # Reconstruct a more stable matrix
                reconstructed = np.zeros_like(state)
                for i, (val, vec) in enumerate(zip(normalized_eigenvalues, eigenvectors.T)):
                    # Project along eigenvectors but with normalized eigenvalues
                    outer_product = np.outer(vec, np.conj(vec))
                    reconstructed += val * outer_product
                
                # Ensure the matrix is real if it started real
                if np.isrealobj(state):
                    reconstructed = np.real(reconstructed)
                
                return reconstructed
            except np.linalg.LinAlgError:
                # If eigendecomposition fails, apply a simpler stabilization
                return 0.9 * state + 0.1 * np.eye(state.shape[0])
    
    return state

def fuzzy_logic_transformation(state: Any) -> Any:
    """
    Applies many-valued logic to resolve binary contradictions.
    """
    if isinstance(state, (int, float)) and 0 <= state <= 1:
        # Apply a sigmoidal transformation to push values away from 0.5
        # and toward 0 or 1, but never reaching the extremes
        if state == 0.5:
            # Pure contradiction remains unchanged
            return state
        
        # This function creates a soft threshold effect
        transformed = 1 / (1 + np.exp(-10 * (state - 0.5)))
        return transformed
    
    elif isinstance(state, str):
        # Apply fuzzy logic principles to text paradoxes
        if "true and false" in state.lower() or "contradiction" in state.lower():
            return "This can be understood as partially true in a fuzzy logic system where truth values exist on a spectrum."
    
    elif isinstance(state, dict) and 'truth_value' in state:
        # For explicit truth value representations
        truth = state['truth_value']
        if isinstance(truth, (int, float)) and 0 <= truth <= 1:
            # If it's close to contradictory (0.5), push it slightly away
            if 0.4 <= truth <= 0.6:
                direction = 1 if truth >= 0.5 else 0
                new_truth = 0.5 + (0.2 * (direction - 0.5))
                state['truth_value'] = new_truth
                state['fuzzy'] = True
            return state
    
    return state

def duality_inversion(state: Any) -> Any:
    """
    Resolves paradoxes by inverting dualities and finding the complementary perspective.
    """
    if isinstance(state, (int, float)):
        # Invert numerical values, with special handling for [0,1] range
        if 0 <= state <= 1:
            return 1 - state
        else:
            # For general numbers, apply a reciprocal transformation with dampening
            dampening = 0.3
            return (1 - dampening) * state + dampening * (1 / (1 + abs(state)))
    
    elif isinstance(state, str):
        # Invert dualistic concepts in text
        opposites = {
            "true": "false", "false": "true",
            "yes": "no", "no": "yes",
            "all": "some", "none": "some",
            "always": "sometimes", "never": "sometimes"
        }
        
        # Check for presence of opposites and replace with middle ground
        lower_state = state.lower()
        for word, opposite in opposites.items():
            if word in lower_state and opposite in lower_state:
                # Replace the dualistic opposition with a middle ground
                pattern = r'\b(' + word + r'|' + opposite + r')\b'
                middle_grounds = {"true/false": "contextual", "yes/no": "conditional", 
                              "all/some": "qualified", "always/never": "situational"}
                
                key = f"{word}/{opposite}"
                replacement = middle_grounds.get(key, "nuanced")
                
                return re.sub(pattern, replacement, state, flags=re.IGNORECASE)
    
    elif isinstance(state, np.ndarray):
        # For matrices, invert in a way that preserves structure
        # Use the complement relative to the identity matrix
        if len(state.shape) == 2 and state.shape[0] == state.shape[1]:
            identity = np.eye(state.shape[0])
            return identity - state * 0.5  # Dampened inversion
    
    return state

def bayesian_update(state: Any) -> Any:
    """
    Applies Bayesian inference principles to update probabilities in paradoxical states.
    """
    if isinstance(state, (int, float)) and 0 <= state <= 1:
        # Interpret as a probability and update with a prior of 0.5
        prior = 0.5
        # Simple Bayesian update formula with a neutral likelihood
        return (state * prior) / (state * prior + (1 - state) * (1 - prior))
    
    elif isinstance(state, list) and all(isinstance(x, (int, float)) for x in state):
        # For lists of probabilities, normalize
        if any(x < 0 for x in state):
            # Handle negative values by shifting
            min_val = min(state)
            shifted = [x - min_val for x in state]
            total = sum(shifted)
            return [x / total if total > 0 else 1.0 / len(state) for x in shifted]
        else:
            total = sum(state)
            return [x / total if total > 0 else 1.0 / len(state) for x in state]
    
    elif isinstance(state, dict) and all(isinstance(state.get(k), (int, float)) for k in state):
        # For dictionaries mapping to probabilities
        values = list(state.values())
        if all(0 <= x <= 1 for x in values):
            # Ensure they sum to 1 as valid probabilities
            total = sum(values)
            if total > 0:
                return {k: v / total for k, v in state.items()}
    
    return state

def recursive_normalization(state: Any) -> Any:
    """
    Normalizes values while preserving the structure of complex nested states.
    """
    if isinstance(state, (int, float)):
        # Simple sigmoid normalization for numerical values
        return 1 / (1 + np.exp(-state))
    
    elif isinstance(state, np.ndarray):
        # Normalize array values
        if np.size(state) > 0:
            abs_max = np.max(np.abs(state))
            if abs_max > 0:
                return state / abs_max
    
    elif isinstance(state, list):
        # Recursively normalize lists
        if all(isinstance(x, (int, float)) for x in state):
            # For numerical lists, scale to [0,1] range
            min_val = min(state) if state else 0
            max_val = max(state) if state else 1
            if max_val > min_val:
                return [(x - min_val) / (max_val - min_val) for x in state]
            else:
                return [0.5 for _ in state]
        else:
            # For mixed type lists, recursively apply normalization
            return [recursive_normalization(item) for item in state]
    
    elif isinstance(state, dict):
        # Recursively normalize dictionary values
        return {k: recursive_normalization(v) for k, v in state.items()}
    
    return state
