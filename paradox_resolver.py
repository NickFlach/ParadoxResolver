import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Union
import copy

class ParadoxResolver:
    """
    Core class for resolving paradoxes through recursive transformation.
    
    This class applies a set of transformation rules iteratively until
    the paradox reaches equilibrium (convergence) or max iterations.
    """
    
    def __init__(
        self, 
        transformation_rules: Dict[str, Callable],
        max_iterations: int = 20,
        convergence_threshold: float = 0.001
    ):
        """
        Initialize the paradox resolver.
        
        Args:
            transformation_rules: Dictionary of named transformation functions
            max_iterations: Maximum number of iterations to perform
            convergence_threshold: Threshold for determining convergence
        """
        self.transformation_rules = transformation_rules
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def resolve(self, initial_state: Any) -> Tuple[Any, List[Any], bool]:
        """
        Resolve a paradox by applying transformation rules recursively.
        
        Args:
            initial_state: The initial paradoxical state
            
        Returns:
            Tuple containing:
            - Final state after resolution
            - List of all intermediate states
            - Boolean indicating if convergence was reached
        """
        current_state = copy.deepcopy(initial_state)
        steps = [copy.deepcopy(current_state)]
        converged = False
        
        for iteration in range(self.max_iterations):
            # Apply each transformation rule in sequence
            new_state = copy.deepcopy(current_state)
            
            for rule_name, rule_function in self.transformation_rules.items():
                new_state = rule_function(new_state)
            
            # Store the new state
            steps.append(copy.deepcopy(new_state))
            
            # Check for convergence
            if self._check_convergence(current_state, new_state):
                converged = True
                break
                
            current_state = new_state
        
        return current_state, steps, converged
    
    def _check_convergence(self, previous_state: Any, current_state: Any) -> bool:
        """
        Check if the paradox resolution has converged.
        
        Args:
            previous_state: State from the previous iteration
            current_state: Current state after applying transformations
            
        Returns:
            True if convergence criteria met, False otherwise
        """
        # Handle different types of states
        if isinstance(current_state, (int, float)):
            if isinstance(previous_state, (int, float)):
                return abs(current_state - previous_state) < self.convergence_threshold
            return False
            
        elif isinstance(current_state, np.ndarray):
            if isinstance(previous_state, np.ndarray) and current_state.shape == previous_state.shape:
                diff = np.abs(current_state - previous_state)
                return np.all(diff < self.convergence_threshold)
            return False
            
        elif isinstance(current_state, dict):
            if isinstance(previous_state, dict) and set(current_state.keys()) == set(previous_state.keys()):
                return all(
                    self._check_convergence(previous_state[k], current_state[k]) 
                    for k in current_state
                )
            return False
            
        elif isinstance(current_state, (list, tuple)) and all(isinstance(x, (int, float)) for x in current_state):
            if (isinstance(previous_state, (list, tuple)) and 
                len(current_state) == len(previous_state) and
                all(isinstance(x, (int, float)) for x in previous_state)):
                diffs = [abs(a - b) for a, b in zip(current_state, previous_state)]
                return all(diff < self.convergence_threshold for diff in diffs)
            return False
            
        # For string-based paradoxes, check exact equality
        elif isinstance(current_state, str) and isinstance(previous_state, str):
            return current_state == previous_state
            
        # Default case for complex or custom objects
        return current_state == previous_state
