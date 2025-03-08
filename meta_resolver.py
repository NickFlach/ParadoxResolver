#!/usr/bin/env python3
"""
Crypto_ParadoxOS Meta-Resolver

This module provides a meta-framework for resolving the core paradox between
recursive resolution (convergence) and informational expansion (divergence).
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

from paradox_resolver import ParadoxResolver
from transformation_rules import get_available_rules

class ResolutionPhase:
    """Represents a phase in the meta-resolution process."""
    
    def __init__(self, 
                 name: str,
                 is_convergent: bool = True,
                 max_iterations: int = 10,
                 threshold: float = 0.001):
        """
        Initialize a resolution phase.
        
        Args:
            name: Name of the phase
            is_convergent: Whether this phase aims for convergence (True) or expansion (False)
            max_iterations: Maximum iterations for this phase
            threshold: Convergence/divergence threshold
        """
        self.name = name
        self.is_convergent = is_convergent
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.rules = []
        self.transitions = {}  # Maps target phase name to condition function
    
    def add_rule(self, rule_name: str) -> 'ResolutionPhase':
        """Add a rule to this phase and return self for chaining."""
        self.rules.append(rule_name)
        return self
    
    def add_transition(self, 
                      target_phase: str, 
                      condition: Callable[[Any], bool]) -> 'ResolutionPhase':
        """
        Add a transition condition to another phase.
        
        Args:
            target_phase: Name of the target phase
            condition: Function that takes a state and returns True if 
                       transition should occur
        
        Returns:
            Self for method chaining
        """
        self.transitions[target_phase] = condition
        return self

class MetaResolver:
    """
    Meta-framework for resolving the paradox between recursive resolution and expansion.
    
    This class orchestrates the resolution process by dynamically switching between
    different phases based on state characteristics and transition conditions.
    """
    
    def __init__(self, api = None):
        """
        Initialize the meta-resolver.
        
        Args:
            api: API instance (unused in simplified implementation)
        """
        self.phases = {}
        self.initial_phase = None
        self.all_rules = get_available_rules()
    
    def add_phase(self, phase: ResolutionPhase) -> 'MetaResolver':
        """
        Add a resolution phase to the meta-resolver.
        
        Args:
            phase: ResolutionPhase instance
            
        Returns:
            Self for method chaining
        """
        self.phases[phase.name] = phase
        
        # Set as initial phase if it's the first one
        if self.initial_phase is None:
            self.initial_phase = phase.name
            
        return self
    
    def set_initial_phase(self, phase_name: str) -> 'MetaResolver':
        """
        Set the initial phase for the resolution process.
        
        Args:
            phase_name: Name of the phase to start with
            
        Returns:
            Self for method chaining
        """
        if phase_name in self.phases:
            self.initial_phase = phase_name
        else:
            raise ValueError(f"Phase '{phase_name}' does not exist")
            
        return self
    
    def resolve(self, 
               initial_state: Any, 
               input_type: str,
               max_phase_transitions: int = 10,
               max_total_iterations: int = 100) -> Dict[str, Any]:
        """
        Resolve a paradox using the meta-framework approach.
        
        This method orchestrates the resolution process by:
        1. Starting with the initial phase
        2. Applying transformation rules according to phase configuration
        3. Checking transition conditions and switching phases as needed
        4. Continuing until convergence, max iterations, or max transitions
        
        Args:
            initial_state: The initial paradoxical state
            input_type: Type of the input ("numerical", "matrix", etc.)
            max_phase_transitions: Maximum number of phase transitions
            max_total_iterations: Maximum total iterations across all phases
            
        Returns:
            Dictionary containing resolution results with phase information
        """
        # Initialize
        current_state = initial_state
        current_phase_name = self.initial_phase
        phase_history = [current_phase_name]
        state_history = [current_state]
        total_iterations = 0
        phase_transitions = 0
        converged = False
        
        # Main resolution loop
        while phase_transitions < max_phase_transitions and total_iterations < max_total_iterations:
            # Get the current phase
            current_phase = self.phases[current_phase_name]
            
            # Select rules for this phase
            selected_rules = {
                name: self.all_rules[name] 
                for name in current_phase.rules 
                if name in self.all_rules
            }
            
            if not selected_rules:
                # Fallback to all rules if none specified
                selected_rules = self.all_rules
            
            # Create a resolver for this phase
            resolver = ParadoxResolver(
                transformation_rules=selected_rules,
                max_iterations=min(current_phase.max_iterations, max_total_iterations - total_iterations),
                convergence_threshold=current_phase.threshold
            )
            
            # Execute this phase
            result, steps, phase_converged = resolver.resolve(current_state)
            converged = phase_converged
            
            # Update state and history
            current_state = result
            state_history.extend(steps[1:])  # Skip first step (duplicate of current_state)
            total_iterations += len(steps) - 1
            
            # Check if we've reached the max iterations
            if total_iterations >= max_total_iterations:
                break
                
            # Check for transitions to other phases
            next_phase = None
            for target, condition in current_phase.transitions.items():
                if condition(current_state):
                    next_phase = target
                    break
            
            # If no transition conditions met
            if next_phase is None:
                # If converged or diverged as expected, we're done
                if phase_converged == current_phase.is_convergent:
                    break
                    
                # Otherwise, stay in the same phase for another cycle
                # Add a safety check to prevent infinite loops
                if len(steps) <= 2:  # If no progress was made (only initial state + 1 iteration)
                    break
                continue
            
            # Move to the next phase
            current_phase_name = next_phase
            phase_history.append(current_phase_name)
            phase_transitions += 1
        
        # Prepare phase results
        phase_results = []
        for phase_name in phase_history:
            phase_results.append({
                "phase": phase_name,
                "iterations": self.phases[phase_name].max_iterations,
                "is_convergent": self.phases[phase_name].is_convergent,
                "type": "Convergent" if self.phases[phase_name].is_convergent else "Divergent"
            })
        
        # Create result dictionary
        result = {
            "final_state": current_state,
            "total_iterations": total_iterations,
            "phase_transitions": phase_transitions,
            "phase_history": phase_history,
            "phase_results": phase_results,  # Add phase results for visualization
            "converged": converged
        }
        
        return result
    
    def create_standard_framework(self) -> 'MetaResolver':
        """
        Create a standard meta-resolution framework with predefined phases.
        
        This provides a balanced approach that alternates between convergence
        and expansion to resolve the core paradox.
        
        Returns:
            Self with configured phases
        """
        # Define phase transition conditions
        def near_convergence(state: Any) -> bool:
            """Detect if we're approaching convergence."""
            if isinstance(state, (int, float)):
                # Near a fixed point (1 or 0 in many cases)
                return abs(state - round(state)) < 0.1
            elif isinstance(state, np.ndarray):
                # Near an identity, zero, or stable matrix
                return np.max(np.abs(state)) < 0.1 or np.max(np.abs(state - np.eye(*state.shape))) < 0.1
            return False
        
        def sufficient_expansion(state: Any) -> bool:
            """Detect if we've expanded enough."""
            if isinstance(state, (int, float)):
                # Values expanding beyond typical range
                return abs(state) > 2.0
            elif isinstance(state, np.ndarray):
                # Matrix elements expanding
                return np.max(np.abs(state)) > 2.0
            return True  # Default to allowing transition
        
        def emerging_stability(state: Any) -> bool:
            """Detect if stability is emerging."""
            # This is a simplified check
            if isinstance(state, (int, float)):
                # Values in middle range often indicate stabilization
                return 0.3 < abs(state) < 0.7
            return True
        
        # Create phases
        convergent = ResolutionPhase(
            name="convergent",
            is_convergent=True,
            max_iterations=10,
            threshold=0.001
        )
        convergent.add_rule("fixed_point_iteration")
        convergent.add_rule("eigenvalue_stabilization")
        convergent.add_rule("recursive_normalization")
        convergent.add_transition("divergent", near_convergence)
        
        divergent = ResolutionPhase(
            name="divergent",
            is_convergent=False,
            max_iterations=5,
            threshold=0.001
        )
        divergent.add_rule("duality_inversion")
        divergent.add_rule("contradiction_resolution")
        divergent.add_rule("self_reference_unwinding")
        divergent.add_transition("exploration", sufficient_expansion)
        
        exploration = ResolutionPhase(
            name="exploration",
            is_convergent=False,
            max_iterations=8,
            threshold=0.001
        )
        exploration.add_rule("fuzzy_logic_transformation")
        exploration.add_rule("bayesian_update")
        exploration.add_transition("convergent", emerging_stability)
        
        # Add phases to resolver
        self.add_phase(convergent)
        self.add_phase(divergent)
        self.add_phase(exploration)
        
        # Set initial phase
        self.set_initial_phase("convergent")
        
        return self