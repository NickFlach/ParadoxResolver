"""
Crypto_ParadoxOS Integration for SIN's Reasoning Engine

This module provides specialized integration with SIN's reasoning engines,
enabling them to leverage paradox resolution for complex logical problems.
"""

import sys
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from pathlib import Path

# Ensure access to core integration module
sys.path.append(str(Path(__file__).parent.parent))

from common.integration_core import ParadoxIntegration, IntegrationConfig

class LogicalStatement:
    """Represents a logical statement for integration with SIN's reasoning system."""
    
    def __init__(self, 
                statement: str,
                truth_value: float = 0.5,  # 0.0 = false, 1.0 = true, 0.5 = uncertain
                dependencies: List[str] = None,
                contradictions: List[str] = None,
                metadata: Dict[str, Any] = None):
        """Initialize logical statement."""
        self.statement = statement
        self.truth_value = truth_value
        self.dependencies = dependencies or []
        self.contradictions = contradictions or []
        self.metadata = metadata or {}
    
    def __str__(self):
        """String representation of the statement."""
        return f"{self.statement} [{self.truth_value:.2f}]"
    
    def __repr__(self):
        """Detailed representation of the statement."""
        return (f"LogicalStatement('{self.statement}', truth_value={self.truth_value}, "
                f"dependencies={self.dependencies}, contradictions={self.contradictions})")


class LogicalSystem:
    """Represents a system of logical statements for paradox resolution."""
    
    def __init__(self, statements: Dict[str, LogicalStatement] = None):
        """Initialize logical system."""
        self.statements = statements or {}
    
    def add_statement(self, key: str, statement: LogicalStatement) -> None:
        """Add a statement to the system."""
        self.statements[key] = statement
    
    def get_statement(self, key: str) -> Optional[LogicalStatement]:
        """Get a statement from the system."""
        return self.statements.get(key)
    
    def to_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Convert the logical system to a matrix representation for paradox resolution.
        
        Returns:
            Tuple of (matrix, statement_keys)
        """
        if not self.statements:
            return np.array([[0.5]]), ["empty"]
        
        # Get all statement keys
        keys = list(self.statements.keys())
        n = len(keys)
        
        # Create a matrix where:
        # - Diagonal elements are truth values
        # - Off-diagonal elements represent relationships:
        #   - Positive value for dependency
        #   - Negative value for contradiction
        #   - Zero for no relationship
        matrix = np.zeros((n, n))
        
        # Fill truth values on diagonal
        for i, key in enumerate(keys):
            matrix[i, i] = self.statements[key].truth_value
        
        # Fill relationships
        for i, src_key in enumerate(keys):
            src = self.statements[src_key]
            
            # Dependencies (positive values)
            for dep_key in src.dependencies:
                if dep_key in keys:
                    j = keys.index(dep_key)
                    matrix[i, j] = 0.5  # Positive dependency
            
            # Contradictions (negative values)
            for contra_key in src.contradictions:
                if contra_key in keys:
                    j = keys.index(contra_key)
                    matrix[i, j] = -0.5  # Negative relationship
        
        return matrix, keys
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, keys: List[str], original: 'LogicalSystem' = None) -> 'LogicalSystem':
        """
        Create a logical system from a matrix representation.
        
        Args:
            matrix: Matrix representation
            keys: Statement keys corresponding to matrix indices
            original: Original logical system to pull non-numeric data from
            
        Returns:
            New LogicalSystem instance
        """
        system = LogicalSystem()
        
        # If original system is provided, copy non-numeric data
        if original is not None:
            for key, statement in original.statements.items():
                # Create a copy of the statement
                system.statements[key] = LogicalStatement(
                    statement=statement.statement,
                    dependencies=statement.dependencies.copy(),
                    contradictions=statement.contradictions.copy(),
                    metadata=statement.metadata.copy()
                )
        
        # Update truth values from matrix diagonal
        for i, key in enumerate(keys):
            if i < matrix.shape[0]:
                # If statement already exists, update it
                if key in system.statements:
                    system.statements[key].truth_value = float(matrix[i, i])
                # Otherwise create a new one
                else:
                    system.statements[key] = LogicalStatement(
                        statement=key,
                        truth_value=float(matrix[i, i])
                    )
                
                # Update dependencies and contradictions if original not provided
                if original is None and matrix.shape[1] > 1:
                    deps = []
                    contras = []
                    
                    for j, other_key in enumerate(keys):
                        if i != j:  # Skip self
                            if matrix[i, j] > 0.1:
                                deps.append(other_key)
                            elif matrix[i, j] < -0.1:
                                contras.append(other_key)
                    
                    system.statements[key].dependencies = deps
                    system.statements[key].contradictions = contras
        
        return system


class ReasoningIntegration(ParadoxIntegration):
    """
    Specialized integration for SIN's reasoning engine.
    
    This integration enables the reasoning engine to use paradox resolution for
    dealing with logical contradictions and complex reasoning problems.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize reasoning integration."""
        super().__init__(config)
        
        # Register specialized transformation rules for logical reasoning
        self._register_reasoning_rules()
        
        self.logger.info("Initialized Reasoning integration with specialized logical rules")
    
    def _register_reasoning_rules(self):
        """Register specialized transformation rules for logical reasoning."""
        
        def truth_value_adjustment(state: Any) -> Any:
            """Adjust truth values based on logical relationships."""
            if isinstance(state, np.ndarray):
                n = state.shape[0]
                result = state.copy()
                
                # Extract truth values from diagonal
                truth_values = np.array([result[i, i] for i in range(n)])
                
                # Adjust based on dependencies and contradictions
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            # Positive relationship (dependency)
                            if result[i, j] > 0.1:
                                # If A depends on B, and B has low truth, reduce A's truth
                                if truth_values[j] < 0.4:
                                    # The more A depends on B, the more it should be reduced
                                    reduction = result[i, j] * (0.5 - truth_values[j])
                                    result[i, i] = max(0.0, result[i, i] - reduction)
                            
                            # Negative relationship (contradiction)
                            elif result[i, j] < -0.1:
                                # If A contradicts B, their truth values should be inversely related
                                # The more certain B is, the less certain A should be of its opposite value
                                if truth_values[j] > 0.7:
                                    # Move truth value toward opposite of contradicted statement
                                    result[i, i] = (result[i, i] * 0.7 + (1.0 - truth_values[j]) * 0.3)
                
                return result
            return state
        
        def consistency_enforcement(state: Any) -> Any:
            """Enforce logical consistency in the system."""
            if isinstance(state, np.ndarray):
                n = state.shape[0]
                result = state.copy()
                
                # First pass: detect direct contradictions
                for i in range(n):
                    for j in range(n):
                        if i != j and result[i, j] < -0.1:
                            # A and B contradict each other
                            
                            # If both have high truth values, move them toward uncertainty
                            if result[i, i] > 0.7 and result[j, j] > 0.7:
                                # Move both toward uncertainty
                                result[i, i] = 0.6
                                result[j, j] = 0.6
                            
                            # If one is much more certain than the other, adjust the less certain one
                            elif abs(result[i, i] - result[j, j]) > 0.4:
                                if result[i, i] > result[j, j]:
                                    # A is more certain, so adjust B toward its logical opposite
                                    result[j, j] = min(result[j, j], 1.0 - result[i, i] + 0.1)
                                else:
                                    # B is more certain, so adjust A toward its logical opposite
                                    result[i, i] = min(result[i, i], 1.0 - result[j, j] + 0.1)
                
                return result
            return state
        
        def multi_valued_logic_transformation(state: Any) -> Any:
            """Apply multi-valued logic to overcome binary limitations."""
            if isinstance(state, np.ndarray):
                n = state.shape[0]
                result = state.copy()
                
                # Apply fuzzy logic principles to truth values
                for i in range(n):
                    # Move extreme values slightly toward center
                    if result[i, i] > 0.95:
                        result[i, i] = 0.95
                    elif result[i, i] < 0.05:
                        result[i, i] = 0.05
                    
                    # For values in uncertainty region, look at network of relationships
                    if 0.4 <= result[i, i] <= 0.6:
                        # Count supporting and contradicting relationships
                        support = 0
                        contradict = 0
                        
                        for j in range(n):
                            if i != j:
                                if result[i, j] > 0.1:  # Dependency
                                    support += result[i, j] * result[j, j]
                                elif result[i, j] < -0.1:  # Contradiction
                                    contradict += -result[i, j] * result[j, j]
                        
                        # Adjust based on network support/contradiction
                        if support > contradict:
                            result[i, i] = min(0.7, result[i, i] + 0.1)
                        elif contradict > support:
                            result[i, i] = max(0.3, result[i, i] - 0.1)
                
                return result
            return state
        
        def transitive_inference(state: Any) -> Any:
            """Apply transitive inference to logical relationships."""
            if isinstance(state, np.ndarray):
                n = state.shape[0]
                result = state.copy()
                
                # Look for transitive relationships: if A depends on B and B depends on C,
                # then A should depend somewhat on C
                for i in range(n):
                    for j in range(n):
                        if i != j and result[i, j] > 0.1:  # A depends on B
                            for k in range(n):
                                if j != k and k != i and result[j, k] > 0.1:  # B depends on C
                                    # Establish transitive relationship if none exists
                                    if abs(result[i, k]) < 0.1:
                                        result[i, k] = 0.3  # Weaker than direct dependency
                                    # If contradictory relationship exists, weaken it
                                    elif result[i, k] < 0:
                                        result[i, k] = result[i, k] * 0.5
                
                # Similarly for contradictions: if A contradicts B and B depends on C,
                # then A should somewhat contradict C
                for i in range(n):
                    for j in range(n):
                        if i != j and result[i, j] < -0.1:  # A contradicts B
                            for k in range(n):
                                if j != k and k != i and result[j, k] > 0.1:  # B depends on C
                                    # Establish transitive contradiction if none exists
                                    if abs(result[i, k]) < 0.1:
                                        result[i, k] = -0.2  # Weaker than direct contradiction
                
                return result
            return state
        
        # Register the specialized reasoning rules
        self.api.register_rule("Truth Value Adjustment", truth_value_adjustment, 
                              "Adjusts truth values based on logical relationships")
        self.api.register_rule("Consistency Enforcement", consistency_enforcement,
                              "Enforces logical consistency within the system")
        self.api.register_rule("Multi-Valued Logic", multi_valued_logic_transformation,
                              "Applies multi-valued logic to overcome binary limitations")
        self.api.register_rule("Transitive Inference", transitive_inference,
                              "Applies transitive inference to logical relationships")
    
    def resolve_contradictions(self, 
                              logical_system: LogicalSystem,
                              max_uncertainty: float = 0.3,
                              prioritize_statements: List[str] = None) -> LogicalSystem:
        """
        Resolve contradictions in a logical system.
        
        Args:
            logical_system: System of logical statements
            max_uncertainty: Maximum acceptable uncertainty in truth values
            prioritize_statements: List of statement keys to prioritize
            
        Returns:
            New LogicalSystem with resolved contradictions
        """
        self.logger.info(f"Resolving contradictions in system with {len(logical_system.statements)} statements")
        
        # Convert to matrix for processing
        matrix, keys = logical_system.to_matrix()
        
        # Create custom config with parameters
        custom_config = {
            "max_uncertainty": max_uncertainty,
            "prioritize_statements": prioritize_statements or [],
            "max_iterations": self.config.max_iterations,
            "convergence_threshold": self.config.convergence_threshold
        }
        
        # Resolve the logical system as a paradox
        result = self.resolve_paradox(matrix, "logical_matrix", custom_config=custom_config)
        
        # Extract the final state and convert back to logical system
        final_matrix = result.get("final_state", matrix)
        
        if not isinstance(final_matrix, np.ndarray):
            self.logger.warning("Unexpected result type, falling back to original logical system")
            final_matrix = matrix
        
        # Create new logical system from the transformed matrix
        resolved_system = LogicalSystem.from_matrix(final_matrix, keys, original=logical_system)
        
        return resolved_system
    
    def generate_consistent_extensions(self, 
                                     base_system: LogicalSystem,
                                     new_statements: Dict[str, LogicalStatement],
                                     count: int = 3) -> List[LogicalSystem]:
        """
        Generate consistent extensions of a logical system with new statements.
        
        Args:
            base_system: Base logical system
            new_statements: New statements to integrate
            count: Number of different consistent extensions to generate
            
        Returns:
            List of extended LogicalSystem instances
        """
        self.logger.info(f"Generating {count} consistent extensions with {len(new_statements)} new statements")
        
        # Create a combined system with all statements
        combined = LogicalSystem()
        
        # Add base statements
        for key, statement in base_system.statements.items():
            combined.add_statement(key, statement)
        
        # Add new statements
        for key, statement in new_statements.items():
            combined.add_statement(key, statement)
        
        # Convert to matrix
        matrix, keys = combined.to_matrix()
        
        # Generate variations
        extensions = []
        
        for i in range(count):
            # Apply small random perturbations to truth values of new statements
            perturbed = matrix.copy()
            
            # Identify indices of new statements
            new_indices = [keys.index(key) for key in new_statements.keys() if key in keys]
            
            # Apply perturbations to just the truth values (diagonal elements)
            for idx in new_indices:
                # Add small random variation, keeping within [0, 1]
                perturbed[idx, idx] = np.clip(perturbed[idx, idx] + np.random.uniform(-0.2, 0.2), 0, 1)
            
            # Resolve each perturbed system
            custom_config = {
                "variation_index": i,
                "enforce_consistency": True
            }
            
            result = self.resolve_paradox(perturbed, "logical_matrix", custom_config=custom_config)
            
            # Extract the final state and convert to logical system
            final_matrix = result.get("final_state", perturbed)
            extension = LogicalSystem.from_matrix(final_matrix, keys, original=combined)
            extensions.append(extension)
        
        return extensions
    
    def infer_missing_relationships(self, 
                                  system: LogicalSystem,
                                  relationship_threshold: float = 0.3) -> LogicalSystem:
        """
        Infer missing logical relationships in a system.
        
        Args:
            system: Logical system
            relationship_threshold: Threshold for establishing new relationships
            
        Returns:
            Enhanced LogicalSystem with inferred relationships
        """
        self.logger.info(f"Inferring missing relationships with threshold {relationship_threshold}")
        
        # Convert to matrix
        matrix, keys = system.to_matrix()
        
        # Use meta-resolution to enable more complex inference
        custom_config = {
            "relationship_threshold": relationship_threshold,
            "inference_mode": True
        }
        
        # Use meta-resolver for this complex task
        result = self.resolve_paradox(matrix, "logical_matrix", use_meta=True, custom_config=custom_config)
        
        # Extract the final state and convert to logical system
        final_matrix = result.get("final_state", matrix)
        
        # Create new system with inferred relationships
        enhanced = LogicalSystem.from_matrix(final_matrix, keys)
        
        # Copy statements from original system
        for key, statement in system.statements.items():
            if key in enhanced.statements:
                # Update with original text and metadata
                enhanced.statements[key].statement = statement.statement
                enhanced.statements[key].metadata = statement.metadata.copy()
        
        return enhanced
    
    def calculate_belief_network(self, 
                               system: LogicalSystem) -> Dict[str, Dict[str, float]]:
        """
        Calculate belief influence network for a logical system.
        
        Args:
            system: Logical system
            
        Returns:
            Dictionary mapping statement keys to dictionaries of influence values
        """
        # Convert to matrix
        matrix, keys = system.to_matrix()
        
        # Initialize belief network
        belief_network = {}
        
        for i, key in enumerate(keys):
            influence_dict = {}
            
            # For each other statement, calculate its influence
            for j, other_key in enumerate(keys):
                if i != j:
                    # Direct relationship
                    direct = matrix[i, j]
                    
                    # Create paths for indirect influences
                    indirect = 0.0
                    for k in range(len(keys)):
                        if k != i and k != j:
                            # Contribution via k is product of relationships
                            indirect += matrix[i, k] * matrix[k, j] * 0.5
                    
                    # Combine direct and indirect influence
                    total_influence = direct + indirect * 0.5
                    
                    # Only include non-trivial influences
                    if abs(total_influence) > 0.1:
                        influence_dict[other_key] = float(total_influence)
            
            belief_network[key] = influence_dict
        
        return belief_network


# Factory function to create reasoning integration
def create_reasoning_integration(**config_kwargs) -> ReasoningIntegration:
    """Create a configured reasoning integration instance."""
    config = IntegrationConfig("SIN", "Reasoning", **config_kwargs)
    return ReasoningIntegration(config)