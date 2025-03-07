#!/usr/bin/env python3
"""
Crypto_ParadoxOS Optimizer

This module extends the core Crypto_ParadoxOS system with specialized functionality
for funding allocation, governance, and decision-making optimization.
"""

import numpy as np
import time
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

# Import core functionality
from crypto_paradox_os import ParadoxResolver, ParadoxState, TransformationRule
from crypto_paradox_os import get_standard_rules
from crypto_paradox_api import CryptoParadoxAPI


class Resource:
    """Represents a resource with attributes and constraints."""
    
    def __init__(self, 
                 name: str,
                 total: float,
                 min_allocation: float = 0.0,
                 max_allocation: Optional[float] = None):
        """
        Initialize a resource.
        
        Args:
            name: Resource name
            total: Total amount of the resource
            min_allocation: Minimum allocation required
            max_allocation: Maximum allocation allowed (None for no limit)
        """
        self.name = name
        self.total = total
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation if max_allocation is not None else total
    
    def __repr__(self) -> str:
        return f"Resource({self.name}, total={self.total})"


class Stakeholder:
    """Represents a stakeholder with preferences and influence."""
    
    def __init__(self, 
                 name: str,
                 influence: float = 1.0,
                 preferences: Dict[str, float] = None):
        """
        Initialize a stakeholder.
        
        Args:
            name: Stakeholder name
            influence: Relative influence weight (0.0 to 1.0)
            preferences: Mapping of resource names to preference weights
        """
        self.name = name
        self.influence = max(0.0, min(1.0, influence))  # Clamp to [0, 1]
        self.preferences = preferences or {}
    
    def __repr__(self) -> str:
        return f"Stakeholder({self.name}, influence={self.influence})"


class AllocationProblem:
    """
    Represents a resource allocation problem with conflicting requirements.
    
    This class models the paradoxical nature of resource allocation where
    different stakeholders have competing preferences and priorities.
    """
    
    def __init__(self, 
                 resources: List[Resource],
                 stakeholders: List[Stakeholder],
                 constraints: List[Callable[[Dict[str, float]], bool]] = None):
        """
        Initialize an allocation problem.
        
        Args:
            resources: List of resources to allocate
            stakeholders: List of stakeholders with preferences
            constraints: Optional list of constraint functions
        """
        self.resources = {r.name: r for r in resources}
        self.stakeholders = {s.name: s for s in stakeholders}
        self.constraints = constraints or []
        self.allocations = {}
    
    def to_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Convert the problem to a matrix representation.
        
        Returns:
            Tuple containing:
            - Preference matrix where rows are stakeholders and columns are resources
            - List of stakeholder names (row labels)
            - List of resource names (column labels)
        """
        stakeholder_names = list(self.stakeholders.keys())
        resource_names = list(self.resources.keys())
        
        # Create matrix with stakeholder preferences
        matrix = np.zeros((len(stakeholder_names), len(resource_names)))
        
        for i, s_name in enumerate(stakeholder_names):
            stakeholder = self.stakeholders[s_name]
            for j, r_name in enumerate(resource_names):
                # Default preference is 0
                matrix[i, j] = stakeholder.preferences.get(r_name, 0.0)
        
        # Normalize preferences per stakeholder
        for i in range(len(stakeholder_names)):
            row_sum = np.sum(matrix[i, :])
            if row_sum > 0:
                matrix[i, :] = matrix[i, :] / row_sum
        
        # Weight by stakeholder influence
        for i, s_name in enumerate(stakeholder_names):
            matrix[i, :] = matrix[i, :] * self.stakeholders[s_name].influence
        
        return matrix, stakeholder_names, resource_names
    
    def __repr__(self) -> str:
        return (f"AllocationProblem({len(self.resources)} resources, "
                f"{len(self.stakeholders)} stakeholders)")


class AllocationOptimizer:
    """
    Optimizer that uses paradox resolution techniques to find optimal allocations.
    
    This class leverages the Crypto_ParadoxOS system to find equilibrium in
    resource allocation problems with competing stakeholder interests.
    """
    
    def __init__(self, 
                api: Optional[CryptoParadoxAPI] = None,
                max_iterations: int = 30,
                convergence_threshold: float = 0.0001):
        """
        Initialize the allocation optimizer.
        
        Args:
            api: CryptoParadoxAPI instance (creates one if None)
            max_iterations: Maximum iterations for optimization
            convergence_threshold: Convergence threshold for optimization
        """
        self.api = api or CryptoParadoxAPI()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._register_specialized_rules()
    
    def _register_specialized_rules(self) -> None:
        """Register specialized transformation rules for resource allocation."""
        
        # Fairness balancing rule
        def fairness_balancing(state: Any) -> Any:
            """Balance allocations to improve fairness across stakeholders."""
            if isinstance(state, np.ndarray) and len(state.shape) == 2:
                # Identify stakeholder satisfaction levels
                satisfaction = np.sum(state, axis=1)
                mean_satisfaction = np.mean(satisfaction)
                
                # Calculate adjustment factor based on deviation from mean
                adjustments = (mean_satisfaction - satisfaction) * 0.1
                
                # Apply adjustments to create more balanced allocations
                adjusted = state.copy()
                for i in range(state.shape[0]):
                    if adjustments[i] != 0:
                        # Adjust allocations proportionally
                        if np.sum(state[i, :]) > 0:
                            adjusted[i, :] = state[i, :] * (1 + adjustments[i])
                
                # Normalize to ensure allocations sum to original total
                row_sums = np.sum(adjusted, axis=1, keepdims=True)
                # Fix: handle normalization row by row to avoid dimension mismatch
                for i in range(adjusted.shape[0]):
                    if row_sums[i] > 0:
                        adjusted[i, :] = adjusted[i, :] / row_sums[i]
                
                return adjusted
            return state
        
        # Constraint satisfaction rule
        def constraint_satisfaction(state: Any) -> Any:
            """Adjust allocations to better satisfy hard constraints."""
            if isinstance(state, np.ndarray) and len(state.shape) == 2:
                # Apply minimum allocation constraints
                # This is a simplified implementation - in practice, this would
                # require knowledge of the specific constraints
                
                # Ensure no negative allocations
                state = np.maximum(state, 0)
                
                # Ensure allocations sum to 1 per resource
                col_sums = np.sum(state, axis=0, keepdims=True)
                # Fix: handle normalization column by column to avoid dimension mismatch
                for j in range(state.shape[1]):
                    if col_sums[0, j] > 0:
                        state[:, j] = state[:, j] / col_sums[0, j]
                
                return state
            return state
        
        # Nash equilibrium seeking rule
        def nash_equilibrium_seeking(state: Any) -> Any:
            """Adjust allocations toward a Nash equilibrium."""
            if isinstance(state, np.ndarray) and len(state.shape) == 2:
                # This is a simplified approach to finding a Nash-like equilibrium
                # Real implementation would be more sophisticated
                
                # Calculate marginal utility matrix
                utility = state.copy()
                
                # Apply a smoothing factor to reduce extreme allocations
                alpha = 0.1
                smoothed = (1 - alpha) * utility + alpha * np.ones_like(utility) / utility.shape[1]
                
                # Normalize to maintain sum constraints
                row_sums = np.sum(smoothed, axis=1, keepdims=True)
                # Fix: handle normalization row by row to avoid dimension mismatch
                for i in range(smoothed.shape[0]):
                    if row_sums[i] > 0:
                        smoothed[i, :] = smoothed[i, :] / row_sums[i]
                
                return smoothed
            return state
        
        # Register the specialized rules
        self.api.register_rule(
            "Fairness Balancing",
            fairness_balancing,
            "Balances allocations to improve fairness across stakeholders."
        )
        
        self.api.register_rule(
            "Constraint Satisfaction",
            constraint_satisfaction,
            "Adjusts allocations to better satisfy hard constraints."
        )
        
        self.api.register_rule(
            "Nash Equilibrium Seeking",
            nash_equilibrium_seeking,
            "Adjusts allocations toward a Nash equilibrium point."
        )
    
    def optimize(self, 
                problem: AllocationProblem,
                custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize resource allocations for the given problem.
        
        Args:
            problem: The allocation problem to solve
            custom_config: Optional custom configuration
            
        Returns:
            Dictionary containing optimization results and allocations
        """
        print(f"Optimizing allocation for {problem}")
        
        # Convert problem to matrix form
        matrix, stakeholder_names, resource_names = problem.to_matrix()
        
        print(f"Generated preference matrix: {matrix.shape}")
        
        # Configure the resolution process
        config = {
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "rules": [
                "Fairness Balancing",
                "Constraint Satisfaction", 
                "Nash Equilibrium Seeking",
                "Eigenvalue Stabilization"  # From standard rules
            ]
        }
        
        if custom_config:
            config.update(custom_config)
        
        # Resolve the paradox using the API
        print("Applying paradox resolution to find optimal allocation...")
        result = self.api.resolve_paradox(matrix, "matrix", config)
        
        if not result["result"]["converged"]:
            print("Warning: Optimization did not fully converge")
        
        # Extract the final allocation matrix
        final_matrix = result["result"]["final_state"]
        
        # Convert to stakeholder-resource allocation dictionary
        allocations = {}
        for i, s_name in enumerate(stakeholder_names):
            s_allocations = {}
            for j, r_name in enumerate(resource_names):
                s_allocations[r_name] = final_matrix[i, j]
            allocations[s_name] = s_allocations
        
        # Calculate aggregate resource allocations
        resource_totals = {}
        for r_name in resource_names:
            # Get the actual total amount for this resource
            total_amount = problem.resources[r_name].total
            
            # Calculate allocation fractions
            r_index = resource_names.index(r_name)
            stakeholder_fractions = {
                s_name: final_matrix[stakeholder_names.index(s_name), r_index]
                for s_name in stakeholder_names
            }
            
            # Normalize fractions to sum to 1
            fraction_sum = sum(stakeholder_fractions.values())
            if fraction_sum > 0:
                normalized_fractions = {
                    s_name: frac / fraction_sum
                    for s_name, frac in stakeholder_fractions.items()
                }
            else:
                # Equal distribution if all fractions are zero
                normalized_fractions = {
                    s_name: 1.0 / len(stakeholder_names)
                    for s_name in stakeholder_names
                }
            
            # Calculate actual allocations
            resource_totals[r_name] = {
                "total": total_amount,
                "allocations": {
                    s_name: normalized_fractions[s_name] * total_amount
                    for s_name in stakeholder_names
                }
            }
        
        # Prepare the results
        optimization_result = {
            "stakeholders": stakeholder_names,
            "resources": resource_names,
            "preference_matrix": matrix.tolist(),
            "final_matrix": final_matrix.tolist(),
            "iterations": result["result"]["iterations"],
            "converged": result["result"]["converged"],
            "execution_time": result["result"]["execution_time"],
            "raw_allocations": allocations,
            "resource_allocations": resource_totals
        }
        
        return optimization_result


# Demonstration 
def demo_allocation():
    """Demonstrate the allocation optimizer with a funding distribution problem."""
    print("Crypto_ParadoxOS Allocation Optimizer Demonstration")
    print("==================================================")
    print("Scenario: Optimizing funding allocation across projects with competing stakeholders")
    
    # Define resources (funds to allocate)
    resources = [
        Resource("Development Fund", total=1000000, min_allocation=200000),
        Resource("Marketing Budget", total=500000, min_allocation=100000),
        Resource("Research Grants", total=750000, min_allocation=150000),
        Resource("Community Rewards", total=250000, min_allocation=50000)
    ]
    
    # Define stakeholders with different preferences
    stakeholders = [
        Stakeholder("Core Developers", influence=0.4, preferences={
            "Development Fund": 0.6,
            "Research Grants": 0.3,
            "Community Rewards": 0.1,
            "Marketing Budget": 0.0
        }),
        Stakeholder("Investors", influence=0.3, preferences={
            "Marketing Budget": 0.5,
            "Development Fund": 0.3,
            "Research Grants": 0.2,
            "Community Rewards": 0.0
        }),
        Stakeholder("Community", influence=0.2, preferences={
            "Community Rewards": 0.4,
            "Development Fund": 0.3,
            "Marketing Budget": 0.2,
            "Research Grants": 0.1
        }),
        Stakeholder("Research Partners", influence=0.1, preferences={
            "Research Grants": 0.7,
            "Development Fund": 0.2,
            "Community Rewards": 0.1,
            "Marketing Budget": 0.0
        })
    ]
    
    # Create the allocation problem
    problem = AllocationProblem(resources, stakeholders)
    
    # Create an optimizer and solve the problem
    optimizer = AllocationOptimizer(max_iterations=50)
    result = optimizer.optimize(problem)
    
    # Display the results
    print("\nOptimization Results:")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Execution time: {result['execution_time']:.4f} seconds")
    
    print("\nResource Allocations:")
    for r_name, r_data in result["resource_allocations"].items():
        print(f"\n{r_name} (Total: ${r_data['total']:,.2f}):")
        for s_name, amount in r_data["allocations"].items():
            print(f"  {s_name}: ${amount:,.2f} ({amount/r_data['total']*100:.1f}%)")
    
    print("\nStakeholder Satisfaction:")
    final_matrix = np.array(result["final_matrix"])
    for i, s_name in enumerate(result["stakeholders"]):
        satisfaction = np.sum(final_matrix[i, :])
        print(f"  {s_name}: {satisfaction:.2f}")
    
    print("\nAllocation Optimization Complete")


if __name__ == "__main__":
    demo_allocation()