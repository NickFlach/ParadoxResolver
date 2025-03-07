import unittest
import numpy as np
from crypto_paradox_optimizer import (
    Resource,
    Stakeholder,
    AllocationProblem,
    AllocationOptimizer
)
from crypto_paradox_api import CryptoParadoxAPI

class TestResource(unittest.TestCase):
    """Test the Resource class functionality."""
    
    def test_resource_initialization(self):
        """Test resource initialization with various parameters."""
        # Test with only required parameters
        resource = Resource("Budget", 1000.0)
        self.assertEqual(resource.name, "Budget")
        self.assertEqual(resource.total, 1000.0)
        self.assertEqual(resource.min_allocation, 0.0)
        self.assertIsNone(resource.max_allocation)
        
        # Test with all parameters
        resource = Resource("Time", 40.0, min_allocation=10.0, max_allocation=35.0)
        self.assertEqual(resource.name, "Time")
        self.assertEqual(resource.total, 40.0)
        self.assertEqual(resource.min_allocation, 10.0)
        self.assertEqual(resource.max_allocation, 35.0)
        
        # Test with min_allocation > 0
        resource = Resource("Staff", 20.0, min_allocation=5.0)
        self.assertEqual(resource.min_allocation, 5.0)
    
    def test_resource_representation(self):
        """Test string representation of resources."""
        resource = Resource("Budget", 1000.0, min_allocation=100.0, max_allocation=900.0)
        repr_str = repr(resource)
        self.assertIn("Budget", repr_str)
        self.assertIn("1000.0", repr_str)
        self.assertIn("100.0", repr_str)
        self.assertIn("900.0", repr_str)


class TestStakeholder(unittest.TestCase):
    """Test the Stakeholder class functionality."""
    
    def test_stakeholder_initialization(self):
        """Test stakeholder initialization with various parameters."""
        # Test with only required parameters
        stakeholder = Stakeholder("Alice")
        self.assertEqual(stakeholder.name, "Alice")
        self.assertEqual(stakeholder.influence, 1.0)
        self.assertEqual(stakeholder.preferences, {})
        
        # Test with all parameters
        preferences = {"Budget": 0.8, "Time": 0.2}
        stakeholder = Stakeholder("Bob", influence=0.7, preferences=preferences)
        self.assertEqual(stakeholder.name, "Bob")
        self.assertEqual(stakeholder.influence, 0.7)
        self.assertEqual(stakeholder.preferences, preferences)
    
    def test_stakeholder_representation(self):
        """Test string representation of stakeholders."""
        preferences = {"Budget": 0.8, "Time": 0.2}
        stakeholder = Stakeholder("Charlie", influence=0.5, preferences=preferences)
        repr_str = repr(stakeholder)
        self.assertIn("Charlie", repr_str)
        self.assertIn("0.5", repr_str)
        self.assertIn("Budget", repr_str)


class TestAllocationProblem(unittest.TestCase):
    """Test the AllocationProblem class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create resources
        self.budget = Resource("Budget", 1000.0, min_allocation=100.0)
        self.time = Resource("Time", 40.0, min_allocation=5.0)
        self.staff = Resource("Staff", 20.0, min_allocation=2.0)
        
        # Create stakeholders
        self.alice = Stakeholder("Alice", influence=0.7, preferences={"Budget": 0.8, "Time": 0.2, "Staff": 0.0})
        self.bob = Stakeholder("Bob", influence=0.5, preferences={"Budget": 0.3, "Time": 0.5, "Staff": 0.2})
        self.charlie = Stakeholder("Charlie", influence=0.3, preferences={"Budget": 0.1, "Time": 0.3, "Staff": 0.6})
        
        # Define constraints
        def budget_time_constraint(allocation):
            """Enforce a minimum ratio of budget to time."""
            if "Budget" not in allocation or "Time" not in allocation:
                return True
            return allocation["Budget"] / allocation["Time"] >= 20.0  # At least $20 per hour
        
        self.constraints = [budget_time_constraint]
        
        # Create the problem
        self.problem = AllocationProblem(
            resources=[self.budget, self.time, self.staff],
            stakeholders=[self.alice, self.bob, self.charlie],
            constraints=self.constraints
        )
    
    def test_to_matrix(self):
        """Test conversion of the problem to a matrix representation."""
        matrix, stakeholder_names, resource_names = self.problem.to_matrix()
        
        # Verify dimensions
        self.assertEqual(matrix.shape, (3, 3))  # 3 stakeholders x 3 resources
        self.assertEqual(len(stakeholder_names), 3)
        self.assertEqual(len(resource_names), 3)
        
        # Verify contents
        self.assertIn("Alice", stakeholder_names)
        self.assertIn("Budget", resource_names)
        
        # Check that the matrix contains preference values
        # Alice's preference for Budget
        alice_idx = stakeholder_names.index("Alice")
        budget_idx = resource_names.index("Budget")
        self.assertAlmostEqual(matrix[alice_idx, budget_idx], 0.8)
    
    def test_representation(self):
        """Test string representation of the allocation problem."""
        repr_str = repr(self.problem)
        self.assertIn("AllocationProblem", repr_str)
        self.assertIn("resources", repr_str)
        self.assertIn("stakeholders", repr_str)
        self.assertIn("constraints", repr_str)


class TestAllocationOptimizer(unittest.TestCase):
    """Test the AllocationOptimizer class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple allocation problem
        budget = Resource("Budget", 1000.0, min_allocation=100.0)
        time = Resource("Time", 40.0, min_allocation=5.0)
        
        alice = Stakeholder("Alice", influence=0.6, preferences={"Budget": 0.7, "Time": 0.3})
        bob = Stakeholder("Bob", influence=0.4, preferences={"Budget": 0.4, "Time": 0.6})
        
        self.problem = AllocationProblem(
            resources=[budget, time],
            stakeholders=[alice, bob],
            constraints=[]
        )
        
        # Create the optimizer
        self.api = CryptoParadoxAPI()
        self.optimizer = AllocationOptimizer(
            api=self.api,
            max_iterations=10,
            convergence_threshold=0.001
        )
    
    def test_optimizer_initialization(self):
        """Test the optimizer initialization."""
        self.assertIsInstance(self.optimizer.api, CryptoParadoxAPI)
        self.assertEqual(self.optimizer.max_iterations, 10)
        self.assertEqual(self.optimizer.convergence_threshold, 0.001)
    
    def test_optimize(self):
        """Test the optimization process."""
        # Run the optimization
        result = self.optimizer.optimize(self.problem)
        
        # Verify basic structure of the result
        self.assertIsInstance(result, dict)
        self.assertIn("resource_allocations", result)
        self.assertIn("raw_allocations", result)
        self.assertIn("converged", result)
        
        # Check resource_allocations
        resource_allocations = result["resource_allocations"]
        self.assertIsInstance(resource_allocations, dict)
        self.assertIn("Budget", resource_allocations)
        self.assertIn("Time", resource_allocations)
        
        # Check that allocations respect resource constraints for Budget
        budget_allocations = resource_allocations["Budget"]["allocations"]
        self.assertIsInstance(budget_allocations, dict)
        for user, amount in budget_allocations.items():
            self.assertGreaterEqual(amount, 0.0)  # Non-negative allocation
            self.assertLessEqual(amount, 1000.0)  # Total resource
        
        # Check that allocations respect resource constraints for Time
        time_allocations = resource_allocations["Time"]["allocations"]
        self.assertIsInstance(time_allocations, dict)
        for user, amount in time_allocations.items():
            self.assertGreaterEqual(amount, 0.0)  # Non-negative allocation
            self.assertLessEqual(amount, 40.0)  # Total resource
        
        # Check raw_allocations structure
        raw_allocations = result["raw_allocations"]
        self.assertIsInstance(raw_allocations, dict)
        self.assertIn("Alice", raw_allocations)
        self.assertIn("Bob", raw_allocations)
        
        # Values in raw_allocations should be between 0 and 1
        for stakeholder in raw_allocations.values():
            for resource_value in stakeholder.values():
                self.assertGreaterEqual(resource_value, 0.0)
                self.assertLessEqual(resource_value, 1.0)
    
    def test_optimize_with_custom_config(self):
        """Test optimization with custom configuration."""
        custom_config = {
            "prioritize_fairness": True,
            "weight_influence": 0.8
        }
        
        result = self.optimizer.optimize(self.problem, custom_config)
        
        # Verify the result
        self.assertIsInstance(result, dict)
        self.assertIn("config_used", result)
        self.assertEqual(result["config_used"]["prioritize_fairness"], True)


if __name__ == "__main__":
    unittest.main()