import unittest
import numpy as np
from paradox_resolver import ParadoxResolver
from transformation_rules import (
    fixed_point_iteration, 
    contradiction_resolution,
    eigenvalue_stabilization,
    fuzzy_logic_transformation,
    duality_inversion,
    bayesian_update,
    recursive_normalization,
    self_reference_unwinding
)
from utils import format_paradox_input, validate_input

class TestTransformationRules(unittest.TestCase):
    """Test the core transformation rules functionality."""
    
    def test_fixed_point_iteration_numeric(self):
        """Test the fixed-point iteration rule with numeric values."""
        # Test with a simple numeric input
        value = 2.0
        result = fixed_point_iteration(value)
        self.assertIsInstance(result, float)
        # Result should move toward the fixed point of the equation x = 1/x
        # which is x = 1 or x = -1
        self.assertNotEqual(result, value)  # Should change the value
        
        # Test with a value that should converge to 1
        close_to_one = 0.9
        result = fixed_point_iteration(close_to_one)
        self.assertGreater(result, close_to_one)  # Should move toward 1
        
        # Test with zero (edge case)
        result = fixed_point_iteration(0.0)
        self.assertNotEqual(result, 0.0)  # Should transform away from zero
    
    def test_fixed_point_iteration_equation(self):
        """Test the fixed-point iteration rule with equation strings."""
        # Test with a reciprocal equation
        equation = "x = 1/x"
        result = fixed_point_iteration(equation)
        self.assertIsInstance(result, str)
        self.assertIn("=", result)  # Should still have an equals sign
        self.assertNotEqual(result, equation)  # Should transform the equation
        
        # Test with an equation that doesn't match our patterns
        equation = "x = x + 1"
        result = fixed_point_iteration(equation)
        self.assertEqual(result, equation)  # Should leave it unchanged
    
    def test_fixed_point_iteration_matrix(self):
        """Test the fixed-point iteration rule with matrix inputs."""
        # Create a matrix
        matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = fixed_point_iteration(matrix)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, matrix.shape)
        # Should change the matrix
        self.assertFalse(np.array_equal(result, matrix))
    
    def test_contradiction_resolution_text(self):
        """Test the contradiction resolution rule with text."""
        # Test with the classic liar paradox
        text = "This statement is false."
        result = contradiction_resolution(text)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, text)  # Should transform it
        
        # Test with a non-paradoxical statement
        text = "The sky is blue."
        result = contradiction_resolution(text)
        self.assertEqual(result, text)  # Should leave it unchanged
    
    def test_eigenvalue_stabilization(self):
        """Test the eigenvalue stabilization rule with matrices."""
        # Create an unstable matrix
        matrix = np.array([[1.5, 0.5], [0.5, 1.5]])
        result = eigenvalue_stabilization(matrix)
        
        # Result should be a matrix of the same shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, matrix.shape)
        
        # The eigenvalues of the result should be smaller
        orig_eigenvalues = np.linalg.eigvals(matrix)
        result_eigenvalues = np.linalg.eigvals(result)
        self.assertLess(np.max(np.abs(result_eigenvalues)), np.max(np.abs(orig_eigenvalues)))
        
        # Test with a non-matrix input
        result = eigenvalue_stabilization(42)
        self.assertEqual(result, 42)  # Should leave non-matrices unchanged
    
    def test_duality_inversion(self):
        """Test the duality inversion rule."""
        # Test with numeric value between 0 and 1
        value = 0.3
        result = duality_inversion(value)
        self.assertAlmostEqual(result, 0.7, places=5)  # Should invert the value
        
        # Test with text containing opposite terms
        text = "Is it true or false?"
        result = duality_inversion(text)
        self.assertNotEqual(result, text)  # Should replace with a middle ground
        self.assertNotIn("true", result.lower())  # The extremes should be replaced
        self.assertNotIn("false", result.lower())
    
    def test_bayesian_update(self):
        """Test the Bayesian update rule."""
        # Test with probability values
        prob = 0.7
        result = bayesian_update(prob)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)
        
        # Test with list of probabilities
        probs = [0.2, 0.3, 0.5]
        result = bayesian_update(probs)
        self.assertEqual(len(result), len(probs))
        self.assertAlmostEqual(sum(result), 1.0, places=5)  # Should normalize to sum to 1
    
    def test_recursive_normalization(self):
        """Test the recursive normalization rule."""
        # Test with a large value
        large_val = 100.0
        result = recursive_normalization(large_val)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)
        
        # Test with an array
        array = np.array([-10, 0, 10])
        result = recursive_normalization(array)
        self.assertEqual(result.shape, array.shape)
        self.assertLessEqual(np.max(result), 1.0)
        self.assertGreaterEqual(np.min(result), -1.0)
        
        # Test with a nested structure
        nested = {"a": 10, "b": {"c": -5, "d": 20}}
        result = recursive_normalization(nested)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["b"], dict)
        self.assertGreaterEqual(result["a"], 0)
        self.assertLessEqual(result["a"], 1)
        
    def test_self_reference_unwinding(self):
        """Test the self-reference unwinding rule."""
        # Test with self-referential text
        text = "This statement refers to itself."
        result = self_reference_unwinding(text)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, text)  # Should transform it
        
        # Test with a structured self-reference
        structure = {"content": "paradoxical", "refers_to": "self", "level": 2}
        result = self_reference_unwinding(structure)
        self.assertIsInstance(result, dict)
        self.assertNotEqual(result["refers_to"], "self")  # Should unwind the self-reference
        self.assertLess(result["level"], structure["level"])  # Level should decrease


class TestParadoxResolver(unittest.TestCase):
    """Test the paradox resolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a resolver with some basic rules
        self.resolver = ParadoxResolver(
            transformation_rules={
                "Fixed-Point": fixed_point_iteration,
                "Duality": duality_inversion
            },
            max_iterations=10,
            convergence_threshold=0.001
        )
    
    def test_numerical_resolution(self):
        """Test resolving a numerical paradox."""
        # Resolve a simple numerical paradox (fixed point of x = 1/x)
        initial_state = 2.0
        result, steps, converged = self.resolver.resolve(initial_state)
        
        self.assertIsInstance(result, float)
        self.assertIsInstance(steps, list)
        self.assertIsInstance(converged, bool)
        
        # Should have created steps including the initial state
        self.assertGreater(len(steps), 1)
        self.assertEqual(steps[0], initial_state)
        
        # Result should not equal the initial state (transformation happened)
        self.assertNotEqual(result, initial_state)
        
        # Check that we're approaching a stable point (but don't require 
        # it to be exactly 1.0 as the implementation may have different dynamics)
        last_delta = abs(steps[-1] - steps[-2])
        self.assertLess(last_delta, 0.1)
    
    def test_matrix_resolution(self):
        """Test resolving a matrix paradox."""
        # Create a resolver with matrix-specific rules
        matrix_resolver = ParadoxResolver(
            transformation_rules={
                "Eigenvalue": eigenvalue_stabilization,
                "Normalization": recursive_normalization
            },
            max_iterations=10
        )
        
        # Resolve an unstable matrix
        initial_state = np.array([[1.2, 0.5], [0.5, 1.2]])
        result, steps, converged = matrix_resolver.resolve(initial_state)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, initial_state.shape)
        self.assertGreater(len(steps), 1)
        
        # The result should be more stable
        orig_eigenvalues = np.linalg.eigvals(initial_state)
        result_eigenvalues = np.linalg.eigvals(result)
        self.assertLess(np.max(np.abs(result_eigenvalues)), np.max(np.abs(orig_eigenvalues)))
    
    def test_convergence_detection(self):
        """Test that the resolver correctly detects convergence."""
        # Create a resolver with very relaxed convergence threshold
        relaxed_resolver = ParadoxResolver(
            transformation_rules={"Identity": lambda x: x},  # Identity function
            convergence_threshold=10.0  # Very large threshold to force convergence
        )
        
        # Should converge immediately since the state doesn't change
        result, steps, converged = relaxed_resolver.resolve(5.0)
        self.assertTrue(converged)
        self.assertEqual(len(steps), 2)  # Initial + 1 iteration
        
        # Create a resolver that should never converge
        divergent_resolver = ParadoxResolver(
            transformation_rules={"Diverge": lambda x: x + 1.0},  # Always increase
            max_iterations=5,
            convergence_threshold=0.001
        )
        
        # Should not converge as the value keeps increasing
        result, steps, converged = divergent_resolver.resolve(0.0)
        self.assertFalse(converged)
        self.assertEqual(len(steps), 6)  # Initial + 5 iterations
        self.assertEqual(result, 5.0)  # 0 + 5 iterations of +1
    
    def test_resolution_with_multiple_rules(self):
        """Test that multiple rules are applied correctly."""
        # Create a resolver with multiple rules that have distinct effects
        multi_resolver = ParadoxResolver(
            transformation_rules={
                "Fixed-Point": fixed_point_iteration,
                "Duality": duality_inversion,
                "Bayesian": bayesian_update
            },
            max_iterations=3
        )
        
        result, steps, converged = multi_resolver.resolve(0.8)
        self.assertEqual(len(steps), 4)  # Initial + 3 iterations
        
        # Each step should be different
        for i in range(1, len(steps)):
            self.assertNotEqual(steps[i], steps[i-1])


class TestUtilityFunctions(unittest.TestCase):
    """Test the utility functions."""
    
    def test_format_paradox_input(self):
        """Test the input formatting function."""
        # Test numerical input formatting
        result = format_paradox_input(5, "Numerical")
        self.assertEqual(result, 5.0)
        
        # Test equation formatting
        result = format_paradox_input("x = 1/x", "Numerical")
        self.assertEqual(result, "x = 1/x")
        
        # Test matrix formatting
        matrix = [[1, 2], [3, 4]]
        result = format_paradox_input(matrix, "Matrix")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        
        # Test text formatting
        result = format_paradox_input("  Test  ", "Text")
        self.assertEqual(result, "Test")
    
    def test_validate_input(self):
        """Test the input validation function."""
        # Test valid inputs
        self.assertTrue(validate_input("This is a test", "Text")[0])
        self.assertTrue(validate_input(5.0, "Numerical")[0])
        self.assertTrue(validate_input("x = y", "Numerical")[0])
        self.assertTrue(validate_input(np.eye(3), "Matrix")[0])
        
        # Test invalid inputs
        self.assertFalse(validate_input("", "Text")[0])  # Too short
        self.assertFalse(validate_input("x y", "Numerical")[0])  # Not a number or equation
        self.assertFalse(validate_input(np.array([1, 2, 3]), "Matrix")[0])  # Not 2D
        self.assertFalse(validate_input(np.array([[1, 2], [3, 4], [5, 6]]), "Matrix")[0])  # Not square


if __name__ == "__main__":
    unittest.main()