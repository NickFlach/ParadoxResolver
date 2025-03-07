import unittest
import uuid
import time
from typing import Dict, Any
from crypto_paradox_api import CryptoParadoxAPI, ParadoxJob

class TestParadoxJob(unittest.TestCase):
    """Test the ParadoxJob class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.job_id = str(uuid.uuid4())
        self.paradox_input = 0.5
        self.input_type = "numerical"
        self.config = {"max_iterations": 10}
        
        # Create a job
        self.job = ParadoxJob(
            paradox_input=self.paradox_input,
            input_type=self.input_type,
            config=self.config
        )
        
        # Set job ID for testing
        self.job.job_id = self.job_id
    
    def test_job_initialization(self):
        """Test job initialization."""
        self.assertEqual(self.job.paradox_input, self.paradox_input)
        self.assertEqual(self.job.input_type, self.input_type)
        self.assertEqual(self.job.config, self.config)
        self.assertEqual(self.job.status, "created")
        self.assertIsNone(self.job.result)
        self.assertIsNone(self.job.error)
        self.assertIsNotNone(self.job.created_at)
    
    def test_job_lifecycle(self):
        """Test the job lifecycle methods."""
        # Test job start
        self.job.start()
        self.assertEqual(self.job.status, "running")
        self.assertIsNotNone(self.job.started_at)
        
        # Create a mock result
        class MockParadoxState:
            def __init__(self):
                self.value = 1.0
                self.type_name = "numerical"
                self.iteration_count = 5
                self.history = [0.5, 0.7, 0.8, 0.9, 1.0]
                self.converged = True
            
            def get_history(self):
                return self.history
                
        mock_result = MockParadoxState()
        
        # Test job completion
        self.job.complete(mock_result)
        self.assertEqual(self.job.status, "completed")
        self.assertIsNotNone(self.job.completed_at)
        self.assertIsNotNone(self.job.result)
        
        # Check formatted result
        formatted_result = self.job.result
        self.assertIn("final_value", formatted_result)
        self.assertIn("history", formatted_result)
        self.assertIn("converged", formatted_result)
        self.assertEqual(formatted_result["final_value"], 1.0)
        self.assertEqual(len(formatted_result["history"]), 5)
        self.assertTrue(formatted_result["converged"])
        
        # Test job failure
        error_job = ParadoxJob(0.0, "numerical")
        error_job.start()
        error_job.fail("Division by zero")
        self.assertEqual(error_job.status, "failed")
        self.assertEqual(error_job.error, "Division by zero")
        self.assertIsNotNone(error_job.completed_at)
    
    def test_to_dict(self):
        """Test conversion of job to dictionary."""
        job_dict = self.job.to_dict()
        
        self.assertIsInstance(job_dict, dict)
        self.assertEqual(job_dict["job_id"], self.job_id)
        self.assertEqual(job_dict["status"], "created")
        self.assertEqual(job_dict["input_type"], self.input_type)
        self.assertIn("created_at", job_dict)


class TestCryptoParadoxAPI(unittest.TestCase):
    """Test the CryptoParadoxAPI class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api = CryptoParadoxAPI()
    
    def test_api_initialization(self):
        """Test API initialization."""
        self.assertIsNotNone(self.api.resolver)
        self.assertGreater(len(self.api.transformation_rules), 0)
    
    def test_load_standard_rules(self):
        """Test loading of standard rules."""
        # Clear rules
        self.api.transformation_rules = {}
        
        # Load standard rules
        self.api.load_standard_rules()
        
        # Verify rules were loaded
        self.assertGreater(len(self.api.transformation_rules), 0)
        standard_rule_names = ["Fixed-Point Iteration", "Contradiction Resolution"]
        for rule_name in standard_rule_names:
            self.assertIn(rule_name, self.api.get_available_rules())
    
    def test_register_rule(self):
        """Test registering custom rules."""
        # Define a custom rule
        def test_rule(state):
            """Test transformation rule."""
            if isinstance(state, (int, float)):
                return state * 2
            return state
        
        # Register the rule
        self.api.register_rule("Double Value", test_rule, "Doubles numeric values")
        
        # Verify rule was registered
        available_rules = self.api.get_available_rules()
        self.assertIn("Double Value", available_rules)
        self.assertEqual(available_rules["Double Value"], "Doubles numeric values")
        
        # Test the rule with the API
        job_id = self.api.create_job(5.0, "numerical")
        result = self.api.execute_job(job_id)
        
        # The result should include the effect of our custom rule
        self.assertNotEqual(result["result"]["final_value"], 5.0)
    
    def test_create_job(self):
        """Test job creation."""
        # Create jobs with different input types
        numerical_job_id = self.api.create_job(0.5, "numerical")
        matrix_job_id = self.api.create_job([[1, 0], [0, 1]], "matrix")
        text_job_id = self.api.create_job("This statement is false.", "text")
        
        # Verify jobs were created
        self.assertIsInstance(numerical_job_id, str)
        self.assertIsInstance(matrix_job_id, str)
        self.assertIsInstance(text_job_id, str)
        
        # Verify jobs are in the API's job dictionary
        self.assertIn(numerical_job_id, self.api.jobs)
        self.assertIn(matrix_job_id, self.api.jobs)
        self.assertIn(text_job_id, self.api.jobs)
        
        # Verify job types
        self.assertEqual(self.api.jobs[numerical_job_id].input_type, "numerical")
        self.assertEqual(self.api.jobs[matrix_job_id].input_type, "matrix")
        self.assertEqual(self.api.jobs[text_job_id].input_type, "text")
    
    def test_get_job_status(self):
        """Test retrieving job status."""
        # Create a job
        job_id = self.api.create_job(0.5, "numerical")
        
        # Get initial status
        status = self.api.get_job_status(job_id)
        self.assertEqual(status["status"], "created")
        
        # Execute job (should change status)
        self.api.execute_job(job_id)
        
        # Get updated status
        status = self.api.get_job_status(job_id)
        self.assertEqual(status["status"], "completed")
        self.assertIn("result", status)
    
    def test_execute_job(self):
        """Test job execution."""
        # Create a job with custom config
        config = {"max_iterations": 5, "convergence_threshold": 0.01}
        job_id = self.api.create_job(0.5, "numerical", config)
        
        # Execute the job
        result = self.api.execute_job(job_id)
        
        # Verify job execution results
        self.assertIsInstance(result, dict)
        self.assertIn("result", result)
        self.assertIn("final_value", result["result"])
        self.assertIn("history", result["result"])
        self.assertIn("converged", result["result"])
        self.assertIn("execution_time", result["result"])
        
        # Verify numeric result is different from input (transformed)
        self.assertNotEqual(result["result"]["final_value"], 0.5)
        
        # Verify history includes steps
        self.assertGreater(len(result["result"]["history"]), 0)
        
        # Test execution of nonexistent job
        with self.assertRaises(ValueError):
            self.api.execute_job("nonexistent-job-id")
    
    def test_resolve_paradox(self):
        """Test the convenience method for paradox resolution."""
        # Resolve different types of paradoxes
        numerical_result = self.api.resolve_paradox(0.5, "numerical")
        matrix_result = self.api.resolve_paradox([[1, 0], [0, 1]], "matrix")
        text_result = self.api.resolve_paradox("This statement is false.", "text")
        
        # Verify results
        self.assertIsInstance(numerical_result, dict)
        self.assertIsInstance(matrix_result, dict)
        self.assertIsInstance(text_result, dict)
        
        # Check structure of results
        for result in [numerical_result, matrix_result, text_result]:
            self.assertIn("result", result)
            self.assertIn("final_value", result["result"])
            self.assertIn("history", result["result"])
            self.assertIn("converged", result["result"])
            self.assertIn("execution_time", result["result"])
    
    def test_get_available_rules(self):
        """Test retrieving available rules."""
        rules = self.api.get_available_rules()
        
        self.assertIsInstance(rules, dict)
        self.assertGreater(len(rules), 0)
        
        # Verify standard rules are included
        standard_rule_names = ["Fixed-Point Iteration", "Contradiction Resolution"]
        for rule_name in standard_rule_names:
            self.assertIn(rule_name, rules)
            self.assertIsInstance(rules[rule_name], str)  # Should be description


if __name__ == "__main__":
    unittest.main()