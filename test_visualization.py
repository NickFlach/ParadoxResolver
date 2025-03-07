import unittest
import numpy as np
from unittest.mock import patch
import visualization

class TestVisualizationUtils(unittest.TestCase):
    """Test the visualization utility functions."""
    
    def test_convert_steps_to_numeric_simple(self):
        """Test converting simple numeric steps."""
        # Test with list of numbers
        steps = [0.5, 0.7, 0.9, 1.1]
        result = visualization.convert_steps_to_numeric(steps, dimension=1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        np.testing.assert_array_almost_equal(result, np.array(steps))
    
    def test_convert_steps_to_numeric_matrices(self):
        """Test converting matrix steps."""
        # Test with list of matrices
        steps = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([[0.8, 0.2], [0.2, 0.8]])
        ]
        result = visualization.convert_steps_to_numeric(steps, dimension=2)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 4))  # 3 steps, 4 elements per matrix
        
        # Test flattening of matrices
        expected = np.array([
            [1, 0, 0, 1],
            [0.9, 0.1, 0.1, 0.9],
            [0.8, 0.2, 0.2, 0.8]
        ])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_convert_steps_to_numeric_lists(self):
        """Test converting lists of lists."""
        # Test with nested lists
        steps = [
            [1, 2, 3],
            [1.5, 2.5, 3.5],
            [2, 3, 4]
        ]
        result = visualization.convert_steps_to_numeric(steps, dimension=3)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 3))
        
        # Test with lists of different lengths
        steps = [
            [1, 2],
            [1, 2, 3],
            [1]
        ]
        result = visualization.convert_steps_to_numeric(steps, dimension=3)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 3))
        # Check that shorter lists are padded with NaN
        self.assertTrue(np.isnan(result[1, 2]))
        self.assertTrue(np.isnan(result[2, 1]))
        self.assertTrue(np.isnan(result[2, 2]))
    
    def test_convert_steps_to_numeric_text(self):
        """Test converting text steps."""
        # Test with string steps
        steps = [
            "This statement is false.",
            "This statement refers to itself.",
            "This statement is contextually dependent."
        ]
        result = visualization.convert_steps_to_numeric(steps, dimension=1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        # Should extract some numeric measure from strings
        self.assertEqual(len(result), len(steps))
    
    def test_convert_steps_to_numeric_unsupported(self):
        """Test converting unsupported types."""
        # Test with mixed types
        steps = [1, "text", [1, 2]]
        result = visualization.convert_steps_to_numeric(steps, dimension=1)
        self.assertIsNone(result)
    
    def test_calculate_text_complexity(self):
        """Test calculating text complexity metrics."""
        texts = [
            "This is a simple statement.",
            "This statement refers to itself and is self-referential.",
            "The statement above is true, but this one is false."
        ]
        metrics = visualization.calculate_text_complexity(texts)
        
        # Verify metrics structure
        self.assertIsInstance(metrics, dict)
        self.assertIn("length", metrics)
        self.assertIn("unique_words", metrics)
        self.assertIn("self_reference", metrics)
        
        # Verify metric values
        self.assertEqual(len(metrics["length"]), 3)
        self.assertEqual(len(metrics["unique_words"]), 3)
        self.assertEqual(len(metrics["self_reference"]), 3)
        
        # Second text should have higher self_reference count
        self.assertGreater(metrics["self_reference"][1], metrics["self_reference"][0])


# Mocking Streamlit for testing visualization functions
class MockStreamlit:
    """Mock class for streamlit functions."""
    
    def __init__(self):
        self.warnings = []
        self.plots = []
    
    def warning(self, text):
        self.warnings.append(text)
        
    def plotly_chart(self, fig, use_container_width=False):
        self.plots.append(fig)


@patch("visualization.st", new_callable=MockStreamlit)
class TestVisualizationFunctions(unittest.TestCase):
    """Test the visualization functions with mocked streamlit."""
    
    def test_visualize_resolution_steps_no_steps(self, mock_st):
        """Test visualization with no steps."""
        visualization.visualize_resolution_steps([], "Line chart")
        self.assertEqual(len(mock_st.warnings), 1)
        self.assertEqual(len(mock_st.plots), 0)
    
    def test_visualize_resolution_steps_numeric(self, mock_st):
        """Test visualization with numeric steps."""
        steps = [0.5, 0.7, 0.9, 1.1]
        visualization.visualize_resolution_steps(steps, "Line chart", dimension=1)
        self.assertEqual(len(mock_st.plots), 1)
    
    def test_visualize_resolution_steps_matrix(self, mock_st):
        """Test visualization with matrix steps."""
        steps = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([[0.8, 0.2], [0.2, 0.8]])
        ]
        visualization.visualize_resolution_steps(steps, "Heatmap", dimension=2)
        self.assertEqual(len(mock_st.plots), 1)
    
    def test_visualize_text_transitions(self, mock_st):
        """Test visualization of text transitions."""
        steps = [
            "This statement is false.",
            "This statement refers to itself.",
            "This statement is contextually dependent."
        ]
        visualization.visualize_text_transitions(steps)
        self.assertEqual(len(mock_st.plots), 1)
    
    def test_plot_convergence(self, mock_st):
        """Test plotting convergence."""
        steps = [0.5, 0.7, 0.9, 1.0, 1.0]
        visualization.plot_convergence(steps, threshold=0.001)
        self.assertEqual(len(mock_st.plots), 1)
    
    def test_plot_convergence_no_convergence(self, mock_st):
        """Test plotting convergence with no convergence."""
        steps = [0.5]  # Only one step
        visualization.plot_convergence(steps, threshold=0.001)
        self.assertEqual(len(mock_st.warnings), 1)
        self.assertEqual(len(mock_st.plots), 0)
    
    def test_create_line_chart(self, mock_st):
        """Test creating a line chart."""
        data = np.array([0.5, 0.7, 0.9, 1.0])
        visualization.create_line_chart(data)
        self.assertEqual(len(mock_st.plots), 1)
    
    def test_create_bar_chart(self, mock_st):
        """Test creating a bar chart."""
        data = np.array([0.5, 0.7, 0.9, 1.0])
        visualization.create_bar_chart(data)
        self.assertEqual(len(mock_st.plots), 1)
    
    def test_create_heatmap(self, mock_st):
        """Test creating a heatmap."""
        data = np.array([0.5, 0.7, 0.9, 1.0])
        visualization.create_heatmap(data, dimension=1)
        self.assertEqual(len(mock_st.plots), 1)
    
    def test_create_3d_projection(self, mock_st):
        """Test creating a 3D projection."""
        data = np.array([
            [1, 2, 3],
            [1.5, 2.5, 3.5],
            [2, 3, 4]
        ])
        visualization.create_3d_projection(data, dimension=3)
        self.assertEqual(len(mock_st.plots), 1)


if __name__ == "__main__":
    unittest.main()