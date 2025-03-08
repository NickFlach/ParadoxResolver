import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Any, Tuple, Dict, Union

def visualize_resolution_steps(steps: List[Any], viz_type: str = "Line chart", dimension: int = 1):
    """
    Visualize the resolution steps of a paradox.
    
    Args:
        steps: List of states during resolution
        viz_type: Type of visualization to generate
        dimension: Dimension of the paradox state
    """
    if not steps:
        st.warning("No steps to visualize.")
        return
    
    # Convert steps to numerical representation for visualization
    numeric_data = convert_steps_to_numeric(steps, dimension)
    
    if numeric_data is None:
        st.warning("Cannot visualize this type of paradox data directly.")
        if isinstance(steps[0], str):
            # For string-based paradoxes, show text transitions
            visualize_text_transitions(steps)
        return
    
    # Generate the appropriate visualization based on type
    if viz_type == "Line chart":
        create_line_chart(numeric_data)
    elif viz_type == "Bar chart":
        create_bar_chart(numeric_data)
    elif viz_type == "Heatmap":
        create_heatmap(numeric_data, dimension)
    elif viz_type == "3D projection":
        create_3d_projection(numeric_data, dimension)

def convert_steps_to_numeric(steps: List[Any], dimension: int) -> Union[np.ndarray, None]:
    """
    Convert steps to a numerical representation for visualization.
    
    Args:
        steps: List of states during resolution
        dimension: Dimension of the paradox state
        
    Returns:
        Numpy array of numeric values or None if conversion not possible
    """
    # Handle empty list
    if not steps:
        return None
        
    # Special handling for a single value
    if len(steps) == 1:
        # If it's a single numeric value, convert to a 1-element array
        if isinstance(steps[0], (int, float)):
            return np.array([steps[0]]).reshape(1, 1)
        # If it's a numpy array, wrap it
        elif isinstance(steps[0], np.ndarray):
            try:
                flat_val = steps[0].flatten()
                return flat_val.reshape(1, len(flat_val))
            except Exception as e:
                st.error(f"Error reshaping numpy array: {e}")
                return np.array([0]).reshape(1, 1)  # Fallback
        # If it's a list or tuple of numbers
        elif isinstance(steps[0], (list, tuple)) and all(isinstance(val, (int, float)) for val in steps[0]):
            return np.array([steps[0]]).reshape(1, len(steps[0]))
        # If it's a dict with numeric values
        elif isinstance(steps[0], dict) and all(isinstance(val, (int, float)) for val in steps[0].values()):
            vals = list(steps[0].values())
            return np.array([vals]).reshape(1, len(vals))
        # If it's a string
        elif isinstance(steps[0], str):
            return np.array([len(steps[0]) + steps[0].count(" ") * 0.5]).reshape(1, 1)
    
    # Multiple values handling
    if all(isinstance(step, (int, float)) for step in steps):
        # 1D numerical values
        return np.array(steps).reshape(len(steps), 1)
    
    elif all(isinstance(step, np.ndarray) for step in steps):
        # Handle matrix evolution
        try:
            shapes = set(step.shape for step in steps if hasattr(step, 'shape'))
            if len(shapes) == 1:  # All have same shape
                return np.array([step.flatten() for step in steps])
            else:
                # Handle different shaped arrays by padding
                max_size = max(np.prod(step.shape) if hasattr(step, 'shape') else 1 for step in steps)
                padded = []
                for step in steps:
                    if hasattr(step, 'flatten'):
                        flat = step.flatten()
                        padded.append(np.pad(flat, (0, max_size - flat.size), 'constant', constant_values=np.nan))
                    else:
                        padded.append(np.array([step]).flatten())
                return np.array(padded)
        except Exception as e:
            # Safely handle any numpy array manipulation errors
            st.error(f"Error processing numpy arrays: {str(e)}")
            return np.zeros((len(steps), 1))  # Fallback
    
    elif all(isinstance(step, (list, tuple)) for step in steps) and all(
            all(isinstance(val, (int, float)) for val in step) for step in steps):
        # Lists or tuples of numbers
        max_len = max(len(step) for step in steps)
        # Pad shorter lists with NaN
        padded = [list(step) + [np.nan] * (max_len - len(step)) for step in steps]
        return np.array(padded)
    
    elif all(isinstance(step, dict) for step in steps) and all(
            all(isinstance(val, (int, float)) for val in step.values()) for step in steps):
        # Dictionaries with numeric values
        all_keys = sorted(set().union(*(step.keys() for step in steps)))
        padded = [[step.get(key, np.nan) for key in all_keys] for step in steps]
        return np.array(padded)
    
    elif all(isinstance(step, str) for step in steps):
        # For strings, try to extract numeric patterns
        try:
            # Try to calculate a complexity measure for each string
            complexity = [len(step) + step.count(" ") * 0.5 for step in steps]
            return np.array(complexity).reshape(len(steps), 1)
        except Exception as e:
            st.error(f"Error processing string metrics: {str(e)}")
            return np.zeros((len(steps), 1))  # Fallback
    
    # If nothing else worked, provide a safe fallback
    return np.zeros((len(steps), 1))

def create_line_chart(data: np.ndarray):
    """Create a line chart visualization of the resolution steps."""
    num_steps, num_dims = data.shape if len(data.shape) > 1 else (data.shape[0], 1)
    
    if num_dims == 1:
        # Simple 1D evolution
        fig = px.line(
            x=list(range(num_steps)), 
            y=data,
            labels={"x": "Step", "y": "Value"},
            title="Paradox Resolution Evolution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Multiple dimensions evolution
        fig = go.Figure()
        for i in range(min(num_dims, 10)):  # Limit to 10 dimensions for clarity
            fig.add_trace(go.Scatter(
                x=list(range(num_steps)),
                y=data[:, i],
                mode='lines+markers',
                name=f'Dimension {i+1}'
            ))
        fig.update_layout(
            title="Paradox Resolution Evolution",
            xaxis_title="Step",
            yaxis_title="Value",
            legend_title="Dimensions"
        )
        st.plotly_chart(fig, use_container_width=True)

def create_bar_chart(data: np.ndarray):
    """Create a bar chart visualization of the resolution steps."""
    num_steps, num_dims = data.shape if len(data.shape) > 1 else (data.shape[0], 1)
    
    if num_dims == 1:
        # Simple 1D evolution as bars
        fig = px.bar(
            x=list(range(num_steps)), 
            y=data,
            labels={"x": "Step", "y": "Value"},
            title="Paradox Resolution Evolution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Use a grouped bar chart for multidimensional data
        # Show only the initial, middle, and final steps if there are many
        if num_steps > 10:
            indices = [0, num_steps//2, num_steps-1]
            labels = ["Initial", "Middle", "Final"]
            selected_data = data[indices, :]
        else:
            indices = list(range(num_steps))
            labels = [f"Step {i}" for i in indices]
            selected_data = data
        
        # Create a grouped bar chart
        fig = go.Figure()
        for i in range(min(num_dims, 8)):  # Limit dimensions for clarity
            fig.add_trace(go.Bar(
                x=labels,
                y=selected_data[:, i],
                name=f'Dimension {i+1}'
            ))
        
        fig.update_layout(
            title="Key Steps in Paradox Resolution",
            xaxis_title="Step",
            yaxis_title="Value",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_heatmap(data: np.ndarray, dimension: int):
    """Create a heatmap visualization of the resolution steps."""
    if len(data.shape) == 1:
        # 1D data - represent as a heatmap over time
        heatmap_data = data.reshape(-1, 1)
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Dimension", y="Step", color="Value"),
            x=['Value'],
            y=[f"Step {i}" for i in range(len(data))],
            color_continuous_scale='viridis'
        )
        fig.update_layout(title="Value Evolution Heatmap")
    
    elif dimension > 1 and len(data.shape) > 1:
        # For matrix data, show the evolution of all elements
        num_steps = data.shape[0]
        
        # If there are too many steps, sample a subset
        if num_steps > 10:
            indices = np.linspace(0, num_steps-1, 10, dtype=int)
            selected_data = data[indices]
            y_labels = [f"Step {i}" for i in indices]
        else:
            selected_data = data
            y_labels = [f"Step {i}" for i in range(num_steps)]
        
        # Create element labels
        x_labels = [f"Element {i+1}" for i in range(data.shape[1])]
        
        fig = px.imshow(
            selected_data,
            labels=dict(x="Matrix Element", y="Step", color="Value"),
            x=x_labels[:20],  # Limit to 20 elements for clarity
            y=y_labels,
            color_continuous_scale='viridis'
        )
        fig.update_layout(title="Matrix Element Evolution Heatmap")
    
    else:
        # Generic heatmap for any numeric data
        fig = px.imshow(
            data,
            labels=dict(x="Dimension", y="Step", color="Value"),
            color_continuous_scale='viridis'
        )
        fig.update_layout(title="Paradox Resolution Heatmap")
    
    st.plotly_chart(fig, use_container_width=True)

def create_3d_projection(data: np.ndarray, dimension: int):
    """Create a 3D projection visualization of the resolution steps."""
    num_steps = data.shape[0]
    
    if len(data.shape) == 1 or data.shape[1] == 1:
        # For 1D data, create a 3D line with time as one dimension
        values = data.flatten()
        fig = go.Figure(data=[go.Scatter3d(
            x=list(range(num_steps)),
            y=[0] * num_steps,  # Fixed y position
            z=values,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=list(range(num_steps)),
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        fig.update_layout(
            title="1D Paradox Resolution in 3D Space",
            scene=dict(
                xaxis_title="Step",
                yaxis_title="",
                zaxis_title="Value"
            )
        )
    
    elif data.shape[1] >= 2:
        # For multidimensional data, show the evolution in 3D space
        # Use first 3 dimensions or create derived dimensions
        
        if data.shape[1] >= 3:
            # Use first 3 dimensions directly
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            title = "Paradox Resolution Trajectory (First 3 Dimensions)"
        else:
            # For 2D data, use step number as 3rd dimension
            x = data[:, 0]
            y = data[:, 1] if data.shape[1] > 1 else np.zeros(num_steps)
            z = np.arange(num_steps)
            title = "Paradox Resolution Trajectory (2D + Time)"
        
        # Create a 3D path showing the evolution
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=list(range(num_steps)),
                colorscale='Viridis',
                opacity=0.8
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        )])
        
        # Add markers for start and end points
        fig.add_trace(go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                symbol='circle'
            ),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x[-1]],
            y=[y[-1]],
            z=[z[-1]],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle'
            ),
            name='End'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3" if data.shape[1] >= 3 else "Step"
            )
        )
    
    else:
        # Fallback for unexpected data
        st.warning("Cannot create 3D projection for this data type.")
        return
    
    st.plotly_chart(fig, use_container_width=True)

def visualize_text_transitions(steps: List[str]):
    """
    Create a visualization for text-based paradox resolution.
    
    Args:
        steps: List of string states during resolution
    """
    # Create a simple table showing the text evolution
    df_data = {
        "Step": [f"Step {i}" for i in range(len(steps))],
        "Statement": steps
    }
    
    st.table(df_data)
    
    # Also create a text complexity chart
    complexity_metrics = calculate_text_complexity(steps)
    
    fig = go.Figure()
    
    # Add length metric
    fig.add_trace(go.Scatter(
        x=list(range(len(steps))),
        y=complexity_metrics['length'],
        mode='lines+markers',
        name='Length'
    ))
    
    # Add unique words metric
    fig.add_trace(go.Scatter(
        x=list(range(len(steps))),
        y=complexity_metrics['unique_words'],
        mode='lines+markers',
        name='Unique Words'
    ))
    
    # Add self-reference metric if calculated
    if 'self_reference' in complexity_metrics:
        fig.add_trace(go.Scatter(
            x=list(range(len(steps))),
            y=complexity_metrics['self_reference'],
            mode='lines+markers',
            name='Self-Reference'
        ))
    
    fig.update_layout(
        title="Text Paradox Resolution Metrics",
        xaxis_title="Step",
        yaxis_title="Value",
        legend_title="Metrics"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def calculate_text_complexity(texts: List[str]) -> Dict[str, List[float]]:
    """
    Calculate complexity metrics for text statements.
    
    Args:
        texts: List of text statements
        
    Returns:
        Dictionary of metric names to lists of values
    """
    metrics = {
        'length': [len(t) for t in texts],
        'unique_words': [len(set(t.lower().split())) for t in texts],
    }
    
    # Calculate self-reference metric (occurrences of self-referential terms)
    self_ref_terms = ['this', 'itself', 'self', 'statement', 'sentence', 'paradox']
    self_ref_count = []
    
    for text in texts:
        count = sum(text.lower().count(term) for term in self_ref_terms)
        self_ref_count.append(count)
    
    metrics['self_reference'] = self_ref_count
    
    return metrics

def plot_convergence(steps: List[Any], threshold: float = 0.001):
    """
    Plot the convergence of the resolution process.
    
    Args:
        steps: List of states during resolution
        threshold: Convergence threshold used
    """
    if len(steps) < 2:
        st.warning("Not enough steps to analyze convergence.")
        return
    
    # Calculate differences between consecutive steps
    numeric_data = convert_steps_to_numeric(steps, 1)
    
    if numeric_data is None:
        st.warning("Cannot analyze convergence for this type of data.")
        return
    
    # Calculate differences between consecutive steps
    if len(numeric_data.shape) == 1:
        # 1D data
        diffs = [abs(numeric_data[i] - numeric_data[i-1]) for i in range(1, len(numeric_data))]
        max_diffs = diffs
    else:
        # Multi-dimensional data
        diffs = [np.abs(numeric_data[i] - numeric_data[i-1]) for i in range(1, len(numeric_data))]
        max_diffs = [np.max(diff) for diff in diffs]
    
    # Create convergence plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(steps))),
        y=max_diffs,
        mode='lines+markers',
        name='Max Difference'
    ))
    
    # Add threshold line
    fig.add_trace(go.Scatter(
        x=[1, len(steps)-1],
        y=[threshold, threshold],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name=f'Threshold ({threshold})'
    ))
    
    fig.update_layout(
        title="Convergence Analysis",
        xaxis_title="Step",
        yaxis_title="Maximum Difference",
        yaxis_type="log" if min(max_diffs) > 0 else "linear",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display convergence metrics
    final_diff = max_diffs[-1] if max_diffs else float('inf')
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(
            "Final Difference", 
            f"{final_diff:.6f}", 
            f"{final_diff - threshold:.6f}"
        )
    
    with metrics_col2:
        is_converged = final_diff <= threshold
        st.metric(
            "Convergence Status",
            "Converged" if is_converged else "Not Converged"
        )
    
    with metrics_col3:
        avg_diff_change = np.mean([max_diffs[i] - max_diffs[i-1] for i in range(1, len(max_diffs))]) if len(max_diffs) > 1 else 0
        steps_to_converge = "N/A"
        
        if not is_converged and avg_diff_change < 0:
            # Estimate steps to convergence if trending downward
            est_steps = int((final_diff - threshold) / abs(avg_diff_change)) + 1 if abs(avg_diff_change) > 0 else "∞"
            steps_to_converge = f"~{est_steps}"
        
        st.metric(
            "Est. Steps to Converge",
            "0" if is_converged else steps_to_converge
        )
