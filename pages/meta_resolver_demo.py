import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

from meta_resolver import MetaResolver, ResolutionPhase
from crypto_paradox_api import CryptoParadoxAPI
from visualization import visualize_resolution_steps

st.set_page_config(
    page_title="Meta-Resolver: Resolving the Core Paradox",
    page_icon="🔄",
    layout="wide"
)

st.title("Meta-Resolver: Resolution of the Core Paradox")
st.markdown("""
This page demonstrates the meta-resolution approach to solving the core paradox in Crypto_ParadoxOS:
the tension between **recursive resolution** (convergence) and **informational expansion** (divergence).

### The Core Paradox

The system faces a fundamental challenge:
- **Recursive Resolution**: Repeatedly apply transformations to reach an equilibrium state (convergence)
- **Informational Expansion**: Generate new insights and approaches through expansion (divergence)

These two goals are seemingly contradictory: one aims to reduce complexity, while the other increases it.
The Meta-Resolver framework transcends this contradiction by orchestrating dynamic transitions between
convergent and divergent phases.
""")

with st.sidebar:
    st.header("Meta-Resolution Configuration")
    
    st.subheader("Input Parameters")
    input_method = st.radio("Input method:", ["Numerical", "Matrix"], index=0)
    
    if input_method == "Numerical":
        initial_value = st.number_input("Initial value:", value=0.5, min_value=0.1, max_value=10.0, step=0.1)
        paradox_input = initial_value
        dimension = 1
    elif input_method == "Matrix":
        dimension = st.slider("Matrix dimension:", min_value=2, max_value=4, value=3, step=1)
        st.write("Enter matrix elements:")
        matrix_input = []
        for i in range(dimension):
            cols = st.columns(dimension)
            row = []
            for j in range(dimension):
                with cols[j]:
                    val = st.number_input(f"[{i},{j}]", value=float(i == j), format="%.2f", key=f"matrix_{i}_{j}")
                    row.append(val)
            matrix_input.append(row)
        paradox_input = np.array(matrix_input)
    
    st.subheader("Meta-Framework Configuration")
    max_transitions = st.slider("Max phase transitions:", min_value=1, max_value=20, value=5)
    
    # Pre-defined framework options
    framework_option = st.selectbox(
        "Resolution framework:",
        ["Standard", "Convergence Heavy", "Expansion Heavy", "Custom"]
    )

# Create dynamic visualization of the meta-resolution process
def visualize_meta_resolution(result, input_type="numerical"):
    # Create tabs for different visualizations
    tabs = st.tabs(["Phase Transitions", "Convergence Analysis", "State Evolution"])
    
    # Extract data safely with defaults
    phase_history = result.get("phase_history", [])
    phase_results = result.get("phase_results", [])
    
    with tabs[0]:
        st.subheader("Phase Transition Flow")
        
        if not phase_results:
            st.info("No phase transitions to display.")
        else:
            # Create a horizontal flow visualization
            phases_df = pd.DataFrame(phase_results)
            
            # Create a Sankey diagram for phase transitions
            if len(phase_history) > 1:
                # Create node labels
                unique_phases = list(set(phase_history))
                node_colors = ["blue" if "convergent" in p.lower() else "orange" if "divergent" in p.lower() else "green" for p in unique_phases]
                
                # Initialize Sankey diagram parameters
                pad = 15
                thickness = 20
                line = dict(color="black", width=0.5)
                color = node_colors
            
                # Create links between consecutive phases
                source = []
                target = []
                values = []
                
                for i in range(len(phase_history) - 1):
                    src_idx = unique_phases.index(phase_history[i])
                    tgt_idx = unique_phases.index(phase_history[i+1])
                    source.append(src_idx)
                    target.append(tgt_idx)
                    values.append(10)  # Fixed value for visualization
            
                # Create Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=pad,
                        thickness=thickness,
                        line=line,
                        label=unique_phases,
                        color=node_colors
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=values
                    )
                )])
            
            fig.update_layout(
                title_text="Phase Transition Flow",
                font_size=12,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show phase details in a table
        st.subheader("Phase Details")
        
        if not phase_results:
            st.info("No phase details to display.")
        else:
            # Prepare dataframe with essential columns
            try:
                # Ensure phases_df is defined
                if 'phases_df' not in locals():
                    phases_df = pd.DataFrame(phase_results)
                
                # Enhance the dataframe with more readable info
                if "is_convergent" in phases_df.columns:
                    phases_df["Convergent"] = phases_df["is_convergent"].apply(lambda x: "Yes" if x else "No")
                else:
                    phases_df["Convergent"] = "Unknown"
                
                if "converged" in phases_df.columns:
                    phases_df["Result"] = phases_df["converged"].apply(lambda x: "Converged" if x else "Did not converge")
                else:
                    phases_df["Result"] = "Unknown"
                
                # Show the table with selected columns
                columns_to_display = ["phase"]
                if "Convergent" in phases_df.columns:
                    columns_to_display.append("Convergent")
                if "iterations" in phases_df.columns:
                    columns_to_display.append("iterations")
                if "Result" in phases_df.columns:
                    columns_to_display.append("Result")
                
                st.dataframe(phases_df[columns_to_display], use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying phase details: {str(e)}")
                st.write(phases_df)
    
    with tabs[1]:
        st.subheader("Convergence Analysis")
        
        if not phase_results:
            st.info("No convergence data to display.")
        else:
            try:
                # Show convergence metrics for each phase
                fig = go.Figure()
                
                # Add bars for iterations
                fig.add_trace(go.Bar(
                    x=[p.get("phase", f"Phase {i}") for i, p in enumerate(phase_results)],
                    y=[p.get("iterations", 0) for p in phase_results],
                    name="Iterations",
                    marker_color=["blue" if p.get("is_convergent", True) else "orange" for p in phase_results]
                ))
                
                # Add points for convergence status (if available)
                if any("converged" in p for p in phase_results):
                    fig.add_trace(go.Scatter(
                        x=[p.get("phase", f"Phase {i}") for i, p in enumerate(phase_results)],
                        y=[p.get("iterations", 0) * 1.1 if p.get("converged", False) else 0 for p in phase_results],
                        mode="markers",
                        name="Converged",
                        marker=dict(
                            symbol="star",
                            size=12,
                            color=["green" if p.get("converged", False) else "red" for p in phase_results]
                        )
                    ))
                
                fig.update_layout(
                    title="Iterations and Convergence by Phase",
                    xaxis_title="Phase",
                    yaxis_title="Iterations",
                    legend_title="Legend",
                    barmode="group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying convergence analysis: {str(e)}")
                st.write("Raw phase data:", phase_results)
        
        # Show final result
        st.write(f"**Total Iterations:** {result.get('total_iterations', 'Unknown')}")
        st.write(f"**Final State:** {result.get('final_state', 'Unknown')}")
        st.write(f"**Meta-Converged:** {'Yes' if result.get('converged', False) else 'No'}")
    
    with tabs[2]:
        st.subheader("State Evolution")
        
        # Create a visualization of state evolution
        # This is simplified - in a real implementation we would track the state at each phase
        
        try:
            # For demonstration purposes, visualize the final state if available
            if 'final_state' in result:
                # Default dimension to 1 if not specified
                dimension = 1
                if input_type == "matrix" and isinstance(result['final_state'], (list, np.ndarray)):
                    # Estimate dimension from the final state if it's a matrix
                    dimension = 2
                
                # Import visualization function if not already imported
                from visualization import visualize_resolution_steps
                visualize_resolution_steps([result['final_state']], viz_type="Line chart", dimension=dimension)
            else:
                st.warning("No final state available to visualize.")
        except Exception as e:
            st.error(f"Error visualizing state evolution: {str(e)}")
        
        st.info("In a full implementation, this tab would show the evolution of the state across all phases.")


# Create and configure the meta-resolver based on selected option
def create_framework(option):
    meta = MetaResolver()
    
    if option == "Standard":
        return meta.create_standard_framework()
    
    elif option == "Convergence Heavy":
        # Create a convergence-focused framework
        initial = ResolutionPhase("Initial Assessment", is_convergent=True, max_iterations=5)
        deep_convergence = ResolutionPhase("Deep Convergence", is_convergent=True, max_iterations=25)
        maintenance = ResolutionPhase("Maintenance", is_convergent=True, max_iterations=10)
        
        # Add rules
        initial.add_rule("Fixed-Point Iteration").add_rule("Eigenvalue Stabilization")
        deep_convergence.add_rule("Fixed-Point Iteration").add_rule("Recursive Normalization")
        maintenance.add_rule("Fixed-Point Iteration").add_rule("Constraint Satisfaction")
        
        # Add transitions
        initial.add_transition("Deep Convergence", lambda x: True)
        deep_convergence.add_transition("Maintenance", lambda x: True)
        
        # Configure meta-resolver
        meta.add_phase(initial).add_phase(deep_convergence).add_phase(maintenance)
        meta.set_initial_phase("Initial Assessment")
        return meta
    
    elif option == "Expansion Heavy":
        # Create an expansion-focused framework
        initial = ResolutionPhase("Initial Assessment", is_convergent=True, max_iterations=5)
        divergence = ResolutionPhase("Divergent Expansion", is_convergent=False, max_iterations=15)
        creative = ResolutionPhase("Creative Transformation", is_convergent=False, max_iterations=10)
        integration = ResolutionPhase("Integration", is_convergent=True, max_iterations=10)
        
        # Add rules
        initial.add_rule("Fixed-Point Iteration")
        divergence.add_rule("Duality Inversion").add_rule("Bayesian Update")
        creative.add_rule("Self-Reference Unwinding").add_rule("Fuzzy Logic Transformation")
        integration.add_rule("Recursive Normalization").add_rule("Eigenvalue Stabilization")
        
        # Add transitions
        initial.add_transition("Divergent Expansion", lambda x: True)
        divergence.add_transition("Creative Transformation", lambda x: True)
        creative.add_transition("Integration", lambda x: True)
        
        # Configure meta-resolver
        meta.add_phase(initial).add_phase(divergence).add_phase(creative).add_phase(integration)
        meta.set_initial_phase("Initial Assessment")
        return meta
    
    elif option == "Custom":
        # Just use standard for now, but could be customized in a more advanced UI
        return meta.create_standard_framework()
    
    # Default case
    return meta.create_standard_framework()


# Main area with execution and visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Meta-Resolution Framework")
    
    # Visualize the framework structure based on selected option
    st.subheader(f"Selected Framework: {framework_option}")
    
    if framework_option == "Standard":
        st.image("https://via.placeholder.com/800x200?text=Standard+Framework+Visualization", use_container_width=True)
        st.markdown("""
        The Standard framework balances convergence and expansion:
        1. **Initial Convergence**: Apply basic stabilization rules
        2. **Information Expansion**: Generate new perspectives through divergent rules
        3. **Integration Refinement**: Integrate expanded information while reducing noise
        4. **Final Convergence**: Reach a stable equilibrium that balances both aspects
        """)
    
    elif framework_option == "Convergence Heavy":
        st.markdown("""
        The Convergence Heavy framework prioritizes stabilization:
        1. **Initial Assessment**: Quick evaluation of the paradox
        2. **Deep Convergence**: Extended iterative resolution toward equilibrium
        3. **Maintenance**: Final adjustments to ensure stability
        """)
    
    elif framework_option == "Expansion Heavy":
        st.markdown("""
        The Expansion Heavy framework prioritizes generative exploration:
        1. **Initial Assessment**: Quick evaluation of the paradox
        2. **Divergent Expansion**: Actively explore alternative perspectives
        3. **Creative Transformation**: Apply transformations that increase information content
        4. **Integration**: Consolidate the expanded perspectives into a coherent whole
        """)
    
    elif framework_option == "Custom":
        st.markdown("""
        Custom framework currently using the standard configuration.
        In a full implementation, users could customize:
        - Number and types of phases
        - Rules applied in each phase
        - Transition conditions between phases
        - Convergence/divergence parameters
        """)

with col2:
    st.header("Execute Resolution")
    
    if st.button("Resolve Core Paradox"):
        with st.spinner("Applying meta-resolution..."):
            # Create framework based on selected option
            meta_resolver = create_framework(framework_option)
            
            # Execute meta-resolution
            start_time = time.time()
            result = meta_resolver.resolve(
                paradox_input, 
                "numerical" if input_method == "Numerical" else "matrix",
                max_phase_transitions=max_transitions
            )
            end_time = time.time()
            
            # Display execution time
            st.success(f"Meta-resolution completed in {end_time - start_time:.4f} seconds")
            
            # Display basic results
            st.metric("Final Value", f"{result['final_state']:.6f}" if isinstance(result['final_state'], (int, float)) else "Matrix")
            st.metric("Phase Transitions", result['phase_transitions'])
            st.metric("Total Iterations", result['total_iterations'])

# Once resolution is complete, show detailed visualizations
if 'result' in locals():
    st.header("Meta-Resolution Analysis")
    visualize_meta_resolution(result, input_method.lower())
    
    st.markdown("""
    ### Resolution of the Core Paradox
    
    The meta-resolution framework successfully addresses the paradox between recursive resolution and informational
    expansion by orchestrating a dynamic process that leverages both approaches:
    
    1. **Transcending the Contradiction**: Rather than choosing between convergence and divergence, the framework
       embraces both as essential components of a higher-order process.
    
    2. **Phase-Based Approach**: By explicitly modeling the tension as different phases with different goals,
       the system can benefit from both recursive stability and creative expansion.
    
    3. **Adaptive Transitions**: The framework uses intelligent transition rules to determine when to
       switch between convergent and divergent processing modes.
    
    This approach demonstrates that apparent contradictions can be resolved at a higher level of organization
    when viewed as complementary rather than opposing forces.
    """)