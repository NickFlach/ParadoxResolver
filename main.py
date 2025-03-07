import streamlit as st
import numpy as np
import time
from paradox_resolver import ParadoxResolver
from transformation_rules import get_available_rules
from visualization import visualize_resolution_steps, plot_convergence
from utils import format_paradox_input, validate_input
from examples import get_example_paradoxes

st.set_page_config(
    page_title="Crypto_ParadoxOS",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Crypto_ParadoxOS: Recursive Paradox-Resolution System")
    st.markdown("""
    This system resolves logical and mathematical paradoxes through recursive transformation 
    until an equilibrium state is reached.
    """)
    
    # Header area with new features announcement
    st.info("""
    **🧬 NEW: Evolutionary Intelligence System**
    
    We've developed revolutionary approaches that generate novel solutions through genetic programming and human-AI collaboration:
    
    * [Interactive Evolution](/interactive_evolution) - Guide evolution with your own intuition and creativity
    * [Evolutionary OS](/evolutionary_os) - Our engine that evolves transformation rules and creates emergent innovation
    * [Meta-Resolver Demo](/meta_resolver_demo) - Our solution to balance recursive resolution and informational expansion
    * [API Documentation](/api_documentation) - For AI systems that want to integrate with Crypto_ParadoxOS
    """)
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Transformation Rules")
        available_rules = get_available_rules()
        selected_rules = st.multiselect(
            "Select transformation rules to apply:",
            options=list(available_rules.keys()),
            default=list(available_rules.keys())[:2]
        )
        
        st.subheader("Resolution Parameters")
        max_iterations = st.slider("Maximum iterations:", 1, 100, 20)
        convergence_threshold = st.slider("Convergence threshold:", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
        
        st.subheader("Visualization Options")
        show_step_details = st.checkbox("Show detailed steps", value=True)
        visualization_type = st.selectbox(
            "Visualization type:", 
            ["Line chart", "Bar chart", "Heatmap", "3D projection"]
        )
        
        # Add links to advanced features
        st.markdown("---")
        st.markdown("### Advanced Features")
        st.markdown("""
        * [Interactive Evolution](/interactive_evolution) - Guide the evolution process with your intuition
        * [Evolutionary OS](/evolutionary_os) - Discover our evolutionary engine that generates novel rules
        * [Meta-Resolver Demo](/meta_resolver_demo) - Explore our solution to the core paradox
        * [API Documentation](/api_documentation) - Integration guide for AI systems
        """)
    
    # Main content area
    tabs = st.tabs(["Input Paradox", "Examples", "About", "Core Paradox"])
    
    with tabs[0]:
        st.header("Enter Your Paradox")
        
        input_method = st.radio("Input method:", ["Text", "Numerical", "Matrix"])
        
        if input_method == "Text":
            paradox_input = st.text_area(
                "Enter your paradoxical statement:",
                "This statement is false."
            )
            dimension = 1
        
        elif input_method == "Numerical":
            col1, col2 = st.columns(2)
            with col1:
                paradox_input = st.text_input(
                    "Enter numerical paradox (e.g., recursive equation):",
                    "x = 1/x"
                )
                initial_value = st.number_input("Initial value:", value=0.5)
            dimension = 1
            
        elif input_method == "Matrix":
            dimension = st.slider("Matrix dimension:", min_value=2, max_value=5, value=3, step=1)
            matrix_input = []
            for i in range(dimension):
                row = []
                cols = st.columns(dimension)
                for j in range(dimension):
                    with cols[j]:
                        val = st.number_input(f"[{i},{j}]", value=float(i == j))
                        row.append(val)
                matrix_input.append(row)
            paradox_input = np.array(matrix_input)
            
        # Process the input when the user clicks the button
        if st.button("Resolve Paradox"):
            if input_method == "Text" and not paradox_input.strip():
                st.error("Please enter a paradoxical statement.")
            else:
                with st.spinner("Resolving paradox..."):
                    try:
                        # Format and validate the input
                        formatted_input = format_paradox_input(paradox_input, input_method, initial_value if input_method == "Numerical" else None)
                        is_valid, validation_msg = validate_input(formatted_input, input_method)
                        
                        if not is_valid:
                            st.error(validation_msg)
                        else:
                            # Create a resolver with the selected rules
                            selected_rule_functions = {name: available_rules[name] for name in selected_rules}
                            resolver = ParadoxResolver(
                                transformation_rules=selected_rule_functions,
                                max_iterations=max_iterations,
                                convergence_threshold=convergence_threshold
                            )
                            
                            # Resolve the paradox
                            start_time = time.time()
                            result, steps, converged = resolver.resolve(formatted_input)
                            end_time = time.time()
                            
                            # Display the results
                            st.subheader("Resolution Results")
                            st.write(f"Processing time: {end_time - start_time:.4f} seconds")
                            
                            if converged:
                                st.success(f"Paradox resolved in {len(steps)-1} iterations!")
                                st.write(f"Final equilibrium state: {result}")
                            else:
                                st.warning(f"Maximum iterations ({max_iterations}) reached without convergence.")
                                st.write(f"Current state: {result}")
                            
                            # Display step-by-step resolution
                            if show_step_details:
                                st.subheader("Resolution Steps")
                                for i, step in enumerate(steps):
                                    st.write(f"Step {i}: {step}")
                            
                            # Visualize the resolution process
                            st.subheader("Visualization")
                            visualize_resolution_steps(steps, visualization_type, dimension)
                            
                            # Show convergence plot
                            st.subheader("Convergence Analysis")
                            plot_convergence(steps, convergence_threshold)
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    with tabs[1]:
        st.header("Example Paradoxes")
        
        example_paradoxes = get_example_paradoxes()
        selected_example = st.selectbox(
            "Choose an example:",
            list(example_paradoxes.keys())
        )
        
        example = example_paradoxes[selected_example]
        st.write(f"**Description**: {example['description']}")
        st.write(f"**Type**: {example['type']}")
        st.write(f"**Paradox**: {example['paradox']}")
        
        if st.button("Load Example"):
            st.session_state.example_loaded = {
                'paradox': example['paradox'],
                'type': example['type']
            }
            st.rerun()
    
    with tabs[2]:
        st.header("About Crypto_ParadoxOS")
        st.markdown("""
        ### What is a Paradox?
        
        A paradox is a statement or situation that contradicts itself or defies intuition. In logic and mathematics, 
        paradoxes often involve self-reference, infinite recursion, or contradictory constraints.
        
        ### How the Resolution System Works
        
        Crypto_ParadoxOS uses a recursive approach to resolve paradoxes:
        
        1. It takes the initial paradoxical input
        2. Applies a series of transformation rules iteratively
        3. Tracks each state during iteration
        4. Continues until the system reaches an equilibrium or a maximum iteration count
        
        ### Transformation Rules
        
        The system supports multiple transformation rules that can be applied to different types of paradoxes:
        
        - **Contradiction Resolution**: Transforms logical contradictions into consistent states
        - **Fixed-Point Iteration**: Finds stable points for recursive equations
        - **Self-Reference Unwinding**: Resolves self-referential statements
        - **Eigenvalue Stabilization**: For matrix-based paradoxes
        - **Fuzzy Logic Transformation**: Applies many-valued logic to resolve binary contradictions
        
        ### Applications
        
        This system can be extended to address challenges in:
        
        - Decision-making under contradictory requirements
        - Resource allocation with competing priorities
        - Governance systems with conflicting incentives
        - AI ethics dilemmas
        - Cryptographic protocol design
        """)
    
    with tabs[3]:
        st.header("The Core Paradox")
        
        st.markdown("""
        ### The Fundamental Tension in Crypto_ParadoxOS
        
        Our system faces a deep architectural paradox between two competing goals:
        
        **Recursive Resolution (Convergence)** | **Informational Expansion (Divergence)**
        -------------------------------------- | ---------------------------------------
        Seeks to reach stable equilibrium | Generates new insights and approaches
        Reduces complexity over iterations | Increases information content
        Tends toward simplification | Tends toward elaboration
        Aims for deterministic outcomes | Embraces probabilistic exploration
        
        These two approaches appear fundamentally contradictory - we can't simultaneously
        converge and diverge, reduce and expand, simplify and elaborate.
        
        ### The Conventional Approach
        
        Traditional systems typically choose one approach:
        
        1. **Convergence-focused systems** like optimization algorithms prioritize finding
           stable solutions but may miss creative alternatives
        
        2. **Divergence-focused systems** like generative AI generate novel outputs
           but may struggle with consistency and stability
        
        ### Our Solution: The Meta-Resolver Framework
        
        We've developed a meta-level solution that transcends this contradiction by:
        
        1. **Embracing both approaches** as essential and complementary
        
        2. **Orchestrating dynamic transitions** between convergent and divergent phases
        
        3. **Employing phase-specific transformation rules** tailored to each mode
        
        4. **Using intelligent transition conditions** to determine when to switch modes
        
        This approach allows us to harmonize these competing forces into a unified framework
        that leverages the strengths of both approaches while mitigating their weaknesses.
        
        [Try the Meta-Resolver Demo](/meta_resolver_demo) to see this approach in action.
        """)

if __name__ == "__main__":
    main()
