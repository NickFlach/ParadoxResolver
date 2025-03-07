import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import random
from typing import Dict, List, Any, Tuple

from evolutionary_engine import EvolutionaryEngine, RuleGenome
from transformation_rules import get_available_rules
from visualization import visualize_resolution_steps

st.set_page_config(
    page_title="Interactive Evolution",
    page_icon="👤",
    layout="wide"
)

# Initialize session state
if "interactive_engine" not in st.session_state:
    st.session_state.interactive_engine = None
    st.session_state.current_population = []
    st.session_state.evolution_history = []
    st.session_state.user_selected_rules = []
    st.session_state.generation = 0
    st.session_state.test_cases = []
    st.session_state.explanation_mode = "basic"  # Can be "basic", "detailed", or "technical"

st.title("Interactive Evolution Lab")
st.markdown("""
## Guided Evolution of Transformation Rules

This interface allows you to participate in the evolution process, guiding the system toward
solutions that you find most promising. By combining human intuition with computational evolution,
we can discover more innovative and effective solutions to complex problems.

### How It Works

1. The system generates an initial population of transformation rules
2. You evaluate and select the rules that seem most promising
3. The system evolves the next generation based on your feedback
4. The process repeats, with each generation improving based on your guidance

This human-in-the-loop approach represents a more intuitive way to navigate the solution space
and can lead to unexpected breakthroughs that neither humans nor machines might discover alone.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Evolution Setup")
    
    # Explanation mode
    st.subheader("Explanation Level")
    explanation_mode = st.radio(
        "Select explanation detail:",
        options=["Basic", "Detailed", "Technical"],
        index=1  # Default to "Detailed"
    )
    st.session_state.explanation_mode = explanation_mode.lower()
    
    # Test case configuration
    st.subheader("Test Cases")
    
    # Define potential test cases
    test_case_options = {
        "Numeric (0.5)": 0.5,
        "Numeric (-1.0)": -1.0,
        "Identity Matrix 2x2": np.eye(2),
        "Random Matrix 2x2": np.random.rand(2, 2),
        "List [0.1, 0.2, 0.3, 0.4]": [0.1, 0.2, 0.3, 0.4],
        "Complex Structure": {"value": 0.5, "uncertainty": 0.1}
    }
    
    selected_cases = st.multiselect(
        "Select test cases for evolution:",
        options=list(test_case_options.keys()),
        default=["Numeric (0.5)", "Identity Matrix 2x2"]
    )
    
    # Extract selected test cases
    test_cases = [test_case_options[case] for case in selected_cases]
    st.session_state.test_cases = test_cases
    
    # Evolution parameters
    st.subheader("Evolution Parameters")
    
    population_size = st.slider("Population size:", min_value=4, max_value=20, value=8, step=2)
    mutation_rate = st.slider("Mutation rate:", min_value=0.1, max_value=0.5, value=0.3, step=0.1)
    
    # Seed rules
    seed_rules = get_available_rules()
    
    st.subheader("Starting Rules")
    selected_seed_rules = st.multiselect(
        "Select seed rules:",
        options=list(seed_rules.keys()),
        default=list(seed_rules.keys())[:2]
    )
    
    # Create filtered dict of selected seed rules
    filtered_seed_rules = {name: seed_rules[name] for name in selected_seed_rules}

# Main content area
st.header("Evolution Laboratory")

if not st.session_state.test_cases:
    st.warning("Please select at least one test case in the sidebar to begin evolution.")
else:
    # Start button (only show if engine not initialized)
    if st.session_state.interactive_engine is None:
        start_col1, start_col2 = st.columns([3, 1])
        with start_col1:
            st.markdown("""
            Start the interactive evolution process with the selected parameters. 
            The system will generate an initial population of rules that you can evaluate.
            """)
        with start_col2:
            if st.button("Start Evolution", use_container_width=True):
                with st.spinner("Initializing evolutionary engine..."):
                    try:
                        # Initialize engine with specified parameters
                        engine = EvolutionaryEngine(
                            seed_rules=filtered_seed_rules,
                            population_size=population_size,
                            mutation_rate=mutation_rate,
                            crossover_rate=0.7
                        )
                        
                        # Store in session state
                        st.session_state.interactive_engine = engine
                        st.session_state.current_population = engine.rule_population
                        st.session_state.generation = 0
                        
                        # Evaluate initial population
                        for genome in st.session_state.current_population:
                            engine._evaluate_fitness(genome, st.session_state.test_cases)
                        
                        st.success("Evolution engine initialized successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error initializing evolution engine: {str(e)}")
    
    # If engine is initialized, show the interactive interface
    if st.session_state.interactive_engine is not None:
        # Display current generation info
        generation_col1, generation_col2 = st.columns([2, 1])
        with generation_col1:
            st.subheader(f"Generation {st.session_state.generation}")
            
            if st.session_state.explanation_mode in ["detailed", "technical"]:
                if st.session_state.generation == 0:
                    st.markdown("""
                    This is the initial population. You're seeing the starting rules, some of which
                    were provided as seeds and others randomly generated. Evaluate each rule by
                    testing it on different inputs, then select the ones you find most promising.
                    """)
                else:
                    st.markdown(f"""
                    This is generation {st.session_state.generation}, evolved based on your previous selections.
                    Notice how rules have combined and mutated from earlier generations, inheriting
                    characteristics from parents you selected previously.
                    """)
        
        with generation_col2:
            # Evolution metrics if we have history
            if len(st.session_state.evolution_history) > 0:
                avg_fitness = np.mean([r.fitness for r in st.session_state.current_population])
                max_fitness = max([r.fitness for r in st.session_state.current_population])
                
                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    st.metric("Avg Fitness", f"{avg_fitness:.2f}")
                with metrics_cols[1]:
                    st.metric("Max Fitness", f"{max_fitness:.2f}")
        
        # Display current population for evaluation
        st.markdown("### Evaluate and Select Rules")
        
        if st.session_state.explanation_mode == "detailed":
            st.info("""
            Test each rule on different inputs to see how it behaves, then select the rules 
            you want to include in the next generation. These will be used as parents for
            breeding a new population through mutation and crossover.
            """)
        
        # Function to show rule explanation based on components
        def explain_rule(rule: RuleGenome) -> str:
            """Generate an explanation of what a rule does based on its components."""
            if not rule.components:
                return "This rule doesn't modify the input (identity function)."
                
            # Basic explanation templates
            templates = {
                "identity": "keeps the input unchanged",
                "inverse": "inverts the value (1/x)",
                "square": "squares the value",
                "sqrt": "takes the square root",
                "normalize": "normalizes values to a standard range",
                "dampening": "reduces the magnitude by 10%",
                "oscillate": "applies a sine wave transformation",
                "truncate": "truncates to integer values",
                "inflate": "increases the magnitude by 10%",
                "reflect": "changes the sign",
                "shift": "adds a small constant",
                "scale": "doubles the value",
                "merge": "combines multiple values into one",
                "split": "splits a value into multiple parts",
                "filter": "removes small values",
                "smooth": "smooths out variations",
                "randomize": "adds random variation",
                "reorganize": "changes the structure",
                "compose": "applies a complex non-linear transformation",
                "extract": "extracts specific elements"
            }
            
            # Generate explanation based on components
            if len(rule.components) == 1:
                return f"This rule {templates.get(rule.components[0], 'applies a transformation')}."
                
            elif len(rule.components) <= 3:
                steps = [f"{i+1}) {templates.get(comp, 'transforms the data')}" 
                         for i, comp in enumerate(rule.components)]
                return f"This rule applies multiple steps: {'; '.join(steps)}."
                
            else:
                main_ops = [templates.get(comp, "transforms") for comp in rule.components[:3]]
                return f"This rule applies a sequence of {len(rule.components)} operations, including: {', '.join(main_ops)}, and more."
        
        # Display rules in a grid for evaluation
        rules_per_row = min(2, len(st.session_state.current_population))
        
        # Use columns to create a grid layout
        selected_rules = []
        
        # Create a dataframe for all rules
        rule_data = []
        for rule in st.session_state.current_population:
            # Create summary for this rule
            if isinstance(rule.fitness, float):
                fitness_display = f"{rule.fitness:.4f}"
            else:
                fitness_display = "Not evaluated"
                
            rule_data.append({
                "Rule Name": rule.rule_name,
                "Fitness": rule.fitness,
                "Fitness Display": fitness_display,
                "Components": len(rule.components),
                "Generation": rule.generation,
                "Explanation": explain_rule(rule),
                "Ancestry": ", ".join(rule.ancestry[-2:]) if rule.ancestry else "None"
            })
        
        # Convert to DataFrame
        rules_df = pd.DataFrame(rule_data)
        
        # Sort by fitness
        rules_df = rules_df.sort_values("Fitness", ascending=False)
        
        # Create tabs for different ways to view rules
        view_tabs = st.tabs(["Card View", "Table View", "Comparison View"])
        
        with view_tabs[0]:
            # Card view - display rules as cards in a grid
            for i in range(0, len(rules_df), rules_per_row):
                cols = st.columns(rules_per_row)
                for j in range(rules_per_row):
                    idx = i + j
                    if idx < len(rules_df):
                        rule_info = rules_df.iloc[idx]
                        with cols[j]:
                            st.markdown(f"#### {rule_info['Rule Name']}")
                            st.markdown(f"**Fitness:** {rule_info['Fitness Display']}")
                            
                            # Expandable details
                            with st.expander("Rule Details"):
                                st.markdown(f"**Explanation:** {rule_info['Explanation']}")
                                st.markdown(f"**Components:** {rule_info['Components']}")
                                if rule_info['Ancestry'] != "None":
                                    st.markdown(f"**Ancestry:** {rule_info['Ancestry']}")
                            
                            # Test this rule on different inputs
                            with st.expander("Test Rule"):
                                test_input_type = st.selectbox(
                                    "Input type:",
                                    options=["Numerical", "Matrix 2x2", "Custom"],
                                    key=f"input_type_{idx}"
                                )
                                
                                if test_input_type == "Numerical":
                                    test_value = st.number_input(
                                        "Enter value:", 
                                        value=0.5,
                                        key=f"num_input_{idx}"
                                    )
                                    current_test = test_value
                                    
                                elif test_input_type == "Matrix 2x2":
                                    st.write("Enter 2x2 matrix:")
                                    test_cols = st.columns(2)
                                    matrix = []
                                    for k in range(2):
                                        row = []
                                        for l in range(2):
                                            with test_cols[l]:
                                                val = st.number_input(
                                                    f"[{k},{l}]", 
                                                    value=float(k == l),
                                                    key=f"matrix_{idx}_{k}_{l}"
                                                )
                                                row.append(val)
                                        matrix.append(row)
                                    current_test = np.array(matrix)
                                    
                                elif test_input_type == "Custom":
                                    json_input = st.text_area(
                                        "JSON Input:", 
                                        value='{"value": 0.5, "uncertainty": 0.1}',
                                        key=f"json_input_{idx}"
                                    )
                                    try:
                                        current_test = json.loads(json_input)
                                    except json.JSONDecodeError:
                                        st.error("Invalid JSON")
                                        current_test = None
                                
                                # Create the actual rule from the population
                                rule = st.session_state.current_population[rules_df.index[idx]]
                                
                                # Test button
                                if st.button("Test", key=f"test_button_{idx}"):
                                    try:
                                        # Get transformation function
                                        transform_func = rule.to_transformation_function()
                                        
                                        # Apply to test case
                                        result = transform_func(current_test)
                                        
                                        # Display result
                                        st.write("**Result:**")
                                        st.write(result)
                                        
                                        # Visualize if numerical
                                        if isinstance(current_test, (int, float)) and isinstance(result, (int, float)):
                                            chart_data = pd.DataFrame({
                                                'State': ['Input', 'Output'],
                                                'Value': [current_test, result]
                                            })
                                            
                                            fig = px.bar(
                                                chart_data, 
                                                x='State', 
                                                y='Value',
                                                title="Transformation",
                                                color='State'
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                            
                            # Select checkbox
                            selected = st.checkbox(
                                "Select for next generation", 
                                key=f"select_{idx}"
                            )
                            if selected:
                                selected_rules.append(rule)
        
        with view_tabs[1]:
            # Table view - display rules in a table
            st.dataframe(
                rules_df[["Rule Name", "Fitness Display", "Components", "Generation", "Explanation"]],
                use_container_width=True
            )
            
            # Multi-select for table view
            selected_rule_names = st.multiselect(
                "Select rules for next generation:",
                options=rules_df["Rule Name"].tolist(),
                default=[]
            )
            
            # Add selected rules to the list
            for rule_name in selected_rule_names:
                # Find the rule in the current population
                for rule in st.session_state.current_population:
                    if rule.rule_name == rule_name and rule not in selected_rules:
                        selected_rules.append(rule)
        
        with view_tabs[2]:
            # Comparison view - allow side-by-side comparison
            st.markdown("### Compare Rules")
            
            # Select rules to compare
            compare_cols = st.columns(2)
            
            with compare_cols[0]:
                rule1_name = st.selectbox(
                    "Select first rule:",
                    options=rules_df["Rule Name"].tolist(),
                    key="compare_rule1"
                )
                rule1_idx = rules_df[rules_df["Rule Name"] == rule1_name].index[0]
                rule1 = st.session_state.current_population[rule1_idx]
                
                st.markdown(f"**Fitness:** {rules_df[rules_df['Rule Name'] == rule1_name]['Fitness Display'].values[0]}")
                st.markdown(f"**Explanation:** {explain_rule(rule1)}")
                
                components_str = ", ".join(rule1.components[:5])
                if len(rule1.components) > 5:
                    components_str += "..."
                st.markdown(f"**Components:** {components_str}")
            
            with compare_cols[1]:
                rule2_name = st.selectbox(
                    "Select second rule:",
                    options=rules_df["Rule Name"].tolist(),
                    key="compare_rule2"
                )
                rule2_idx = rules_df[rules_df["Rule Name"] == rule2_name].index[0]
                rule2 = st.session_state.current_population[rule2_idx]
                
                st.markdown(f"**Fitness:** {rules_df[rules_df['Rule Name'] == rule2_name]['Fitness Display'].values[0]}")
                st.markdown(f"**Explanation:** {explain_rule(rule2)}")
                
                components_str = ", ".join(rule2.components[:5])
                if len(rule2.components) > 5:
                    components_str += "..."
                st.markdown(f"**Components:** {components_str}")
            
            # Test both rules on the same input
            st.markdown("### Test Both Rules")
            
            test_input_type = st.selectbox(
                "Input type:",
                options=["Numerical", "Matrix 2x2", "Custom"],
                key="compare_input_type"
            )
            
            if test_input_type == "Numerical":
                test_value = st.number_input("Enter value:", value=0.5, key="compare_num_input")
                current_test = test_value
                
            elif test_input_type == "Matrix 2x2":
                st.write("Enter 2x2 matrix:")
                test_cols = st.columns(2)
                matrix = []
                for i in range(2):
                    row = []
                    for j in range(2):
                        with test_cols[j]:
                            val = st.number_input(f"[{i},{j}]", value=float(i == j), key=f"compare_matrix_{i}_{j}")
                            row.append(val)
                    matrix.append(row)
                current_test = np.array(matrix)
                
            elif test_input_type == "Custom":
                json_input = st.text_area(
                    "JSON Input:", 
                    value='{"value": 0.5, "uncertainty": 0.1}',
                    key="compare_json_input"
                )
                try:
                    current_test = json.loads(json_input)
                except json.JSONDecodeError:
                    st.error("Invalid JSON")
                    current_test = None
            
            # Test button
            if st.button("Compare Rules", key="compare_button"):
                try:
                    # Get transformation functions
                    transform_func1 = rule1.to_transformation_function()
                    transform_func2 = rule2.to_transformation_function()
                    
                    # Apply to test case
                    result1 = transform_func1(current_test)
                    result2 = transform_func2(current_test)
                    
                    # Display results side by side
                    result_cols = st.columns(2)
                    
                    with result_cols[0]:
                        st.markdown(f"**{rule1_name} Result:**")
                        st.write(result1)
                    
                    with result_cols[1]:
                        st.markdown(f"**{rule2_name} Result:**")
                        st.write(result2)
                    
                    # Visualize if numerical
                    if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
                        chart_data = pd.DataFrame({
                            'Rule': [rule1_name, rule2_name],
                            'Output': [result1, result2]
                        })
                        
                        fig = px.bar(
                            chart_data, 
                            x='Rule', 
                            y='Output',
                            title="Comparison of Results",
                            color='Rule'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            # Add buttons to select either or both for next generation
            selection_cols = st.columns(3)
            
            with selection_cols[0]:
                if st.button(f"Select {rule1_name}", key="select_rule1"):
                    if rule1 not in selected_rules:
                        selected_rules.append(rule1)
                        st.success(f"Added {rule1_name} to selection")
            
            with selection_cols[1]:
                if st.button(f"Select {rule2_name}", key="select_rule2"):
                    if rule2 not in selected_rules:
                        selected_rules.append(rule2)
                        st.success(f"Added {rule2_name} to selection")
            
            with selection_cols[2]:
                if st.button("Select Both", key="select_both"):
                    if rule1 not in selected_rules:
                        selected_rules.append(rule1)
                    if rule2 not in selected_rules:
                        selected_rules.append(rule2)
                    st.success("Added both rules to selection")
        
        # Show selected rules
        st.markdown("### Selected Rules")
        if selected_rules:
            selected_names = [rule.rule_name for rule in selected_rules]
            st.write(f"You've selected {len(selected_rules)} rules: {', '.join(selected_names)}")
        else:
            st.warning("You haven't selected any rules yet. Please select at least one rule to continue evolution.")
        
        # Next generation button
        next_gen_col1, next_gen_col2 = st.columns([3, 1])
        with next_gen_col1:
            if st.session_state.explanation_mode in ["detailed", "technical"]:
                st.markdown("""
                Once you've selected the rules you want to keep, click 'Evolve Next Generation' to create 
                a new population. The system will use your selected rules as parents to breed a new
                generation through mutation and crossover, preserving the characteristics you value.
                """)
        
        with next_gen_col2:
            if st.button("Evolve Next Generation", disabled=len(selected_rules) == 0, use_container_width=True):
                with st.spinner("Evolving next generation..."):
                    try:
                        engine = st.session_state.interactive_engine
                        
                        # Store current population in history
                        st.session_state.evolution_history.append(st.session_state.current_population)
                        
                        # Store selected rules for reference
                        st.session_state.user_selected_rules.append(selected_rules)
                        
                        # Create next generation
                        next_generation = []
                        
                        # Always keep the selected rules (elitism)
                        next_generation.extend(selected_rules)
                        
                        # Fill the rest with offspring
                        while len(next_generation) < population_size:
                            # Select random parents from selected rules
                            parent1 = random.choice(selected_rules)
                            parent2 = random.choice(selected_rules)
                            
                            # Create child through crossover
                            child = parent1.crossover(parent2)
                            
                            # Potentially mutate
                            if random.random() < mutation_rate:
                                child = child.mutate()
                            
                            next_generation.append(child)
                        
                        # Evaluate new generation
                        for genome in next_generation:
                            engine._evaluate_fitness(genome, st.session_state.test_cases)
                        
                        # Update session state
                        st.session_state.current_population = next_generation
                        st.session_state.generation += 1
                        
                        st.success(f"Generated new population (Generation {st.session_state.generation})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error evolving next generation: {str(e)}")
        
        # Show evolution history if available
        if len(st.session_state.evolution_history) > 0:
            st.header("Evolution History")
            
            # Create metrics for tracking progress
            history_data = []
            
            for i, population in enumerate(st.session_state.evolution_history):
                avg_fitness = np.mean([r.fitness for r in population])
                max_fitness = max([r.fitness for r in population])
                diversity = st.session_state.interactive_engine._calculate_diversity()
                
                history_data.append({
                    "Generation": i,
                    "Average Fitness": avg_fitness,
                    "Maximum Fitness": max_fitness,
                    "Diversity": diversity
                })
            
            # Add current generation
            current_avg = np.mean([r.fitness for r in st.session_state.current_population])
            current_max = max([r.fitness for r in st.session_state.current_population])
            current_diversity = st.session_state.interactive_engine._calculate_diversity()
            
            history_data.append({
                "Generation": st.session_state.generation,
                "Average Fitness": current_avg,
                "Maximum Fitness": current_max,
                "Diversity": current_diversity
            })
            
            # Create dataframe
            history_df = pd.DataFrame(history_data)
            
            # Plot evolution progress
            st.subheader("Evolution Metrics")
            
            fig = px.line(
                history_df, 
                x="Generation", 
                y=["Average Fitness", "Maximum Fitness", "Diversity"],
                title="Evolution Progress"
            )
            
            fig.update_layout(
                xaxis_title="Generation",
                yaxis_title="Value",
                legend_title="Metric",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # User selections history
            if st.session_state.explanation_mode in ["detailed", "technical"]:
                st.subheader("Your Selection History")
                
                for i, selected in enumerate(st.session_state.user_selected_rules):
                    st.markdown(f"**Generation {i}:** You selected {len(selected)} rules: {', '.join([r.rule_name for r in selected])}")

# Information about interactive evolution
st.header("About Interactive Evolution")

st.markdown("""
### The Power of Human-Guided Evolution

Traditional evolutionary algorithms rely solely on predefined fitness functions to guide the evolution process.
While effective, this approach misses out on the intuitive pattern recognition and creativity that humans bring.

Interactive evolution combines the computational power of evolutionary algorithms with human intuition by:

1. **Leveraging Human Evaluation**: Humans can recognize promising patterns that are difficult to capture in a fitness function
2. **Incorporating Subjective Criteria**: Quality, aesthetics, and "interestingness" can be evaluated by humans
3. **Accelerating Convergence**: Human selection can guide evolution toward promising areas of the solution space
4. **Facilitating Discoveries**: The combination often leads to unexpected and creative solutions

This hybrid approach has proven effective in domains ranging from art and music to complex engineering problems
and scientific discovery.

### Applications in Paradox Resolution

In the context of Crypto_ParadoxOS, interactive evolution allows you to guide the development of transformation
rules that can resolve paradoxes in ways that might not be discovered through automated evolution alone.

By selecting rules that exhibit interesting behaviors or approach problems from novel angles, you're 
steering the system toward more innovative and effective solutions.
""")

# Reset button at the bottom
if st.session_state.interactive_engine is not None:
    if st.button("Reset Evolution"):
        # Clear session state
        st.session_state.interactive_engine = None
        st.session_state.current_population = []
        st.session_state.evolution_history = []
        st.session_state.user_selected_rules = []
        st.session_state.generation = 0
        st.success("Evolution reset. You can start a new evolution process.")
        st.rerun()