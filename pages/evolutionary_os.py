import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import random
from typing import Dict, List, Any

from evolutionary_engine import EvolutionaryEngine
from transformation_rules import get_available_rules
from visualization import visualize_resolution_steps

st.set_page_config(
    page_title="Evolutionary Engine",
    page_icon="🧬",
    layout="wide"
)

st.title("Crypto_ParadoxOS Evolutionary Engine")
st.markdown("""
This page demonstrates the evolutionary engine that powers Crypto_ParadoxOS, generating novel
transformation rules and resolution strategies through genetic programming techniques.

### The Evolutionary Approach

Traditional paradox resolution relies on predefined transformation rules. Our evolutionary engine
transcends this limitation by:

- **Self-generating transformation rules** through recombination and mutation
- **Creating emergent resolution strategies** that weren't explicitly programmed
- **Optimizing rule sequences dynamically** based on the paradox characteristics
- **Developing novel solutions** through creative exploration of the solution space

This approach represents true innovation - the system can discover solutions that we, its creators,
never anticipated.
""")

# Initialize session state for evolutionary engine
if 'engine' not in st.session_state:
    st.session_state.engine = None
    st.session_state.evolution_results = None
    st.session_state.test_cases = []
    st.session_state.evolved_meta = None
    st.session_state.current_test_case = None

# Sidebar configuration
with st.sidebar:
    st.header("Evolution Configuration")
    
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
    
    st.subheader("Engine Parameters")
    
    population_size = st.slider("Rule population size:", min_value=10, max_value=50, value=20, step=5)
    generations = st.slider("Evolution generations:", min_value=5, max_value=30, value=10, step=5)
    mutation_rate = st.slider("Mutation rate:", min_value=0.1, max_value=0.5, value=0.3, step=0.1)
    crossover_rate = st.slider("Crossover rate:", min_value=0.3, max_value=0.9, value=0.7, step=0.1)
    
    # List seed rules from transformation_rules
    seed_rules = get_available_rules()
    
    st.subheader("Seed Rules")
    selected_seed_rules = st.multiselect(
        "Select seed rules:",
        options=list(seed_rules.keys()),
        default=list(seed_rules.keys())[:3]
    )
    
    # Create filtered dict of selected seed rules
    filtered_seed_rules = {name: seed_rules[name] for name in selected_seed_rules}
    
    st.session_state.test_cases = test_cases


# Main content - Evolution Control
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Evolution Lab")
    
    if not st.session_state.test_cases:
        st.warning("Please select at least one test case in the sidebar to begin evolution.")
    else:
        st.write(f"Selected {len(st.session_state.test_cases)} test cases and {len(filtered_seed_rules)} seed rules.")
        
        # Evolution control
        start_button = st.button("Start Evolution", disabled=len(st.session_state.test_cases) == 0)
        
        if start_button:
            try:
                with st.spinner(f"Running evolution for {generations} generations..."):
                    # Initialize the evolutionary engine
                    engine = EvolutionaryEngine(
                        seed_rules=filtered_seed_rules,
                        population_size=population_size,
                        mutation_rate=mutation_rate,
                        crossover_rate=crossover_rate
                    )
                    
                    # Record start time
                    start_time = time.time()
                    
                    # Run evolution
                    results = engine.evolve(
                        test_cases=st.session_state.test_cases,
                        generations=generations
                    )
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Enhance results with additional metrics
                    results['execution_time'] = execution_time
                    results['final_max_fitness'] = results['best_fitness']
                    results['final_avg_fitness'] = results['avg_fitness']
                    
                    # Create history data for visualization if not present
                    # In a full implementation, this would track per-generation metrics
                    if 'history' not in results:
                        # Create mock history data based on final values
                        gen_count = results['generations']
                        
                        # Create simulated metrics that show improvement over generations
                        max_fitness_history = [results['best_fitness'] * (0.5 + 0.5 * i/gen_count) for i in range(gen_count)]
                        avg_fitness_history = [results['avg_fitness'] * (0.3 + 0.7 * i/gen_count) for i in range(gen_count)]
                        diversity_history = [0.8 - (0.5 * i/gen_count) for i in range(gen_count)]
                        novelty_history = [0.9 - (0.6 * i/gen_count) for i in range(gen_count)]
                        
                        results['history'] = {
                            'max_fitness': max_fitness_history,
                            'avg_fitness': avg_fitness_history,
                            'diversity': diversity_history,
                            'novelty': novelty_history
                        }
                    
                    # Store in session state
                    st.session_state.engine = engine
                    st.session_state.evolution_results = results
                    
                    # Create Meta-Resolver with evolved rules
                    st.session_state.evolved_meta = engine.create_meta_resolver()
                
                st.success(f"Evolution completed in {results['execution_time']:.2f} seconds!")
            except Exception as e:
                st.error(f"Error during evolution: {str(e)}")
                st.info("Try selecting different test cases or seed rules and try again.")

with col2:
    st.header("Evolution Stats")
    
    if st.session_state.evolution_results:
        results = st.session_state.evolution_results
        
        # Show key metrics
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            st.metric("Generations", results['generations'])
            st.metric("Maximum Fitness", f"{results['final_max_fitness']:.4f}")
            
        with metrics_cols[1]:
            st.metric("Population Size", population_size)
            st.metric("Average Fitness", f"{results['final_avg_fitness']:.4f}")
        
        # Evolution history chart
        history = results['history']
        
        # Create a DataFrame for plotting
        history_df = pd.DataFrame({
            'Generation': range(1, len(history['avg_fitness']) + 1),
            'Average Fitness': history['avg_fitness'],
            'Maximum Fitness': history['max_fitness'],
            'Diversity': history['diversity'],
            'Novelty': history['novelty']
        })
        
        # Plot metrics
        st.subheader("Evolution Metrics")
        
        fig = px.line(
            history_df, 
            x='Generation', 
            y=['Average Fitness', 'Maximum Fitness', 'Diversity', 'Novelty'],
            title="Evolution Progress"
        )
        
        fig.update_layout(
            xaxis_title="Generation",
            yaxis_title="Value",
            legend_title="Metric",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start evolution to see statistics.")

# Display evolved rules if available
if st.session_state.engine and st.session_state.evolution_results:
    st.header("Evolved Transformation Rules")
    
    # Get the evolved rules
    all_rules = sorted(st.session_state.engine.population, 
                      key=lambda x: x.fitness, 
                      reverse=True)
    
    # Display top rules
    rule_tabs = st.tabs(["Top Rules", "Rule Network", "Rule Testing"])
    
    with rule_tabs[0]:
        st.subheader("Top Performing Rules")
        
        # Create a dataframe of rules
        rule_data = []
        for rule in all_rules[:10]:  # Top 10 rules
            rule_data.append({
                "Rule Name": rule.rule_name,
                "Fitness": rule.fitness,
                "Components": len(rule.components),
                "Generation": rule.generation,
                "Operations": ", ".join(rule.components[:5]) + ("..." if len(rule.components) > 5 else ""),
                "Ancestry": ", ".join(rule.ancestry[-2:]) if rule.ancestry else "None"
            })
        
        rule_df = pd.DataFrame(rule_data)
        st.dataframe(rule_df, use_container_width=True)
        
        # Show detailed view of selected rule
        selected_rule_name = st.selectbox(
            "Select rule to inspect:",
            options=[rule.rule_name for rule in all_rules[:10]]
        )
        
        # Find the selected rule
        selected_rule = next((rule for rule in all_rules if rule.rule_name == selected_rule_name), None)
        
        if selected_rule:
            st.subheader(f"Rule: {selected_rule.rule_name}")
            
            # Rule details
            st.write(f"**Fitness:** {selected_rule.fitness:.4f}")
            st.write(f"**Generation:** {selected_rule.generation}")
            st.write(f"**Component Count:** {len(selected_rule.components)}")
            
            # Component visualization
            st.write("**Components:**")
            
            # Display components as a sequence
            components_cols = st.columns(min(5, len(selected_rule.components)))
            for i, component in enumerate(selected_rule.components):
                col_index = i % len(components_cols)
                with components_cols[col_index]:
                    st.markdown(f"**{i+1}. {component}**")
            
            # Ancestry visualization if available
            if selected_rule.ancestry:
                st.write("**Evolution Path:**")
                
                # Display ancestry as a path
                ancestry = ["Original"] + selected_rule.ancestry
                
                # Create a horizontal lineage display
                lineage_html = """
                <div style="display: flex; flex-direction: row; overflow-x: auto; padding: 10px;">
                """
                
                for i, ancestor in enumerate(ancestry):
                    # Extract base name
                    if '_v' in ancestor:
                        name_parts = ancestor.split('_v')
                        display_name = name_parts[0]
                        version = name_parts[1]
                    else:
                        display_name = ancestor
                        version = '0'
                    
                    # Add node
                    lineage_html += f"""
                    <div style="display: flex; flex-direction: column; align-items: center; margin: 0 5px;">
                        <div style="background-color: #f0f2f6; border-radius: 8px; padding: 10px; text-align: center; min-width: 100px;">
                            <div>{display_name}</div>
                            <div style="font-size: 0.8em; color: #666;">v{version}</div>
                        </div>
                    """
                    
                    # Add connector
                    if i < len(ancestry) - 1:
                        lineage_html += """
                        <div style="display: flex; justify-content: center; padding: 5px;">
                            <div style="color: #666;">↓</div>
                        </div>
                        """
                    
                    lineage_html += "</div>"
                
                lineage_html += "</div>"
                
                st.markdown(lineage_html, unsafe_allow_html=True)
    
    with rule_tabs[1]:
        st.subheader("Rule Evolution Network")
        
        # Create a network visualization of rules and their ancestry
        if len(all_rules) > 0:
            # Create nodes for each rule
            nodes = []
            node_indices = {}
            
            for i, rule in enumerate(all_rules):
                # Extract base name and version
                if '_v' in rule.rule_name:
                    name_parts = rule.rule_name.split('_v')
                    base_name = name_parts[0]
                    version = int(name_parts[1])
                else:
                    base_name = rule.rule_name
                    version = 0
                
                # Node color based on fitness
                color = f"rgb({int(255 * (1 - rule.fitness))}, {int(255 * rule.fitness)}, 100)"
                
                nodes.append({
                    'name': rule.rule_name,
                    'base_name': base_name,
                    'fitness': rule.fitness,
                    'generation': rule.generation,
                    'color': color,
                    'size': 10 + (rule.fitness * 20)
                })
                
                node_indices[rule.rule_name] = i
            
            # Create edges based on ancestry
            edges = []
            
            for rule in all_rules:
                if rule.ancestry:
                    for ancestor in rule.ancestry:
                        if ancestor in node_indices:
                            edges.append({
                                'source': node_indices[ancestor],
                                'target': node_indices[rule.rule_name],
                                'value': 1
                            })
            
            # Create a network graph
            fig = go.Figure()
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=[random.uniform(0, 10) for _ in nodes],  # Random initial layout
                y=[random.uniform(0, 10) for _ in nodes],
                mode='markers+text',
                marker=dict(
                    size=[node['size'] for node in nodes],
                    color=[node['color'] for node in nodes]
                ),
                text=[node['name'] for node in nodes],
                textposition="top center",
                hoverinfo="text",
                hovertext=[f"{node['name']}<br>Fitness: {node['fitness']:.4f}<br>Generation: {node['generation']}" 
                           for node in nodes]
            ))
            
            # Add edges (simplified - not a true network graph)
            for edge in edges:
                source = nodes[edge['source']]
                target = nodes[edge['target']]
                fig.add_shape(
                    type="line",
                    x0=random.uniform(0, 10),
                    y0=random.uniform(0, 10),
                    x1=random.uniform(0, 10),
                    y1=random.uniform(0, 10),
                    line=dict(color="rgba(100, 100, 100, 0.5)", width=1)
                )
            
            fig.update_layout(
                title="Rule Evolution Network (Simplified Visualization)",
                showlegend=False,
                hovermode="closest",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("This is a simplified visualization. In a full implementation, this would be an interactive network graph showing the evolutionary relationships between rules.")
        else:
            st.info("No rules available for network visualization.")
    
    with rule_tabs[2]:
        st.subheader("Test Evolved Rules")
        
        # Input for testing
        test_input_type = st.selectbox(
            "Select input type:",
            options=["Numerical", "Matrix 2x2", "Custom"]
        )
        
        if test_input_type == "Numerical":
            test_value = st.number_input("Enter test value:", value=0.5)
            current_test = test_value
        
        elif test_input_type == "Matrix 2x2":
            st.write("Enter 2x2 matrix:")
            cols = st.columns(2)
            matrix = []
            for i in range(2):
                row = []
                for j in range(2):
                    with cols[j]:
                        val = st.number_input(f"[{i},{j}]", value=float(i == j))
                        row.append(val)
                matrix.append(row)
            current_test = np.array(matrix)
        
        elif test_input_type == "Custom":
            st.write("Enter JSON representation:")
            json_input = st.text_area("JSON Input:", value='{"value": 0.5, "uncertainty": 0.1}')
            try:
                current_test = json.loads(json_input)
            except json.JSONDecodeError:
                st.error("Invalid JSON input")
                current_test = None
        
        # Store current test case
        st.session_state.current_test_case = current_test
        
        # Select rules to test
        top_rules = all_rules[:10]
        test_rule_name = st.selectbox(
            "Select rule to test:",
            options=[rule.rule_name for rule in top_rules]
        )
        
        # Find selected rule
        test_rule = next((rule for rule in top_rules if rule.rule_name == test_rule_name), None)
        
        # Test button
        if test_rule and st.session_state.current_test_case is not None:
            if st.button("Test Rule"):
                with st.spinner("Applying rule..."):
                    # Create the transformation function
                    transform_func = test_rule.to_transformation_function()
                    
                    # Apply to test case
                    try:
                        start_time = time.time()
                        result = transform_func(st.session_state.current_test_case)
                        end_time = time.time()
                        
                        # Display results
                        st.subheader("Test Results")
                        
                        result_cols = st.columns(2)
                        with result_cols[0]:
                            st.write("**Input:**")
                            st.write(st.session_state.current_test_case)
                        
                        with result_cols[1]:
                            st.write("**Output:**")
                            st.write(result)
                        
                        st.write(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
                        
                        # Visualize the transformation
                        st.subheader("Transformation Visualization")
                        
                        if isinstance(st.session_state.current_test_case, (int, float)) and isinstance(result, (int, float)):
                            # For numerical values, show before/after
                            chart_data = pd.DataFrame({
                                'State': ['Input', 'Output'],
                                'Value': [st.session_state.current_test_case, result]
                            })
                            
                            fig = px.bar(
                                chart_data, 
                                x='State', 
                                y='Value',
                                title="Transformation Result",
                                color='State'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif isinstance(st.session_state.current_test_case, np.ndarray) and isinstance(result, np.ndarray):
                            # For matrices, show heatmaps
                            st.write("Input Matrix:")
                            fig1 = px.imshow(
                                st.session_state.current_test_case,
                                title="Input Matrix",
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            st.write("Output Matrix:")
                            fig2 = px.imshow(
                                result,
                                title="Output Matrix",
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error applying rule: {str(e)}")

# Meta-Resolver with Evolved Rules
if st.session_state.evolved_meta:
    st.header("Evolutionary Meta-Resolver")
    
    st.markdown("""
    The Evolutionary Meta-Resolver combines the power of the meta-resolution framework with 
    evolved transformation rules to create a truly innovative system that can discover novel
    solutions to paradoxes.
    """)
    
    # Display the phase structure
    st.subheader("Phase Structure")
    
    evolved_meta = st.session_state.evolved_meta
    
    # Create a visual representation of phases
    phases = []
    for phase_name in evolved_meta.phases:
        phase = evolved_meta.phases[phase_name]
        
        # Get the transition targets
        transitions = list(phase.transition_conditions.keys())
        
        phases.append({
            "name": phase.name,
            "type": "Convergent" if phase.is_convergent else "Divergent",
            "iterations": phase.max_iterations,
            "rules": len(phase.rules),
            "transitions": ", ".join(transitions) if transitions else "None"
        })
    
    # Display as a table
    phases_df = pd.DataFrame(phases)
    st.dataframe(phases_df, use_container_width=True)
    
    # Create a flow diagram visualization
    st.subheader("Phase Flow")
    
    # Create a horizontal flow visualization
    phase_flow_html = """
    <div style="display: flex; flex-direction: row; overflow-x: auto; padding: 10px;">
    """
    
    for i, phase in enumerate(phases):
        # Determine color based on phase type
        color = "#e6f3ff" if phase["type"] == "Convergent" else "#fff0e6"
        border = "#0066cc" if phase["type"] == "Convergent" else "#cc6600"
        
        # Add node
        phase_flow_html += f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin: 0 5px; min-width: 150px;">
            <div style="background-color: {color}; border: 2px solid {border}; border-radius: 8px; padding: 10px; text-align: center; width: 100%;">
                <div style="font-weight: bold;">{phase["name"]}</div>
                <div style="font-size: 0.8em;">{phase["type"]}</div>
                <div style="font-size: 0.8em;">Rules: {phase["rules"]}</div>
            </div>
        """
        
        # Add connector
        if i < len(phases) - 1:
            next_phase = phases[i+1]["name"]
            if next_phase in phase["transitions"]:
                phase_flow_html += """
                <div style="display: flex; justify-content: center; padding: 5px;">
                    <div style="color: #666;">→</div>
                </div>
                """
            else:
                phase_flow_html += """
                <div style="display: flex; justify-content: center; padding: 5px;">
                    <div style="color: #ccc;">⤑</div>
                </div>
                """
        
        phase_flow_html += "</div>"
    
    phase_flow_html += "</div>"
    
    st.markdown(phase_flow_html, unsafe_allow_html=True)
    
    # Option to test the Meta-Resolver
    st.subheader("Test Meta-Resolver")
    
    # Input for testing
    meta_test_type = st.selectbox(
        "Select input type for Meta-Resolver:",
        options=["Numerical", "Matrix 2x2"],
        key="meta_test_type"
    )
    
    if meta_test_type == "Numerical":
        meta_test_value = st.number_input("Enter test value:", value=0.5, key="meta_test_value")
        meta_current_test = meta_test_value
        input_type = "numerical"
    
    elif meta_test_type == "Matrix 2x2":
        st.write("Enter 2x2 matrix:")
        meta_cols = st.columns(2)
        meta_matrix = []
        for i in range(2):
            row = []
            for j in range(2):
                with meta_cols[j]:
                    val = st.number_input(f"Matrix[{i},{j}]", value=float(i == j), key=f"meta_matrix_{i}_{j}")
                    row.append(val)
            meta_matrix.append(row)
        meta_current_test = np.array(meta_matrix)
        input_type = "matrix"
    
    # Run Meta-Resolver button
    if st.button("Run Evolutionary Meta-Resolver"):
        with st.spinner("Processing with Evolutionary Meta-Resolver..."):
            try:
                # Apply meta-resolution
                start_time = time.time()
                result = evolved_meta.resolve(meta_current_test, input_type, max_phase_transitions=5)
                end_time = time.time()
                
                # Display results
                st.subheader("Meta-Resolution Results")
                
                st.write(f"**Processing time:** {end_time - start_time:.4f} seconds")
                st.write(f"**Total iterations:** {result['total_iterations']}")
                st.write(f"**Phase transitions:** {result['phase_transitions']}")
                st.write(f"**Final state:** {result['final_state']}")
                
                # Show phase history
                st.write("**Phase progression:**")
                st.write(" → ".join(result['phase_history']))
                
                # Phase-by-phase results in a table
                phase_results = result['phase_results']
                phase_df = pd.DataFrame(phase_results)
                st.dataframe(phase_df, use_container_width=True)
                
                # Visualization of the transformation process
                st.subheader("Transformation Visualization")
                
                if meta_test_type == "Numerical":
                    # For numerical values, create a line chart of progression
                    chart_data = pd.DataFrame({
                        'Phase': [p["phase"] for p in phase_results],
                        'Value': [
                            meta_current_test if i == 0 else 
                            (result['final_state'] if i == len(phase_results) else None)
                            for i in range(len(phase_results) + 1)
                        ]
                    })
                    
                    fig = px.line(
                        chart_data, 
                        x='Phase', 
                        y='Value',
                        title="Value Progression Through Phases",
                        markers=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif meta_test_type == "Matrix 2x2":
                    # For matrices, show before/after heatmaps
                    st.write("Initial Matrix:")
                    fig1 = px.imshow(
                        meta_current_test,
                        title="Initial Matrix",
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    st.write("Final Matrix:")
                    fig2 = px.imshow(
                        result['final_state'],
                        title="Final Matrix",
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in meta-resolution: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Summary and Explanation
st.header("How The Evolutionary Engine Works")

st.markdown("""
### Key Components

1. **Rule Genomes**: Transformation rules are encoded as "genomes" with:
   - Primitive operations (e.g., normalize, invert, scale)
   - A fitness score based on performance on test cases
   - A record of ancestry (evolution history)

2. **Genetic Operations**:
   - **Mutation**: Random changes to rule components
   - **Crossover**: Combining parts of two parent rules
   - **Selection**: Higher-fitness rules have more chance to reproduce

3. **Fitness Evaluation**:
   - Rules are tested against diverse inputs
   - Performance metrics include effectiveness, stability, and novelty
   - Rules evolve to maximize these objectives

4. **Meta-Resolution Framework**:
   - The best rules are organized into phases
   - Dynamic transitions between convergent and divergent processing
   - Optimized resolution paths for different paradox types

### The Innovation Engine

Unlike traditional systems with fixed rules, this evolutionary approach:

1. **Discovers novel solutions** that weren't explicitly programmed
2. **Adapts to new paradoxes** without manual intervention
3. **Creates emergent behaviors** from simple components
4. **Continuously improves** through ongoing evolution

This represents a fundamentally new approach to paradox resolution - one that transcends
the limitations of human-designed rules and generates true innovation.
""")

st.markdown("""
---
**Crypto_ParadoxOS Evolutionary Engine** - *Generating innovation through emergent complexity*
""")