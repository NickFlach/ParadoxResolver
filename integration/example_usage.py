"""
Example Usage of Crypto_ParadoxOS Integrations

This script demonstrates how to use the various integrations to connect
Crypto_ParadoxOS with MusicPortal and SIN.
"""

import sys
import logging
import numpy as np
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("integration_examples")

# Import integration modules
from musicportal.shinobi_integration import create_shinobi_integration, MusicStructure
from musicportal.lumira_integration import create_lumira_integration, SoundParameters
from sin.reasoning_integration import create_reasoning_integration, LogicalSystem, LogicalStatement
from sin.pattern_recognition_integration import create_pattern_recognition_integration, PatternInstance, PatternSet


def demo_shinobi_integration():
    """Demonstrate the integration with MusicPortal's Shinobi composition engine."""
    logger.info("Demonstrating Shinobi integration")
    
    # Create integration with custom configuration
    shinobi = create_shinobi_integration(
        use_evolution=True,
        max_iterations=50
    )
    
    # Create an initial music structure
    initial_structure = MusicStructure(
        sections=[
            {"intensity": 0.3, "duration": 8, "complexity": 0.2, "tension": 0.1, "resolution": 0.9},
            {"intensity": 0.5, "duration": 16, "complexity": 0.4, "tension": 0.5, "resolution": 0.5},
            {"intensity": 0.8, "duration": 8, "complexity": 0.7, "tension": 0.8, "resolution": 0.3},
        ],
        tempo=120,
        key="F minor",
        time_signature=(4, 4)
    )
    
    # Use the integration to compose a structure
    composed_structure = shinobi.compose_structure(
        initial_structure=initial_structure,
        style_influences=["orchestral", "minimalist"],
        complexity_target=0.6
    )
    
    # Generate variations of the composed structure
    variations = shinobi.suggest_variations(
        structure=composed_structure,
        count=2,
        variation_degree=0.4
    )
    
    # Merge the original and a variation
    merged_structure = shinobi.merge_structures(
        structures=[composed_structure, variations[0]],
        weights=[0.7, 0.3]
    )
    
    # Display results
    logger.info(f"Original structure had {len(initial_structure.sections)} sections")
    logger.info(f"Composed structure has {len(composed_structure.sections)} sections")
    logger.info(f"Generated {len(variations)} variations")
    logger.info(f"Merged structure has {len(merged_structure.sections)} sections")
    
    # Print section details for the original and composed structures
    logger.info("\nOriginal Structure Sections:")
    for i, section in enumerate(initial_structure.sections):
        logger.info(f"  Section {i+1}: {section}")
    
    logger.info("\nComposed Structure Sections:")
    for i, section in enumerate(composed_structure.sections):
        logger.info(f"  Section {i+1}: {section}")
    
    return shinobi, composed_structure


def demo_lumira_integration():
    """Demonstrate the integration with MusicPortal's Lumira sound processing system."""
    logger.info("\nDemonstrating Lumira integration")
    
    # Create integration
    lumira = create_lumira_integration(
        max_iterations=30,
        convergence_threshold=0.001
    )
    
    # Create initial sound parameters
    initial_params = SoundParameters(
        frequency_envelope=[1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
        amplitude_envelope=[0.0, 0.4, 0.8, 1.0, 0.7, 0.3, 0.0],
        filter_settings={"cutoff": 0.7, "resonance": 0.3, "gain": 0.5, "q": 0.4},
        effects={"reverb": 0.4, "delay": 0.2, "distortion": 0.1, "chorus": 0.3},
        spatial_position=(0.2, 0.0, 0.3)
    )
    
    # Transform the sound parameters
    transformed_params = lumira.transform_sound(
        parameters=initial_params,
        transformation_intensity=0.6,
        target_aesthetic="ambient"
    )
    
    # Generate interpolation between original and transformed parameters
    interpolated_params = lumira.interpolate_parameters(
        start_params=initial_params,
        end_params=transformed_params,
        steps=5,
        smoothing=0.3
    )
    
    # Display results
    logger.info("\nOriginal Parameters:")
    logger.info(f"  Frequency Envelope: {initial_params.frequency_envelope}")
    logger.info(f"  Amplitude Envelope: {initial_params.amplitude_envelope}")
    logger.info(f"  Filter Settings: {initial_params.filter_settings}")
    logger.info(f"  Effects: {initial_params.effects}")
    logger.info(f"  Spatial Position: {initial_params.spatial_position}")
    
    logger.info("\nTransformed Parameters:")
    logger.info(f"  Frequency Envelope: {transformed_params.frequency_envelope}")
    logger.info(f"  Amplitude Envelope: {transformed_params.amplitude_envelope}")
    logger.info(f"  Filter Settings: {transformed_params.filter_settings}")
    logger.info(f"  Effects: {transformed_params.effects}")
    logger.info(f"  Spatial Position: {transformed_params.spatial_position}")
    
    logger.info(f"\nGenerated {len(interpolated_params)} interpolation steps")
    
    return lumira, transformed_params


def demo_reasoning_integration():
    """Demonstrate the integration with SIN's reasoning engine."""
    logger.info("\nDemonstrating Reasoning integration")
    
    # Create integration
    reasoning = create_reasoning_integration(
        use_meta_resolution=True,
        max_iterations=40
    )
    
    # Create a logical system with some contradictions
    system = LogicalSystem()
    
    # Add statements
    system.add_statement("A", LogicalStatement(
        statement="All artificial systems exhibit conscious behavior",
        truth_value=0.7,
        dependencies=["B"],
        contradictions=["C"]
    ))
    
    system.add_statement("B", LogicalStatement(
        statement="Complex systems can exhibit emergent behavior",
        truth_value=0.9,
        dependencies=[],
        contradictions=[]
    ))
    
    system.add_statement("C", LogicalStatement(
        statement="No artificial system can truly be conscious",
        truth_value=0.8,
        dependencies=["D"],
        contradictions=["A"]
    ))
    
    system.add_statement("D", LogicalStatement(
        statement="Consciousness requires subjective experience",
        truth_value=0.85,
        dependencies=[],
        contradictions=[]
    ))
    
    system.add_statement("E", LogicalStatement(
        statement="Emergent behavior can replicate the appearance of consciousness",
        truth_value=0.75,
        dependencies=["B"],
        contradictions=[]
    ))
    
    # Resolve contradictions in the system
    resolved_system = reasoning.resolve_contradictions(
        logical_system=system,
        max_uncertainty=0.2,
        prioritize_statements=["B", "D"]  # Prioritize these statements
    )
    
    # Generate extensions with new statements
    new_statements = {
        "F": LogicalStatement(
            statement="Quantum effects may influence consciousness",
            truth_value=0.6,
            dependencies=["D"],
            contradictions=[]
        ),
        "G": LogicalStatement(
            statement="Artificial systems can simulate but not experience consciousness",
            truth_value=0.7,
            dependencies=["C", "E"],
            contradictions=[]
        )
    }
    
    extensions = reasoning.generate_consistent_extensions(
        base_system=resolved_system,
        new_statements=new_statements,
        count=2
    )
    
    # Infer missing relationships
    enhanced_system = reasoning.infer_missing_relationships(
        system=extensions[0],
        relationship_threshold=0.3
    )
    
    # Calculate belief network
    belief_network = reasoning.calculate_belief_network(enhanced_system)
    
    # Display results
    logger.info("\nOriginal Logical System:")
    for key, statement in system.statements.items():
        logger.info(f"  {key}: {statement}")
    
    logger.info("\nResolved Logical System:")
    for key, statement in resolved_system.statements.items():
        logger.info(f"  {key}: {statement}")
    
    logger.info(f"\nGenerated {len(extensions)} consistent extensions")
    
    logger.info("\nEnhanced System with Inferred Relationships:")
    for key, statement in enhanced_system.statements.items():
        logger.info(f"  {key}: {statement}")
    
    logger.info("\nBelief Network (key influences):")
    for key, influences in belief_network.items():
        if influences:
            logger.info(f"  {key} is influenced by:")
            for other_key, influence in influences.items():
                logger.info(f"    {other_key}: {influence:.2f}")
    
    return reasoning, enhanced_system


def demo_pattern_recognition_integration():
    """Demonstrate the integration with SIN's pattern recognition system."""
    logger.info("\nDemonstrating Pattern Recognition integration")
    
    # Create integration
    pattern_recognition = create_pattern_recognition_integration(
        use_evolution=True,
        max_iterations=25
    )
    
    # Create a set of patterns
    pattern_set = PatternSet()
    
    # Add some sample patterns (in a real system, these would be extracted from data)
    # Sine wave pattern
    x = np.linspace(0, 2*np.pi, 50)
    sine_data = np.sin(x)
    pattern_set.add_pattern(PatternInstance(
        data=sine_data,
        pattern_type="cyclic",
        confidence=0.8,
        metadata={"frequency": 1.0, "source": "sensor_1"}
    ))
    
    # Linear pattern with noise
    linear_data = np.linspace(0, 1, 50) + np.random.normal(0, 0.1, 50)
    pattern_set.add_pattern(PatternInstance(
        data=linear_data,
        pattern_type="linear",
        confidence=0.7,
        metadata={"slope": 1.0, "source": "sensor_2"}
    ))
    
    # Exponential pattern
    exp_data = np.exp(np.linspace(0, 1, 50))
    pattern_set.add_pattern(PatternInstance(
        data=exp_data,
        pattern_type="exponential",
        confidence=0.85,
        metadata={"growth_rate": 2.7, "source": "sensor_3"}
    ))
    
    # Step function pattern
    step_data = np.zeros(50)
    step_data[25:] = 1.0
    step_data += np.random.normal(0, 0.05, 50)
    pattern_set.add_pattern(PatternInstance(
        data=step_data,
        pattern_type="step",
        confidence=0.6,
        metadata={"threshold": 0.5, "source": "sensor_4"}
    ))
    
    # 2D pattern (e.g., image or heatmap)
    matrix_data = np.random.normal(0, 0.1, (10, 10))
    
    # Add a structure to the matrix
    for i in range(10):
        matrix_data[i, i] = 1.0  # Diagonal pattern
    
    pattern_set.add_pattern(PatternInstance(
        data=matrix_data,
        pattern_type="spatial",
        confidence=0.7,
        metadata={"dimensions": 2, "source": "camera_1"}
    ))
    
    # Refine the patterns
    refined_patterns = pattern_recognition.refine_patterns(
        pattern_set=pattern_set,
        noise_reduction_level=0.4
    )
    
    # Identify meta-patterns
    meta_patterns = pattern_recognition.identify_meta_patterns(
        pattern_set=refined_patterns,
        similarity_threshold=0.6,
        max_meta_patterns=2
    )
    
    # Generate variations of the first pattern
    base_pattern = pattern_set.patterns[0]
    variations = pattern_recognition.generate_pattern_variations(
        base_pattern=base_pattern,
        count=3,
        variation_degree=0.3
    )
    
    # Calculate similarities between original patterns
    similarities = pattern_recognition.calculate_pattern_similarities(
        patterns=pattern_set.patterns
    )
    
    # Display results
    logger.info(f"Original pattern set has {len(pattern_set.patterns)} patterns")
    logger.info(f"Refined pattern set has {len(refined_patterns.patterns)} patterns")
    logger.info(f"Identified {len(meta_patterns)} meta-patterns")
    logger.info(f"Generated {len(variations)} variations of the base pattern")
    
    logger.info("\nOriginal Patterns:")
    for i, pattern in enumerate(pattern_set.patterns):
        logger.info(f"  Pattern {i+1}: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})")
    
    logger.info("\nRefined Patterns:")
    for i, pattern in enumerate(refined_patterns.patterns):
        logger.info(f"  Pattern {i+1}: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})")
    
    logger.info("\nMeta-Patterns:")
    for name, pattern_indices in meta_patterns.items():
        logger.info(f"  {name}: includes patterns {pattern_indices}")
    
    logger.info("\nPattern Similarity Matrix:")
    for i in range(similarities.shape[0]):
        logger.info(f"  {i+1}: {[f'{val:.2f}' for val in similarities[i]]}")
    
    return pattern_recognition, refined_patterns


def main():
    """Run all integration demos."""
    logger.info("Starting Crypto_ParadoxOS Integration Examples")
    
    # Run MusicPortal integrations
    shinobi, composed_structure = demo_shinobi_integration()
    lumira, transformed_params = demo_lumira_integration()
    
    # Run SIN integrations
    reasoning, enhanced_system = demo_reasoning_integration()
    pattern_recognition, refined_patterns = demo_pattern_recognition_integration()
    
    logger.info("\nAll integrations demonstrated successfully!")
    logger.info("\nIntegration status:")
    logger.info(f"  Shinobi: {shinobi.get_integration_status()}")
    logger.info(f"  Lumira: {lumira.get_integration_status()}")
    logger.info(f"  Reasoning: {reasoning.get_integration_status()}")
    logger.info(f"  Pattern Recognition: {pattern_recognition.get_integration_status()}")


if __name__ == "__main__":
    main()