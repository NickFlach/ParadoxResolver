#!/usr/bin/env python3
"""
Crypto_ParadoxOS Evolutionary Engine

This module implements a creative evolutionary engine that generates
novel transformation rules and resolution strategies through recursive
self-modification and genetic programming techniques.
"""

import numpy as np
import time
import random
import copy
from typing import Any, Dict, List, Tuple, Callable, Optional, Union, Set

from crypto_paradox_os import ParadoxState, TransformationRule
from meta_resolver import MetaResolver, ResolutionPhase

class RuleGenome:
    """
    Represents the genetic encoding of a transformation rule.
    
    This class enables rules to evolve and mutate over time.
    """
    
    def __init__(self, 
                rule_name: str,
                components: List[str] = None,
                complexity: float = 1.0,
                fitness: float = 0.0):
        """
        Initialize a rule genome.
        
        Args:
            rule_name: Name of the transformation rule
            components: List of primitive operations that make up the rule
            complexity: Measure of the rule's complexity
            fitness: Current fitness score for the rule
        """
        self.rule_name = rule_name
        self.components = components or []
        self.complexity = complexity
        self.fitness = fitness
        self.generation = 0
        self.ancestry = []
        
    def mutate(self, mutation_rate: float = 0.2) -> 'RuleGenome':
        """
        Create a mutated copy of this genome.
        
        Args:
            mutation_rate: Probability of mutation for each component
            
        Returns:
            New mutated RuleGenome
        """
        # Create a copy of the current genome
        new_genome = copy.deepcopy(self)
        new_genome.generation = self.generation + 1
        new_genome.ancestry = self.ancestry + [self.rule_name]
        
        # Possible operations for mutation
        operations = [
            "add_component",
            "remove_component",
            "modify_component",
            "reorder_components"
        ]
        
        # Apply random mutations based on mutation rate
        if random.random() < mutation_rate:
            operation = random.choice(operations)
            
            if operation == "add_component" and len(new_genome.components) < 10:
                new_component = random.choice(PRIMITIVE_OPERATIONS)
                position = random.randint(0, len(new_genome.components))
                new_genome.components.insert(position, new_component)
                new_genome.complexity += 0.1
                
            elif operation == "remove_component" and len(new_genome.components) > 1:
                position = random.randint(0, len(new_genome.components) - 1)
                new_genome.components.pop(position)
                new_genome.complexity = max(0.1, new_genome.complexity - 0.1)
                
            elif operation == "modify_component" and new_genome.components:
                position = random.randint(0, len(new_genome.components) - 1)
                new_genome.components[position] = random.choice(PRIMITIVE_OPERATIONS)
                
            elif operation == "reorder_components" and len(new_genome.components) > 1:
                random.shuffle(new_genome.components)
        
        # Generate a new name based on ancestry and mutation
        prefix = self.rule_name.split("_v")[0] if "_v" in self.rule_name else self.rule_name
        new_genome.rule_name = f"{prefix}_v{new_genome.generation}"
        
        # Reset fitness as it needs to be re-evaluated
        new_genome.fitness = 0.0
        
        return new_genome
    
    def crossover(self, other: 'RuleGenome') -> 'RuleGenome':
        """
        Create a new genome by crossing this genome with another.
        
        Args:
            other: Another RuleGenome to cross with
            
        Returns:
            New RuleGenome resulting from crossover
        """
        # Create a new genome
        child = RuleGenome(
            rule_name=f"Hybrid_{self.rule_name}_{other.rule_name}",
            complexity=(self.complexity + other.complexity) / 2
        )
        
        # Combine components from both parents
        child.generation = max(self.generation, other.generation) + 1
        child.ancestry = list(set(self.ancestry + other.ancestry + [self.rule_name, other.rule_name]))
        
        # Perform crossover of components
        if not self.components or not other.components:
            child.components = self.components or other.components
        else:
            # Single-point crossover
            crossover_point = random.randint(1, min(len(self.components), len(other.components)) - 1)
            child.components = self.components[:crossover_point] + other.components[crossover_point:]
        
        # Generate a new name based on ancestry
        parent_prefixes = []
        for parent in [self.rule_name, other.rule_name]:
            prefix = parent.split("_v")[0] if "_v" in parent else parent
            parent_prefixes.append(prefix)
            
        child.rule_name = f"{'_'.join(parent_prefixes)}_v{child.generation}"
        
        return child
    
    def to_transformation_function(self) -> Callable[[Any], Any]:
        """
        Compile the genome into an executable transformation function.
        
        Returns:
            A function that applies the rule to a state
        """
        if not self.components:
            # Default identity transformation if no components
            return lambda x: x
        
        def transformation_function(state: Any) -> Any:
            """Apply the compiled transformation rule to a state."""
            current_state = state
            
            for component in self.components:
                # Apply each component in sequence
                op_func = OPERATION_FUNCTIONS.get(component)
                if op_func:
                    try:
                        current_state = op_func(current_state)
                    except Exception:
                        # If any operation fails, skip it and continue
                        pass
            
            return current_state
        
        return transformation_function
    
    def __repr__(self) -> str:
        """String representation of the genome."""
        return f"RuleGenome({self.rule_name}, components={len(self.components)}, fitness={self.fitness:.4f})"


class EvolutionaryEngine:
    """
    Engine for evolving novel transformation rules and resolution strategies.
    
    This class implements genetic algorithms to generate creative solutions
    that weren't explicitly programmed.
    """
    
    def __init__(self, 
                seed_rules: Dict[str, Callable] = None,
                population_size: int = 20,
                mutation_rate: float = 0.3,
                crossover_rate: float = 0.7,
                elitism_ratio: float = 0.2):
        """
        Initialize the evolutionary engine.
        
        Args:
            seed_rules: Dictionary of seed transformation functions
            population_size: Size of the rule population to maintain
            mutation_rate: Probability of mutation during reproduction
            crossover_rate: Probability of crossover during reproduction
            elitism_ratio: Proportion of top performers to preserve
        """
        self.seed_rules = seed_rules or {}
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        # Rule populations
        self.rule_population: List[RuleGenome] = []
        self.elite_rules: List[RuleGenome] = []
        self.generations = 0
        
        # Phase populations
        self.phase_templates: List[ResolutionPhase] = []
        
        # Evolution metrics
        self.evolution_history = {
            "avg_fitness": [],
            "max_fitness": [],
            "diversity": [],
            "novelty": []
        }
        
        # Initialize with seed rules
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Initialize the rule population with seed rules and random variations."""
        # Convert seed rules to genomes
        for rule_name, rule_func in self.seed_rules.items():
            # Analyze the function to extract potential components
            components = self._analyze_function(rule_func)
            
            # Create a genome for this rule
            genome = RuleGenome(
                rule_name=rule_name,
                components=components,
                complexity=len(components) * 0.2,
                fitness=0.5  # Initial conservative fitness estimate
            )
            
            self.rule_population.append(genome)
        
        # Create additional rules to fill the population
        while len(self.rule_population) < self.population_size:
            # Either mutate an existing rule or create a random one
            if self.rule_population and random.random() < 0.7:
                # Mutate an existing rule
                parent = random.choice(self.rule_population)
                new_genome = parent.mutate(mutation_rate=0.5)  # Higher mutation for initialization
            else:
                # Create a random rule
                components = [random.choice(PRIMITIVE_OPERATIONS) for _ in range(random.randint(1, 5))]
                new_genome = RuleGenome(
                    rule_name=f"Random_Rule_{len(self.rule_population)}",
                    components=components,
                    complexity=len(components) * 0.2
                )
            
            self.rule_population.append(new_genome)
    
    def _analyze_function(self, func: Callable) -> List[str]:
        """
        Analyze a function to extract potential components.
        
        This is a simplified approach - a real implementation would
        use code analysis to extract operations.
        
        Args:
            func: The function to analyze
            
        Returns:
            List of operation components
        """
        # In a real implementation, we would parse the function's code
        # For now, just return some random components
        return random.sample(PRIMITIVE_OPERATIONS, min(5, len(PRIMITIVE_OPERATIONS)))
    
    def _evaluate_fitness(self, genome: RuleGenome, test_cases: List[Any]) -> float:
        """
        Evaluate the fitness of a rule genome on test cases.
        
        Args:
            genome: The rule genome to evaluate
            test_cases: List of test inputs
            
        Returns:
            Fitness score (0.0 to 1.0)
        """
        if not test_cases:
            return 0.0
        
        # Compile the genome into a function
        transform_func = genome.to_transformation_function()
        
        # Metrics for fitness
        effectiveness = 0.0  # Does it produce meaningful change?
        stability = 0.0  # Does it maintain stability where appropriate?
        novelty = 0.0  # Does it produce unique results?
        
        previous_results = []
        
        for test_case in test_cases:
            try:
                # Apply the transformation
                result = transform_func(test_case)
                
                # Check if it produced a change
                if isinstance(test_case, (int, float)) and isinstance(result, (int, float)):
                    # For numerical values, measure relative change
                    if abs(test_case) > 1e-10:
                        change_magnitude = abs((result - test_case) / test_case)
                        # We want some change, but not too extreme
                        effectiveness += min(change_magnitude, 0.5) / 0.5
                    else:
                        # For near-zero values, look at absolute change
                        effectiveness += min(abs(result - test_case), 1.0)
                
                # Check for stability in complex types
                if isinstance(test_case, np.ndarray) and isinstance(result, np.ndarray):
                    if test_case.shape == result.shape:
                        # For matrices, measure norm of difference
                        norm_diff = np.linalg.norm(result - test_case)
                        stability += 1.0 / (1.0 + norm_diff)  # Higher stability for smaller changes
                
                # Check for novelty compared to previous results
                for prev_result in previous_results:
                    if isinstance(result, type(prev_result)):
                        try:
                            # Simple equality check for novelty
                            if result != prev_result:
                                novelty += 0.1
                        except:
                            # If comparison fails, assume they're different
                            novelty += 0.1
                
                previous_results.append(result)
                
            except Exception:
                # Penalize rules that cause errors
                effectiveness -= 0.2
                stability -= 0.2
        
        # Normalize scores
        effectiveness = max(0.0, min(1.0, effectiveness / len(test_cases)))
        stability = max(0.0, min(1.0, stability / len(test_cases)))
        novelty = max(0.0, min(1.0, novelty / max(1, len(test_cases) - 1)))
        
        # Combine scores (weighted sum)
        fitness = (0.4 * effectiveness) + (0.3 * stability) + (0.3 * novelty)
        
        # Update the genome's fitness
        genome.fitness = fitness
        
        return fitness
    
    def evolve(self, 
              test_cases: List[Any],
              generations: int = 10) -> Dict[str, Any]:
        """
        Evolve the rule population for a number of generations.
        
        Args:
            test_cases: List of inputs to evaluate fitness on
            generations: Number of generations to evolve
            
        Returns:
            Dictionary with evolution results
        """
        start_time = time.time()
        
        for generation in range(generations):
            self.generations += 1
            
            # Evaluate fitness of all rules
            for genome in self.rule_population:
                self._evaluate_fitness(genome, test_cases)
            
            # Sort by fitness
            self.rule_population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Record evolution metrics
            avg_fitness = sum(g.fitness for g in self.rule_population) / len(self.rule_population)
            max_fitness = self.rule_population[0].fitness if self.rule_population else 0.0
            
            self.evolution_history["avg_fitness"].append(avg_fitness)
            self.evolution_history["max_fitness"].append(max_fitness)
            
            # Calculate diversity (average pairwise difference in components)
            diversity = self._calculate_diversity()
            self.evolution_history["diversity"].append(diversity)
            
            # Calculate novelty (difference from initial population)
            novelty = self._calculate_novelty()
            self.evolution_history["novelty"].append(novelty)
            
            # Create the next generation
            next_generation = []
            
            # Elitism: Keep the best performers
            elite_count = max(1, int(self.population_size * self.elitism_ratio))
            elite_rules = self.rule_population[:elite_count]
            next_generation.extend(elite_rules)
            
            # Update elite rules
            self.elite_rules = elite_rules.copy()
            
            # Fill the rest of the population
            while len(next_generation) < self.population_size:
                # Selection
                parent1 = self._select_parent()
                
                # Crossover
                if random.random() < self.crossover_rate and len(self.rule_population) > 1:
                    parent2 = self._select_parent(exclude=parent1)
                    child = parent1.crossover(parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = child.mutate()
                
                next_generation.append(child)
            
            # Replace population
            self.rule_population = next_generation
        
        end_time = time.time()
        
        # Return results
        return {
            "generations": self.generations,
            "evolved_rules": self.rule_population,
            "elite_rules": self.elite_rules,
            "execution_time": end_time - start_time,
            "final_avg_fitness": self.evolution_history["avg_fitness"][-1],
            "final_max_fitness": self.evolution_history["max_fitness"][-1],
            "history": self.evolution_history
        }
    
    def _select_parent(self, exclude: RuleGenome = None) -> RuleGenome:
        """
        Select a parent using tournament selection.
        
        Args:
            exclude: Optional genome to exclude from selection
            
        Returns:
            Selected RuleGenome
        """
        if not self.rule_population:
            raise ValueError("Cannot select parent from empty population")
        
        candidates = [g for g in self.rule_population if g != exclude]
        
        if not candidates:
            # If all candidates excluded, just return a random one
            return random.choice(self.rule_population)
        
        # Tournament selection (select best of random subset)
        tournament_size = min(3, len(candidates))
        tournament = random.sample(candidates, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _calculate_diversity(self) -> float:
        """
        Calculate the diversity of the current population.
        
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if len(self.rule_population) <= 1:
            return 0.0
        
        # Calculate average pairwise difference in components
        total_diff = 0
        comparisons = 0
        
        for i, genome1 in enumerate(self.rule_population):
            for genome2 in self.rule_population[i+1:]:
                # Jaccard distance between component sets
                set1 = set(genome1.components)
                set2 = set(genome2.components)
                
                if not set1 and not set2:
                    continue
                    
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                if union > 0:
                    similarity = intersection / union
                    difference = 1.0 - similarity
                    total_diff += difference
                    comparisons += 1
        
        if comparisons == 0:
            return 0.0
            
        return total_diff / comparisons
    
    def _calculate_novelty(self) -> float:
        """
        Calculate the novelty of the current population compared to initial seed rules.
        
        Returns:
            Novelty score (0.0 to 1.0)
        """
        if not self.rule_population or not self.seed_rules:
            return 0.0
        
        # Extract the original rule names (without version suffixes)
        original_names = set()
        for name in self.seed_rules.keys():
            base_name = name.split("_v")[0]
            original_names.add(base_name)
        
        # Count rules that aren't derived from original seeds
        novel_count = 0
        for genome in self.rule_population:
            base_name = genome.rule_name.split("_v")[0]
            if base_name not in original_names:
                novel_count += 1
        
        return novel_count / len(self.rule_population)
    
    def get_best_rules(self, count: int = 5) -> List[Tuple[str, Callable]]:
        """
        Get the best performing rules from the population.
        
        Args:
            count: Number of rules to return
            
        Returns:
            List of (rule_name, transform_function) tuples
        """
        # Sort by fitness and return the top rules
        sorted_rules = sorted(self.rule_population, key=lambda x: x.fitness, reverse=True)
        top_rules = sorted_rules[:count]
        
        return [(genome.rule_name, genome.to_transformation_function()) for genome in top_rules]
    
    def create_meta_resolver(self) -> MetaResolver:
        """
        Create a meta-resolver with evolved rules and optimized phases.
        
        Returns:
            Configured MetaResolver
        """
        # Get the best rules
        best_rules = self.get_best_rules(count=8)
        
        # Create a new API with the evolved rules
        from crypto_paradox_api import CryptoParadoxAPI
        api = CryptoParadoxAPI()
        
        # Register evolved rules
        for rule_name, rule_func in best_rules:
            api.register_rule(rule_name, rule_func, f"Evolved rule: {rule_name}")
        
        # Create a meta-resolver
        meta = MetaResolver(api)
        
        # Create an evolutionary framework
        self._create_evolutionary_framework(meta)
        
        return meta
    
    def _create_evolutionary_framework(self, meta: MetaResolver) -> None:
        """
        Create an evolutionary framework in the meta-resolver.
        
        This framework incorporates both convergent and divergent phases,
        with transitions based on the state's characteristics.
        
        Args:
            meta: MetaResolver to configure
        """
        # Get best rules for different purposes
        all_rules = sorted(self.rule_population, key=lambda x: x.fitness, reverse=True)
        
        # Initialize phases
        seed_phase = ResolutionPhase(
            "Initial Seeding", 
            is_convergent=True,
            max_iterations=5,
            threshold=0.01
        )
        
        divergent_phase = ResolutionPhase(
            "Creative Divergence",
            is_convergent=False,
            max_iterations=8,
            threshold=0.1
        )
        
        exploration_phase = ResolutionPhase(
            "Possibility Exploration",
            is_convergent=False,
            max_iterations=10,
            threshold=0.05
        )
        
        convergent_phase = ResolutionPhase(
            "Convergent Integration",
            is_convergent=True,
            max_iterations=12,
            threshold=0.001
        )
        
        refinement_phase = ResolutionPhase(
            "Final Refinement",
            is_convergent=True,
            max_iterations=15,
            threshold=0.0001
        )
        
        # Add top rules to each phase
        for i, rule in enumerate(all_rules):
            if i % 5 == 0:
                seed_phase.add_rule(rule.rule_name)
            elif i % 5 == 1:
                divergent_phase.add_rule(rule.rule_name)
            elif i % 5 == 2:
                exploration_phase.add_rule(rule.rule_name)
            elif i % 5 == 3:
                convergent_phase.add_rule(rule.rule_name)
            else:
                refinement_phase.add_rule(rule.rule_name)
        
        # Add standard rules if needed
        if len(seed_phase.rules) < 2:
            seed_phase.add_rule("Fixed-Point Iteration")
        
        if len(divergent_phase.rules) < 2:
            divergent_phase.add_rule("Duality Inversion")
            
        if len(exploration_phase.rules) < 2:
            exploration_phase.add_rule("Bayesian Update")
            
        if len(convergent_phase.rules) < 2:
            convergent_phase.add_rule("Eigenvalue Stabilization")
            
        if len(refinement_phase.rules) < 2:
            refinement_phase.add_rule("Recursive Normalization")
        
        # Define transition conditions
        def to_divergent(state: Any) -> bool:
            """Transition to divergent phase when near convergence."""
            if isinstance(state, (int, float)):
                return abs(state - 1.0) < 0.1
            return False
        
        def to_exploration(state: Any) -> bool:
            """Always transition to exploration after divergence."""
            return True
        
        def to_convergent(state: Any) -> bool:
            """Transition to convergent phase after exploration."""
            return True
        
        def to_refinement(state: Any) -> bool:
            """Transition to refinement for final touches."""
            return True
        
        # Configure transitions
        seed_phase.add_transition("Creative Divergence", to_divergent)
        divergent_phase.add_transition("Possibility Exploration", to_exploration)
        exploration_phase.add_transition("Convergent Integration", to_convergent)
        convergent_phase.add_transition("Final Refinement", to_refinement)
        
        # Add phases to meta-resolver
        meta.add_phase(seed_phase)
        meta.add_phase(divergent_phase)
        meta.add_phase(exploration_phase)
        meta.add_phase(convergent_phase)
        meta.add_phase(refinement_phase)
        
        # Set initial phase
        meta.set_initial_phase("Initial Seeding")


# Primitive operations for building evolutionary rules
PRIMITIVE_OPERATIONS = [
    "identity",
    "inverse",
    "square",
    "sqrt",
    "normalize",
    "dampening",
    "oscillate",
    "truncate",
    "inflate",
    "reflect",
    "shift",
    "scale",
    "merge",
    "split",
    "filter",
    "smooth",
    "randomize",
    "reorganize",
    "compose",
    "extract"
]

# Functions corresponding to primitive operations
OPERATION_FUNCTIONS = {
    "identity": lambda x: x,
    
    "inverse": lambda x: (
        1.0 / x if isinstance(x, (int, float)) and x != 0 else
        1.0 / (x + 1e-10) if isinstance(x, (int, float)) else
        1.0 / (np.array(x) + 1e-10) if isinstance(x, np.ndarray) else
        x
    ),
    
    "square": lambda x: (
        x ** 2 if isinstance(x, (int, float)) else
        np.power(x, 2) if isinstance(x, np.ndarray) else
        x
    ),
    
    "sqrt": lambda x: (
        np.sqrt(x) if isinstance(x, (int, float)) and x >= 0 else
        np.sqrt(np.abs(x)) if isinstance(x, (int, float)) else
        np.sqrt(np.abs(x)) if isinstance(x, np.ndarray) else
        x
    ),
    
    "normalize": lambda x: (
        x / abs(x) if isinstance(x, (int, float)) and x != 0 else
        x / np.max(np.abs(x)) if isinstance(x, np.ndarray) and np.max(np.abs(x)) > 0 else
        x
    ),
    
    "dampening": lambda x: (
        x * 0.9 if isinstance(x, (int, float)) else
        x * 0.9 if isinstance(x, np.ndarray) else
        x
    ),
    
    "oscillate": lambda x: (
        np.sin(x) if isinstance(x, (int, float)) else
        np.sin(x) if isinstance(x, np.ndarray) else
        x
    ),
    
    "truncate": lambda x: (
        int(x) if isinstance(x, float) else
        np.trunc(x) if isinstance(x, np.ndarray) else
        x
    ),
    
    "inflate": lambda x: (
        x * 1.1 if isinstance(x, (int, float)) else
        x * 1.1 if isinstance(x, np.ndarray) else
        x
    ),
    
    "reflect": lambda x: (
        -x if isinstance(x, (int, float)) else
        -x if isinstance(x, np.ndarray) else
        x
    ),
    
    "shift": lambda x: (
        x + 0.1 if isinstance(x, (int, float)) else
        x + 0.1 if isinstance(x, np.ndarray) else
        x
    ),
    
    "scale": lambda x: (
        x * 2.0 if isinstance(x, (int, float)) else
        x * 2.0 if isinstance(x, np.ndarray) else
        x
    ),
    
    "merge": lambda x: (
        np.mean(x) if isinstance(x, list) and all(isinstance(i, (int, float)) for i in x) else
        np.mean(x, axis=0) if isinstance(x, np.ndarray) and len(x.shape) > 1 else
        x
    ),
    
    "split": lambda x: (
        [x/2, x/2] if isinstance(x, (int, float)) else
        np.array([x/2, x/2]) if isinstance(x, np.ndarray) and x.size == 1 else
        x
    ),
    
    "filter": lambda x: (
        x if isinstance(x, (int, float)) and abs(x) > 0.1 else
        0 if isinstance(x, (int, float)) else
        x * (np.abs(x) > 0.1) if isinstance(x, np.ndarray) else
        x
    ),
    
    "smooth": lambda x: (
        x if isinstance(x, (int, float)) else
        np.convolve(x, np.ones(3)/3, mode='same') if isinstance(x, np.ndarray) and len(x.shape) == 1 and x.shape[0] >= 3 else
        x
    ),
    
    "randomize": lambda x: (
        x * (0.9 + 0.2 * random.random()) if isinstance(x, (int, float)) else
        x * (0.9 + 0.2 * np.random.random(x.shape)) if isinstance(x, np.ndarray) else
        x
    ),
    
    "reorganize": lambda x: (
        x if not isinstance(x, np.ndarray) or len(x.shape) < 2 else
        x.T if len(x.shape) == 2 else
        x
    ),
    
    "compose": lambda x: (
        np.sin(np.sqrt(np.abs(x))) if isinstance(x, (int, float)) else
        np.sin(np.sqrt(np.abs(x))) if isinstance(x, np.ndarray) else
        x
    ),
    
    "extract": lambda x: (
        x if isinstance(x, (int, float)) else
        x.flatten()[0] if isinstance(x, np.ndarray) and x.size > 0 else
        x
    )
}


def demo_evolutionary_engine():
    """Demonstrate the evolutionary engine."""
    print("Crypto_ParadoxOS Evolutionary Engine Demonstration")
    print("=================================================")
    
    # Create test cases for evolution
    test_cases = [
        0.5,  # Numerical value
        -1.0,  # Negative value
        np.array([[1.0, 0.5], [0.3, 1.2]]),  # 2x2 matrix
        [0.1, 0.2, 0.3, 0.4],  # List of numbers
        {
            'value': 0.5,
            'uncertainty': 0.1
        }  # Dictionary representation
    ]
    
    # Create engine with seed rules from transformation_rules
    from transformation_rules import get_available_rules
    seed_rules = get_available_rules()
    
    engine = EvolutionaryEngine(
        seed_rules=seed_rules,
        population_size=20,
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    print(f"\nInitialized evolutionary engine with {len(engine.rule_population)} rules")
    print(f"Starting evolution for 10 generations...")
    
    # Run evolution
    results = engine.evolve(test_cases, generations=10)
    
    print("\nEvolution completed:")
    print(f"- Generations: {results['generations']}")
    print(f"- Final average fitness: {results['final_avg_fitness']:.4f}")
    print(f"- Final maximum fitness: {results['final_max_fitness']:.4f}")
    print(f"- Execution time: {results['execution_time']:.2f} seconds")
    
    # Show the best rules
    print("\nTop evolved rules:")
    best_rules = sorted(results['evolved_rules'], key=lambda x: x.fitness, reverse=True)[:5]
    
    for i, rule in enumerate(best_rules):
        print(f"{i+1}. {rule.rule_name} (Fitness: {rule.fitness:.4f}, Components: {len(rule.components)})")
        print(f"   Components: {', '.join(rule.components[:3])}{'...' if len(rule.components) > 3 else ''}")
        
        if rule.ancestry:
            print(f"   Ancestry: {' → '.join(rule.ancestry[-2:])}")
    
    print("\nCreating Meta-Resolver with evolved rules...")
    meta_resolver = engine.create_meta_resolver()
    
    print("\nMeta-Resolver phase structure:")
    for phase_name in meta_resolver.phases:
        phase = meta_resolver.phases[phase_name]
        print(f"- {phase.name} ({'Convergent' if phase.is_convergent else 'Divergent'})")
        print(f"  Rules: {', '.join(phase.rules[:3])}{'...' if len(phase.rules) > 3 else ''}")
        print(f"  Transitions: {', '.join(phase.transition_conditions.keys())}")
    
    print("\nEvolutionary Engine Demonstration Complete")


if __name__ == "__main__":
    demo_evolutionary_engine()