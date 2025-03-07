#!/usr/bin/env python3
"""
Crypto_ParadoxOS Evolutionary Engine

This module implements a creative evolutionary engine that generates
novel transformation rules and resolution strategies through recursive
self-modification and genetic programming techniques.
"""

import random
import math
import numpy as np
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

from meta_resolver import MetaResolver

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
        self.ancestry = []  # List of parent rule names
        self.generation = 0  # Generation this rule was created in
        
        # Initialize with default components if empty
        if not self.components:
            self.components = [
                "add", "subtract", "multiply", "divide",
                "sin", "cos", "exp", "log", "tanh",
                "normalize", "clip", "invert", "reflect"
            ]
            # Randomly select a subset of components
            random.shuffle(self.components)
            self.components = self.components[:random.randint(2, 5)]
    
    def mutate(self, mutation_rate: float = 0.2) -> 'RuleGenome':
        """
        Create a mutated copy of this genome.
        
        Args:
            mutation_rate: Probability of mutation for each component
            
        Returns:
            New mutated RuleGenome
        """
        # Copy components
        new_components = self.components.copy()
        
        # Possible operations
        all_ops = [
            "add", "subtract", "multiply", "divide", 
            "sin", "cos", "exp", "log", "tanh",
            "normalize", "clip", "invert", "reflect",
            "transpose", "negate", "absolute", "square",
            "smooth", "dampen", "amplify", "stabilize"
        ]
        
        # Mutate by adding, removing, or replacing components
        for i in range(len(new_components)):
            if random.random() < mutation_rate:
                # Replace with a random operation
                new_components[i] = random.choice(all_ops)
        
        # Occasionally add a new component
        if random.random() < mutation_rate:
            new_components.append(random.choice(all_ops))
        
        # Occasionally remove a component (if more than 1)
        if len(new_components) > 1 and random.random() < mutation_rate:
            new_components.pop(random.randrange(len(new_components)))
        
        # Create new genome with mutated components
        new_name = f"evolved_{self.rule_name.split('_')[-1]}_{random.randint(100, 999)}"
        new_complexity = self.complexity * (1.0 + (random.random() - 0.5) * 0.2)  # +/- 10%
        
        new_genome = RuleGenome(
            rule_name=new_name,
            components=new_components,
            complexity=new_complexity,
            fitness=0.0  # Reset fitness
        )
        
        # Track ancestry
        new_genome.ancestry = self.ancestry.copy()
        new_genome.ancestry.append(self.rule_name)
        new_genome.generation = self.generation + 1
        
        return new_genome
    
    def crossover(self, other: 'RuleGenome') -> 'RuleGenome':
        """
        Create a new genome by crossing this genome with another.
        
        Args:
            other: Another RuleGenome to cross with
            
        Returns:
            New RuleGenome resulting from crossover
        """
        # Simple one-point crossover
        crossover_point = random.randint(1, min(len(self.components), len(other.components)) - 1)
        
        new_components = self.components[:crossover_point] + other.components[crossover_point:]
        
        # Create new genome with crossed components
        new_name = f"cross_{self.rule_name.split('_')[-1]}_{other.rule_name.split('_')[-1]}"
        new_complexity = (self.complexity + other.complexity) / 2.0
        
        new_genome = RuleGenome(
            rule_name=new_name,
            components=new_components,
            complexity=new_complexity,
            fitness=0.0  # Reset fitness
        )
        
        # Track ancestry from both parents
        new_genome.ancestry = list(set(self.ancestry + other.ancestry))
        new_genome.ancestry.extend([self.rule_name, other.rule_name])
        new_genome.generation = max(self.generation, other.generation) + 1
        
        return new_genome
    
    def to_transformation_function(self) -> Callable[[Any], Any]:
        """
        Compile the genome into an executable transformation function.
        
        Returns:
            A function that applies the rule to a state
        """
        def transformation_function(state: Any) -> Any:
            """Apply the compiled transformation rule to a state."""
            result = state
            
            for operation in self.components:
                try:
                    if operation == "add":
                        # Add a small constant
                        if isinstance(result, (int, float)):
                            result += 0.1
                        elif isinstance(result, np.ndarray):
                            result += 0.1
                    
                    elif operation == "subtract":
                        # Subtract a small constant
                        if isinstance(result, (int, float)):
                            result -= 0.1
                        elif isinstance(result, np.ndarray):
                            result -= 0.1
                    
                    elif operation == "multiply":
                        # Multiply by a constant
                        if isinstance(result, (int, float)):
                            result *= 0.95
                        elif isinstance(result, np.ndarray):
                            result *= 0.95
                    
                    elif operation == "divide":
                        # Divide by a constant (safely)
                        if isinstance(result, (int, float)):
                            if result != 0:
                                result = result / 1.05
                        elif isinstance(result, np.ndarray):
                            # Avoid division by zero
                            result = result / (1.05 + 1e-10 * (np.abs(result) < 1e-10))
                    
                    elif operation == "sin":
                        # Apply sine function
                        if isinstance(result, (int, float)):
                            result = math.sin(result)
                        elif isinstance(result, np.ndarray):
                            result = np.sin(result)
                    
                    elif operation == "cos":
                        # Apply cosine function
                        if isinstance(result, (int, float)):
                            result = math.cos(result)
                        elif isinstance(result, np.ndarray):
                            result = np.cos(result)
                    
                    elif operation == "exp":
                        # Apply exponential function (safely)
                        if isinstance(result, (int, float)):
                            result = math.exp(min(result, 10))  # Avoid overflow
                        elif isinstance(result, np.ndarray):
                            result = np.exp(np.minimum(result, 10))
                    
                    elif operation == "log":
                        # Apply logarithm (safely)
                        if isinstance(result, (int, float)):
                            result = math.log(abs(result) + 1e-10)
                        elif isinstance(result, np.ndarray):
                            result = np.log(np.abs(result) + 1e-10)
                    
                    elif operation == "tanh":
                        # Apply hyperbolic tangent
                        if isinstance(result, (int, float)):
                            result = math.tanh(result)
                        elif isinstance(result, np.ndarray):
                            result = np.tanh(result)
                    
                    elif operation == "normalize":
                        # Normalize values
                        if isinstance(result, (int, float)):
                            result = result / (1.0 + abs(result))
                        elif isinstance(result, np.ndarray):
                            max_abs = np.max(np.abs(result))
                            if max_abs > 0:
                                result = result / max_abs
                    
                    elif operation == "clip":
                        # Clip values to [-1, 1]
                        if isinstance(result, (int, float)):
                            result = max(-1.0, min(1.0, result))
                        elif isinstance(result, np.ndarray):
                            result = np.clip(result, -1.0, 1.0)
                    
                    elif operation == "invert":
                        # Invert values
                        if isinstance(result, (int, float)):
                            if result != 0:
                                result = 1.0 / result
                        elif isinstance(result, np.ndarray):
                            # Avoid division by zero
                            mask = np.abs(result) > 1e-10
                            result[mask] = 1.0 / result[mask]
                    
                    elif operation == "reflect":
                        # Reflect around zero
                        if isinstance(result, (int, float)):
                            result = -result
                        elif isinstance(result, np.ndarray):
                            result = -result
                    
                    elif operation == "transpose":
                        # Transpose matrix
                        if isinstance(result, np.ndarray) and len(result.shape) == 2:
                            result = result.T
                    
                    elif operation == "negate":
                        # Logical negation for boolean or near-boolean values
                        if isinstance(result, bool):
                            result = not result
                        elif isinstance(result, (int, float)):
                            if 0 <= result <= 1:
                                result = 1 - result
                        elif isinstance(result, np.ndarray):
                            if np.all((0 <= result) & (result <= 1)):
                                result = 1 - result
                    
                    elif operation == "absolute":
                        # Absolute value
                        if isinstance(result, (int, float)):
                            result = abs(result)
                        elif isinstance(result, np.ndarray):
                            result = np.abs(result)
                    
                    elif operation == "square":
                        # Square values
                        if isinstance(result, (int, float)):
                            result = result * result
                        elif isinstance(result, np.ndarray):
                            result = result * result
                    
                    elif operation == "smooth":
                        # Apply smoothing
                        if isinstance(result, (int, float)):
                            result = 0.9 * result
                        elif isinstance(result, np.ndarray):
                            result = 0.9 * result
                    
                    elif operation == "dampen":
                        # Dampen oscillations
                        if isinstance(result, (int, float)):
                            result = result / (1 + 0.1 * abs(result))
                        elif isinstance(result, np.ndarray):
                            result = result / (1 + 0.1 * np.abs(result))
                    
                    elif operation == "amplify":
                        # Amplify small values
                        if isinstance(result, (int, float)):
                            result = result * (1.1 + 0.2 / (1 + abs(result)))
                        elif isinstance(result, np.ndarray):
                            result = result * (1.1 + 0.2 / (1 + np.abs(result)))
                    
                    elif operation == "stabilize":
                        # Push values toward fixed points
                        if isinstance(result, (int, float)):
                            if abs(result) < 0.5:
                                result = result * 0.9
                            else:
                                result = result + 0.1 * (1 if result > 0 else -1)
                        elif isinstance(result, np.ndarray):
                            small_mask = np.abs(result) < 0.5
                            result[small_mask] *= 0.9
                            result[~small_mask] += 0.1 * np.sign(result[~small_mask])
                
                except Exception:
                    # Skip operations that cause errors
                    pass
            
            return result
        
        return transformation_function
    
    def __repr__(self) -> str:
        """String representation of the genome."""
        return f"RuleGenome(name='{self.rule_name}', components={self.components}, fitness={self.fitness:.4f})"

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
        
        self.population = []
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Initialize the rule population with seed rules and random variations."""
        # Start with seed rules if provided
        if self.seed_rules:
            for name, func in self.seed_rules.items():
                components = self._analyze_function(func)
                self.population.append(
                    RuleGenome(rule_name=name, components=components, complexity=1.0)
                )
        
        # Fill the rest of the population with random genomes
        while len(self.population) < self.population_size:
            name = f"random_rule_{len(self.population):03d}"
            components = []
            
            # Generate random components
            for _ in range(random.randint(2, 5)):
                # Possible operations
                ops = [
                    "add", "subtract", "multiply", "divide", 
                    "sin", "cos", "exp", "log", "tanh",
                    "normalize", "clip", "invert", "reflect",
                    "transpose", "negate", "absolute", "square"
                ]
                components.append(random.choice(ops))
            
            self.population.append(
                RuleGenome(rule_name=name, components=components, complexity=1.0)
            )
    
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
        # In a real implementation, we would analyze the function's AST or source code
        # For this demo, we'll just return some plausible operations
        operations = ["normalize", "clip", "smooth"]
        
        # Add some randomness
        if random.random() < 0.5:
            operations.append("add")
        if random.random() < 0.5:
            operations.append("multiply")
        if random.random() < 0.3:
            operations.append("sin")
        if random.random() < 0.3:
            operations.append("tanh")
        
        return operations
    
    def _evaluate_fitness(self, genome: RuleGenome, test_cases: List[Any]) -> float:
        """
        Evaluate the fitness of a rule genome on test cases.
        
        Args:
            genome: The rule genome to evaluate
            test_cases: List of test inputs
            
        Returns:
            Fitness score (0.0 to 1.0)
        """
        # Convert genome to a function
        transform_fn = genome.to_transformation_function()
        
        # Track performance metrics
        convergence_score = 0.0
        stability_score = 0.0
        novelty_score = 0.0
        
        for case in test_cases:
            try:
                # Apply the function multiple times to check for convergence
                states = [case]
                for _ in range(10):
                    states.append(transform_fn(states[-1]))
                
                # Check for convergence (are later states stabilizing?)
                if len(states) >= 3:
                    # For numeric cases
                    if all(isinstance(s, (int, float)) for s in states):
                        deltas = [abs(states[i+1] - states[i]) for i in range(len(states)-1)]
                        # Are deltas decreasing?
                        if deltas[0] > 0 and all(deltas[i] <= deltas[i-1] for i in range(1, len(deltas))):
                            convergence_score += 1.0
                        # Did we reach approximate convergence?
                        if deltas[-1] < 0.01:
                            stability_score += 1.0
                    
                    # For numpy arrays
                    elif all(isinstance(s, np.ndarray) for s in states):
                        deltas = [np.max(np.abs(states[i+1] - states[i])) for i in range(len(states)-1)]
                        # Are deltas decreasing?
                        if deltas[0] > 0 and all(deltas[i] <= deltas[i-1] for i in range(1, len(deltas))):
                            convergence_score += 1.0
                        # Did we reach approximate convergence?
                        if deltas[-1] < 0.01:
                            stability_score += 1.0
                
                # Is the final state different from the input?
                if isinstance(case, (int, float)) and isinstance(states[-1], (int, float)):
                    if abs(states[-1] - case) > 0.1:
                        novelty_score += 1.0
                elif isinstance(case, np.ndarray) and isinstance(states[-1], np.ndarray):
                    if np.max(np.abs(states[-1] - case)) > 0.1:
                        novelty_score += 1.0
            
            except Exception:
                # Failed tests reduce fitness
                pass
        
        # Normalize scores
        if test_cases:
            convergence_score /= len(test_cases)
            stability_score /= len(test_cases)
            novelty_score /= len(test_cases)
        
        # Combine scores with weights
        fitness = (
            0.4 * convergence_score +
            0.4 * stability_score +
            0.2 * novelty_score
        )
        
        # Adjust fitness based on complexity (prefer simpler rules)
        complexity_penalty = max(0, (genome.complexity - 1.0) * 0.1)
        fitness = max(0, fitness - complexity_penalty)
        
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
        # Initial fitness evaluation
        for genome in self.population:
            genome.fitness = self._evaluate_fitness(genome, test_cases)
            genome.generation = 0  # Set initial generation
        
        # Track evolution history
        history = {
            'max_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'novelty': []
        }
        
        # Best fitness found so far
        best_fitness = max(genome.fitness for genome in self.population)
        avg_fitness = sum(genome.fitness for genome in self.population) / len(self.population)
        
        # Main evolution loop
        for generation in range(generations):
            # Sort by fitness (descending)
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            
            # Record statistics
            current_max_fitness = self.population[0].fitness
            current_avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
            current_diversity = self._calculate_diversity()
            
            # Novelty is the relative improvement in fitness
            novelty = 0
            if generation > 0:
                # Compare to previous generation
                prev_max = history['max_fitness'][-1]
                novelty = max(0, (current_max_fitness - prev_max) / max(0.001, prev_max))
            
            # Update history
            history['max_fitness'].append(current_max_fitness)
            history['avg_fitness'].append(current_avg_fitness)
            history['diversity'].append(current_diversity)
            history['novelty'].append(novelty)
            
            # Update best fitness
            best_fitness = max(best_fitness, current_max_fitness)
            avg_fitness = current_avg_fitness
            
            # Create next generation
            elite_count = int(self.population_size * self.elitism_ratio)
            new_population = self.population[:elite_count]  # Elitism
            
            # Breed the rest of the population
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self._select_parent()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    parent2 = self._select_parent(exclude=parent1)
                    child = parent1.crossover(parent2)
                else:
                    child = RuleGenome(
                        rule_name=f"clone_{parent1.rule_name}",
                        components=parent1.components.copy(),
                        complexity=parent1.complexity
                    )
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = child.mutate()
                
                # Add to new population
                new_population.append(child)
            
            # Replace old population
            self.population = new_population
            
            # Evaluate fitness of new genomes
            for genome in self.population:
                if genome.fitness == 0.0:  # Only evaluate new genomes
                    genome.fitness = self._evaluate_fitness(genome, test_cases)
        
        # Sort final population by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Return results
        best_fitness = self.population[0].fitness if self.population else 0.0
        avg_fitness = sum(g.fitness for g in self.population) / len(self.population) if self.population else 0.0
        diversity = self._calculate_diversity()
        
        return {
            "best_rules": [(g.rule_name, g.to_transformation_function()) for g in self.population[:5]],
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "diversity": diversity,
            "generations": generations,
            "population_size": self.population_size,
            "history": history
        }
    
    def _select_parent(self, exclude: RuleGenome = None) -> RuleGenome:
        """
        Select a parent using tournament selection.
        
        Args:
            exclude: Optional genome to exclude from selection
            
        Returns:
            Selected RuleGenome
        """
        # Tournament selection
        tournament_size = 3
        candidates = [genome for genome in self.population if genome != exclude]
        
        if not candidates:
            return self.population[0]  # Fallback
        
        tournament = random.sample(candidates, min(tournament_size, len(candidates)))
        return max(tournament, key=lambda g: g.fitness)
    
    def _calculate_diversity(self) -> float:
        """
        Calculate the diversity of the current population.
        
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if not self.population or len(self.population) < 2:
            return 0.0
        
        # Calculate average difference in components between genomes
        total_diff = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                # Count differences in components
                g1 = set(self.population[i].components)
                g2 = set(self.population[j].components)
                
                # Jaccard distance
                if g1 or g2:  # Avoid division by zero
                    diff = 1.0 - len(g1 & g2) / len(g1 | g2)
                    total_diff += diff
                    comparisons += 1
        
        return total_diff / comparisons if comparisons > 0 else 0.0
    
    def get_best_rules(self, count: int = 5) -> List[Tuple[str, Callable]]:
        """
        Get the best performing rules from the population.
        
        Args:
            count: Number of rules to return
            
        Returns:
            List of (rule_name, transform_function) tuples
        """
        # Sort by fitness (descending)
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Return the top rules
        return [(g.rule_name, g.to_transformation_function()) 
                for g in self.population[:min(count, len(self.population))]]
    
    def create_meta_resolver(self) -> MetaResolver:
        """
        Create a meta-resolver with evolved rules and optimized phases.
        
        Returns:
            Configured MetaResolver
        """
        # Create a new meta-resolver
        meta = MetaResolver()
        
        # Create an evolutionary framework with best rules
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
        from meta_resolver import ResolutionPhase
        
        # Create custom transition functions
        def to_divergent(state: Any) -> bool:
            """Transition to divergent phase when near convergence."""
            if isinstance(state, (int, float)):
                return abs(state) < 0.2
            return False
        
        def to_exploration(state: Any) -> bool:
            """Always transition to exploration after divergence."""
            return True
        
        def to_convergent(state: Any) -> bool:
            """Transition to convergent phase after exploration."""
            if isinstance(state, (int, float)):
                return abs(state) > 0.5
            return True
        
        # Get best rules
        best_rules = [g.rule_name for g, _ in zip(self.population, range(3))]
        
        # Create phases
        convergent = ResolutionPhase("evo_convergent", is_convergent=True)
        for rule in best_rules[:2]:  # Use top 2 for convergence
            convergent.add_rule(rule)
        convergent.add_transition("evo_divergent", to_divergent)
        
        divergent = ResolutionPhase("evo_divergent", is_convergent=False)
        for rule in best_rules[2:4]:  # Use next 2 for divergence
            divergent.add_rule(rule)
        divergent.add_transition("evo_exploration", to_exploration)
        
        exploration = ResolutionPhase("evo_exploration", is_convergent=False)
        for rule in best_rules[3:]:  # Use remaining rules for exploration
            exploration.add_rule(rule)
        exploration.add_transition("evo_convergent", to_convergent)
        
        # Add phases to meta-resolver
        meta.add_phase(convergent)
        meta.add_phase(divergent)
        meta.add_phase(exploration)
        
        # Set initial phase
        meta.set_initial_phase("evo_convergent")