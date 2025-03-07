#!/usr/bin/env python3
"""
Crypto_ParadoxOS Command Line Interface (CLI)

This module provides a command-line interface to the Crypto_ParadoxOS system,
enabling direct access to core functionality from the terminal.
"""

import argparse
import json
import sys
import time
import os
import numpy as np
from typing import Any, Dict, List, Tuple

from paradox_resolver import ParadoxResolver
from transformation_rules import get_available_rules
from meta_resolver import MetaResolver
from evolutionary_engine import EvolutionaryEngine
from utils import format_paradox_input, validate_input

class ParadoxCLI:
    """Command-line interface for the Crypto_ParadoxOS system."""
    
    def __init__(self):
        """Initialize the CLI with parsers and commands."""
        self.parser = argparse.ArgumentParser(
            description="Crypto_ParadoxOS: A Recursive Paradox-Resolution System",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self._setup_parsers()
    
    def _setup_parsers(self):
        """Set up command-line argument parsers."""
        subparsers = self.parser.add_subparsers(dest="command", help="Command to execute")
        
        # Resolve command
        resolve_parser = subparsers.add_parser("resolve", help="Resolve a paradox")
        resolve_parser.add_argument("--input", required=True, help="Paradox input value or expression")
        resolve_parser.add_argument("--type", choices=["numerical", "matrix", "text"], 
                                  default="numerical", help="Type of the paradox input")
        resolve_parser.add_argument("--initial-value", type=float, default=0.5,
                                  help="Initial value for numerical paradoxes")
        resolve_parser.add_argument("--iterations", type=int, default=20,
                                  help="Maximum iterations to perform")
        resolve_parser.add_argument("--threshold", type=float, default=0.001,
                                  help="Convergence threshold")
        resolve_parser.add_argument("--rules", nargs="+", 
                                  help="Transformation rules to apply (default: all)")
        resolve_parser.add_argument("--output", help="Output file for results (JSON)")
        
        # Evolve command
        evolve_parser = subparsers.add_parser("evolve", help="Run the evolutionary engine")
        evolve_parser.add_argument("--generations", type=int, default=10,
                                 help="Number of generations to evolve")
        evolve_parser.add_argument("--population", type=int, default=20,
                                 help="Population size")
        evolve_parser.add_argument("--mutation-rate", type=float, default=0.3,
                                 help="Mutation rate (0.0 to 1.0)")
        evolve_parser.add_argument("--crossover-rate", type=float, default=0.7,
                                 help="Crossover rate (0.0 to 1.0)")
        evolve_parser.add_argument("--test-cases", help="JSON file with test cases")
        evolve_parser.add_argument("--output", help="Output file for evolved rules (JSON)")
        
        # Meta-resolve command
        meta_parser = subparsers.add_parser("meta-resolve", 
                                         help="Use the meta-resolver framework")
        meta_parser.add_argument("--input", required=True, help="Paradox input value or expression")
        meta_parser.add_argument("--type", choices=["numerical", "matrix", "text"], 
                               default="numerical", help="Type of the paradox input")
        meta_parser.add_argument("--framework", choices=["standard", "convergence", "expansion", "custom"],
                               default="standard", help="Meta-resolution framework to use")
        meta_parser.add_argument("--config", help="JSON file with custom framework configuration")
        meta_parser.add_argument("--max-transitions", type=int, default=10,
                               help="Maximum phase transitions")
        meta_parser.add_argument("--output", help="Output file for results (JSON)")
        
        # Rules command
        rules_parser = subparsers.add_parser("rules", help="List available transformation rules")
        rules_parser.add_argument("--type", choices=["standard", "evolved", "all"],
                                default="all", help="Type of rules to list")
        rules_parser.add_argument("--verbose", action="store_true", 
                                help="Show detailed rule descriptions")
    
    def parse_input(self, input_str: str, input_type: str) -> Any:
        """
        Parse input value based on its type.
        
        Args:
            input_str: String representation of input
            input_type: Type of input ("numerical", "matrix", "text")
            
        Returns:
            Parsed input value
        """
        if input_type == "numerical":
            return input_str
        elif input_type == "matrix":
            try:
                # Try to parse as a JSON array
                matrix = json.loads(input_str)
                return np.array(matrix)
            except json.JSONDecodeError:
                # If not valid JSON, try to evaluate as Python expression
                try:
                    matrix = eval(input_str)
                    return np.array(matrix)
                except Exception:
                    raise ValueError(f"Could not parse matrix input: {input_str}")
        else:  # text
            return input_str
    
    def command_resolve(self, args):
        """Execute the resolve command."""
        # Parse and validate input
        paradox_input = self.parse_input(args.input, args.type)
        formatted_input = format_paradox_input(
            paradox_input, args.type, 
            args.initial_value if args.type == "numerical" else None
        )
        is_valid, validation_msg = validate_input(formatted_input, args.type)
        
        if not is_valid:
            print(f"Error: {validation_msg}")
            return 1
        
        # Set up transformation rules
        available_rules = get_available_rules()
        if args.rules:
            selected_rules = {name: available_rules[name] for name in args.rules if name in available_rules}
            if not selected_rules:
                print("Error: No valid transformation rules specified")
                return 1
        else:
            selected_rules = available_rules
        
        # Initialize resolver and run resolution
        print(f"Beginning resolution with {len(selected_rules)} rules")
        print(f"Initial state: {formatted_input}")
        
        resolver = ParadoxResolver(
            transformation_rules=selected_rules,
            max_iterations=args.iterations,
            convergence_threshold=args.threshold
        )
        
        start_time = time.time()
        result, steps, converged = resolver.resolve(formatted_input)
        end_time = time.time()
        
        # Display results
        processing_time = end_time - start_time
        if converged:
            print(f"Paradox resolved in {len(steps)-1} iterations!")
        else:
            print(f"Maximum iterations ({args.iterations}) reached without convergence.")
        
        print(f"Final state: {result}")
        print(f"Processing time: {processing_time:.4f} seconds")
        
        # Save results if output file specified
        if args.output:
            results_dict = {
                "input": args.input,
                "input_type": args.type,
                "final_state": result.tolist() if isinstance(result, np.ndarray) else result,
                "converged": converged,
                "iterations": len(steps) - 1,
                "processing_time": processing_time,
                "steps": [step.tolist() if isinstance(step, np.ndarray) else step for step in steps]
            }
            
            with open(args.output, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to {args.output}")
        
        return 0
    
    def command_evolve(self, args):
        """Execute the evolve command."""
        # Load test cases
        if args.test_cases:
            try:
                with open(args.test_cases, 'r') as f:
                    test_cases = json.load(f)
            except Exception as e:
                print(f"Error loading test cases: {str(e)}")
                return 1
        else:
            # Default test cases if none provided
            test_cases = [0.5, -1.0, 2.0, [0.8, 0.2], [[0.7, 0.3], [0.2, 0.8]]]
        
        # Initialize evolutionary engine
        engine = EvolutionaryEngine(
            population_size=args.population,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate
        )
        
        # Run evolution
        print(f"Starting evolution with population={args.population}, generations={args.generations}")
        print(f"Test cases: {test_cases}")
        
        start_time = time.time()
        results = engine.evolve(test_cases, generations=args.generations)
        end_time = time.time()
        
        # Display results
        print(f"Evolution completed in {end_time - start_time:.4f} seconds")
        print(f"Best fitness: {results['best_fitness']:.4f}")
        print(f"Population diversity: {results['diversity']:.4f}")
        
        print("\nTop evolved rules:")
        for i, (name, _) in enumerate(results['best_rules'][:5], 1):
            print(f"{i}. {name}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                # Clean numpy arrays for JSON serialization
                clean_results = {
                    "generations": args.generations,
                    "population_size": args.population,
                    "best_fitness": float(results['best_fitness']),
                    "avg_fitness": float(results['avg_fitness']),
                    "diversity": float(results['diversity']),
                    "best_rules": [{"name": name} for name, _ in results['best_rules'][:5]]
                }
                json.dump(clean_results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        return 0
    
    def command_meta_resolve(self, args):
        """Execute the meta-resolve command."""
        # Parse and validate input
        paradox_input = self.parse_input(args.input, args.type)
        
        # Create meta-resolver based on selected framework
        meta = MetaResolver()
        
        if args.framework == "standard":
            meta.create_standard_framework()
        elif args.framework == "custom" and args.config:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                # Custom framework initialization would go here
                # This is a simplified version
                meta.create_standard_framework()  # Fallback to standard for demo
            except Exception as e:
                print(f"Error loading custom framework: {str(e)}")
                return 1
        else:
            meta.create_standard_framework()
        
        # Execute meta-resolution
        print(f"Beginning meta-resolution with framework: {args.framework}")
        print(f"Initial state: {paradox_input}")
        
        start_time = time.time()
        result = meta.resolve(
            paradox_input, 
            args.type,
            max_phase_transitions=args.max_transitions
        )
        end_time = time.time()
        
        # Display results
        processing_time = end_time - start_time
        print(f"Meta-resolution completed in {processing_time:.4f} seconds")
        print(f"Final state: {result['final_state']}")
        print(f"Phase transitions: {result['phase_transitions']}")
        print(f"Total iterations: {result['total_iterations']}")
        
        # Save results if output file specified
        if args.output:
            # Clean numpy arrays for JSON serialization
            if isinstance(result['final_state'], np.ndarray):
                result['final_state'] = result['final_state'].tolist()
            
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        
        return 0
    
    def command_rules(self, args):
        """Execute the rules command."""
        available_rules = get_available_rules()
        
        print("Available transformation rules:")
        print("------------------------------")
        
        for name in available_rules:
            if args.verbose:
                # In a full implementation, we would include detailed descriptions
                print(f"\n{name}:")
                print(f"  Type: Standard")
                print(f"  Description: Transformation rule for paradox resolution")
            else:
                print(f"- {name}")
        
        return 0
    
    def run(self, args=None):
        """
        Parse arguments and execute the appropriate command.
        
        Args:
            args: Command-line arguments (default: sys.argv[1:])
        """
        parsed_args = self.parser.parse_args(args)
        
        if parsed_args.command == "resolve":
            return self.command_resolve(parsed_args)
        elif parsed_args.command == "evolve":
            return self.command_evolve(parsed_args)
        elif parsed_args.command == "meta-resolve":
            return self.command_meta_resolve(parsed_args)
        elif parsed_args.command == "rules":
            return self.command_rules(parsed_args)
        else:
            self.parser.print_help()
            return 1

def main():
    """Main entry point for the CLI."""
    cli = ParadoxCLI()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())