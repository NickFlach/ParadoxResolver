#!/usr/bin/env python3
"""
Crypto_ParadoxOS Command Line Interface

This module provides a unified command-line interface to the Crypto_ParadoxOS
system, allowing access to all major features through a simple CLI.
"""

import argparse
import json
import sys
import numpy as np
import time
from typing import Any, Dict, List, Optional

# Import core functionality
from crypto_paradox_os import ParadoxResolver, ParadoxState, TransformationRule
from crypto_paradox_os import get_standard_rules, main as core_main
from crypto_paradox_api import CryptoParadoxAPI, demo_api
from crypto_paradox_optimizer import (
    AllocationOptimizer, AllocationProblem, Resource, Stakeholder, 
    demo_allocation
)


class ParadoxCLI:
    """Command-line interface for the Crypto_ParadoxOS system."""
    
    def __init__(self):
        """Initialize the CLI with parsers and commands."""
        self.api = CryptoParadoxAPI()
        self.optimizer = AllocationOptimizer(api=self.api)
        
        # Set up argument parser
        self.parser = argparse.ArgumentParser(
            description="Crypto_ParadoxOS: Recursive Paradox-Resolution System"
        )
        self._setup_parsers()
    
    def _setup_parsers(self):
        """Set up command-line argument parsers."""
        subparsers = self.parser.add_subparsers(
            dest="command",
            help="Command to execute"
        )
        
        # 'resolve' command
        resolve_parser = subparsers.add_parser(
            "resolve",
            help="Resolve a paradoxical input"
        )
        resolve_parser.add_argument(
            "input",
            help="Input value or path to input file"
        )
        resolve_parser.add_argument(
            "--type",
            choices=["numerical", "matrix", "text"],
            default="numerical",
            help="Type of input data"
        )
        resolve_parser.add_argument(
            "--iterations",
            type=int,
            default=20,
            help="Maximum iterations for resolution"
        )
        resolve_parser.add_argument(
            "--threshold",
            type=float,
            default=0.001,
            help="Convergence threshold"
        )
        resolve_parser.add_argument(
            "--rules",
            nargs="+",
            help="Specific rules to apply (space-separated)"
        )
        resolve_parser.add_argument(
            "--output",
            help="Path to output file for results (JSON)"
        )
        
        # 'optimize' command
        optimize_parser = subparsers.add_parser(
            "optimize",
            help="Optimize resource allocations"
        )
        optimize_parser.add_argument(
            "config",
            help="Path to optimization configuration file (JSON)"
        )
        optimize_parser.add_argument(
            "--iterations",
            type=int,
            default=50,
            help="Maximum iterations for optimization"
        )
        optimize_parser.add_argument(
            "--threshold",
            type=float,
            default=0.0001,
            help="Convergence threshold"
        )
        optimize_parser.add_argument(
            "--output",
            help="Path to output file for results (JSON)"
        )
        
        # 'demo' command
        demo_parser = subparsers.add_parser(
            "demo",
            help="Run a demonstration of the system"
        )
        demo_parser.add_argument(
            "type",
            choices=["basic", "api", "allocation"],
            default="basic",
            help="Type of demonstration to run"
        )
        
        # 'rules' command
        rules_parser = subparsers.add_parser(
            "rules",
            help="List available transformation rules"
        )
        
        # 'interactive' command
        interactive_parser = subparsers.add_parser(
            "interactive",
            help="Start interactive mode"
        )
    
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
            try:
                return float(input_str)
            except ValueError:
                # Check if it's a file path
                try:
                    with open(input_str, 'r') as f:
                        return float(f.read().strip())
                except:
                    print(f"Error: Could not parse '{input_str}' as a numerical value")
                    sys.exit(1)
        
        elif input_type == "matrix":
            # Check if it's a file path
            try:
                with open(input_str, 'r') as f:
                    lines = f.readlines()
                    matrix = []
                    for line in lines:
                        row = [float(x) for x in line.strip().split()]
                        matrix.append(row)
                    return np.array(matrix)
            except:
                print(f"Error: Could not parse '{input_str}' as a matrix file")
                print("Matrix files should contain space-separated values, one row per line")
                sys.exit(1)
        
        elif input_type == "text":
            # Check if it's a file path
            try:
                with open(input_str, 'r') as f:
                    return f.read().strip()
            except:
                # Assume it's a direct text input
                return input_str
        
        # Default case
        return input_str
    
    def load_optimization_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load optimization configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing the optimization configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            if "resources" not in config or "stakeholders" not in config:
                raise ValueError("Configuration must include 'resources' and 'stakeholders'")
            
            return config
            
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            sys.exit(1)
    
    def create_optimization_problem(self, config: Dict[str, Any]) -> AllocationProblem:
        """
        Create an allocation problem from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            AllocationProblem instance
        """
        # Create resources
        resources = []
        for r_config in config["resources"]:
            resources.append(Resource(
                name=r_config["name"],
                total=r_config["total"],
                min_allocation=r_config.get("min_allocation", 0.0),
                max_allocation=r_config.get("max_allocation")
            ))
        
        # Create stakeholders
        stakeholders = []
        for s_config in config["stakeholders"]:
            stakeholders.append(Stakeholder(
                name=s_config["name"],
                influence=s_config.get("influence", 1.0),
                preferences=s_config.get("preferences", {})
            ))
        
        # Create and return the problem
        return AllocationProblem(resources, stakeholders)
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to output file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        try:
            with open(output_path, 'w') as f:
                json.dump(convert_numpy(results), f, indent=2)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    def run_interactive(self):
        """Run the CLI in interactive mode."""
        print("Crypto_ParadoxOS Interactive Mode")
        print("=================================")
        print("Type 'help' for commands, 'exit' to quit")
        
        while True:
            try:
                command = input("\nparadox> ").strip()
                
                if command.lower() in ('exit', 'quit', 'q'):
                    break
                
                elif command.lower() in ('help', '?', 'h'):
                    print("\nAvailable commands:")
                    print("  resolve [value] - Resolve a paradoxical value")
                    print("  rules - List available transformation rules")
                    print("  optimize - Start optimization wizard")
                    print("  demo - Run a demonstration")
                    print("  exit/quit - Exit interactive mode")
                
                elif command.lower().startswith('resolve'):
                    parts = command.split(maxsplit=1)
                    if len(parts) == 1:
                        value = input("Enter value to resolve: ")
                    else:
                        value = parts[1]
                    
                    input_type = input("Enter input type (numerical, matrix, text) [numerical]: ").strip()
                    if not input_type:
                        input_type = "numerical"
                    
                    try:
                        parsed_input = self.parse_input(value, input_type)
                        print(f"Resolving {input_type} input: {parsed_input}")
                        
                        config = {
                            "max_iterations": 20,
                            "convergence_threshold": 0.001
                        }
                        
                        result = self.api.resolve_paradox(parsed_input, input_type, config)
                        
                        print("\nResolution Results:")
                        print(f"Converged: {result['result']['converged']}")
                        print(f"Iterations: {result['result']['iterations']}")
                        print(f"Final state: {result['result']['final_state']}")
                        
                    except Exception as e:
                        print(f"Error: {str(e)}")
                
                elif command.lower() == 'rules':
                    print("\nAvailable Transformation Rules:")
                    for name, desc in self.api.get_available_rules().items():
                        print(f"- {name}: {desc}")
                
                elif command.lower() == 'optimize':
                    print("\nOptimization wizard not implemented in interactive mode.")
                    print("Use 'paradox_cli.py optimize config.json' for optimization.")
                
                elif command.lower().startswith('demo'):
                    parts = command.split(maxsplit=1)
                    if len(parts) == 1:
                        demo_type = input("Enter demo type (basic, api, allocation) [basic]: ").strip()
                        if not demo_type:
                            demo_type = "basic"
                    else:
                        demo_type = parts[1]
                    
                    if demo_type == "basic":
                        core_main()
                    elif demo_type == "api":
                        demo_api()
                    elif demo_type == "allocation":
                        demo_allocation()
                    else:
                        print(f"Unknown demo type: {demo_type}")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def run(self, args=None):
        """
        Parse arguments and execute the appropriate command.
        
        Args:
            args: Command-line arguments (default: sys.argv[1:])
        """
        parsed_args = self.parser.parse_args(args)
        
        if parsed_args.command == "resolve":
            # Parse the input
            input_value = self.parse_input(parsed_args.input, parsed_args.type)
            
            # Configure the resolution
            config = {
                "max_iterations": parsed_args.iterations,
                "convergence_threshold": parsed_args.threshold
            }
            
            if parsed_args.rules:
                config["rules"] = parsed_args.rules
            
            # Resolve the paradox
            print(f"Resolving {parsed_args.type} paradox...")
            result = self.api.resolve_paradox(input_value, parsed_args.type, config)
            
            # Display the results
            print(f"\nResolution completed in {result['result']['execution_time']:.4f} seconds")
            print(f"Converged: {result['result']['converged']}")
            print(f"Iterations: {result['result']['iterations']}")
            print(f"Final state: {result['result']['final_state']}")
            
            # Save results if output path specified
            if parsed_args.output:
                self.save_results(result, parsed_args.output)
        
        elif parsed_args.command == "optimize":
            # Load the optimization configuration
            config = self.load_optimization_config(parsed_args.config)
            
            # Create the allocation problem
            problem = self.create_optimization_problem(config)
            
            # Configure the optimizer
            self.optimizer.max_iterations = parsed_args.iterations
            self.optimizer.convergence_threshold = parsed_args.threshold
            
            # Perform the optimization
            print("Optimizing resource allocations...")
            result = self.optimizer.optimize(problem)
            
            # Display the results
            print(f"\nOptimization completed in {result['execution_time']:.4f} seconds")
            print(f"Converged: {result['converged']}")
            print(f"Iterations: {result['iterations']}")
            
            print("\nResource Allocations:")
            for r_name, r_data in result["resource_allocations"].items():
                print(f"\n{r_name} (Total: ${r_data['total']:,.2f}):")
                for s_name, amount in r_data["allocations"].items():
                    print(f"  {s_name}: ${amount:,.2f} ({amount/r_data['total']*100:.1f}%)")
            
            # Save results if output path specified
            if parsed_args.output:
                self.save_results(result, parsed_args.output)
        
        elif parsed_args.command == "demo":
            if parsed_args.type == "basic":
                core_main()
            elif parsed_args.type == "api":
                demo_api()
            elif parsed_args.type == "allocation":
                demo_allocation()
        
        elif parsed_args.command == "rules":
            print("Available Transformation Rules:")
            for name, desc in self.api.get_available_rules().items():
                print(f"- {name}: {desc}")
        
        elif parsed_args.command == "interactive":
            self.run_interactive()
        
        else:
            # If no command specified, print help
            self.parser.print_help()


def main():
    """Main entry point for the CLI."""
    cli = ParadoxCLI()
    cli.run()


if __name__ == "__main__":
    main()