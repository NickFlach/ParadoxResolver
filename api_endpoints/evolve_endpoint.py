#!/usr/bin/env python3
"""
Evolutionary engine endpoint for ParadoxResolver service
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolutionary_engine import EvolutionaryEngine
from transformation_rules import get_available_rules

def main():
    try:
        # Parse input from command line
        if len(sys.argv) > 1:
            config = json.loads(sys.argv[1])
        else:
            config = json.load(sys.stdin)
        
        test_cases = config.get('test_cases', [])
        generations = config.get('generations', 10)
        population_size = config.get('population_size', 20)
        mutation_rate = config.get('mutation_rate', 0.3)
        
        # Create evolutionary engine with seed rules
        seed_rules = get_available_rules()
        engine = EvolutionaryEngine(
            seed_rules=seed_rules,
            population_size=population_size,
            mutation_rate=mutation_rate
        )
        
        # Evolve
        result = engine.evolve(test_cases=test_cases, generations=generations)
        
        # Prepare output (exclude actual functions, just names and fitness)
        best_rules_info = [
            {
                'name': name,
                'fitness': engine.population[i].fitness if i < len(engine.population) else 0,
                'complexity': engine.population[i].complexity if i < len(engine.population) else 0,
                'components': engine.population[i].components if i < len(engine.population) else []
            }
            for i, (name, func) in enumerate(result['best_rules'][:5])
        ]
        
        output = {
            'success': True,
            'best_rules': best_rules_info,
            'best_fitness': result['best_fitness'],
            'avg_fitness': result['avg_fitness'],
            'diversity': result['diversity'],
            'generations': result['generations'],
            'population_size': result['population_size'],
            'history': result['history']
        }
        
        print(json.dumps(output))
    
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()
