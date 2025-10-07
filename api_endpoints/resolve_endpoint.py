#!/usr/bin/env python3
"""
Basic resolution endpoint for ParadoxResolver service
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paradox_resolver import ParadoxResolver
from transformation_rules import get_available_rules

def main():
    try:
        # Parse input from command line
        if len(sys.argv) > 1:
            config = json.loads(sys.argv[1])
        else:
            config = json.load(sys.stdin)
        
        initial_state = config.get('initial_state')
        input_type = config.get('input_type', 'numerical')
        max_iterations = config.get('max_iterations', 20)
        convergence_threshold = config.get('convergence_threshold', 0.001)
        rule_names = config.get('rules', [])
        
        # Get rules
        all_rules = get_available_rules()
        
        # Filter rules if specified
        if rule_names:
            rules = {name: all_rules[name] for name in rule_names if name in all_rules}
        else:
            rules = all_rules
        
        # Create resolver
        resolver = ParadoxResolver(
            transformation_rules=rules,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )
        
        # Resolve
        final_state, states, converged = resolver.resolve(initial_state)
        
        # Prepare output
        result = {
            'success': True,
            'final_state': final_state if not hasattr(final_state, 'tolist') else final_state.tolist(),
            'converged': converged,
            'iterations': len(states) - 1,
            'states_history': [s if not hasattr(s, 'tolist') else s.tolist() for s in states],
            'input_type': input_type
        }
        
        print(json.dumps(result))
    
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
