#!/usr/bin/env python3
"""
List available transformation rules endpoint
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformation_rules import get_available_rules

def main():
    try:
        # Get all available rules
        rules = get_available_rules()
        
        # Prepare output with rule descriptions
        rules_info = {}
        
        rule_descriptions = {
            'fixed_point_iteration': 'Finds stable points for recursive equations using damped iteration',
            'contradiction_resolution': 'Transforms logical contradictions into consistent states',
            'self_reference_unwinding': 'Resolves self-referential statements by unwinding one level',
            'eigenvalue_stabilization': 'Stabilizes matrix-based paradoxes using eigenvalue techniques',
            'fuzzy_logic_transformation': 'Applies many-valued logic to resolve binary contradictions',
            'duality_inversion': 'Resolves paradoxes by inverting dualities and finding complementary perspectives',
            'bayesian_update': 'Applies Bayesian inference to update probabilities in paradoxical states',
            'recursive_normalization': 'Normalizes values while preserving structure of complex nested states'
        }
        
        for rule_name in rules.keys():
            rules_info[rule_name] = {
                'name': rule_name,
                'description': rule_descriptions.get(rule_name, 'No description available'),
                'available': True
            }
        
        output = {
            'success': True,
            'rules': rules_info,
            'count': len(rules_info)
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
