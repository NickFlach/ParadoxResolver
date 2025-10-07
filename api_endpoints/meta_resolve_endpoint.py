#!/usr/bin/env python3
"""
Meta-resolution endpoint for ParadoxResolver service
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_resolver import MetaResolver

def main():
    try:
        # Parse input from command line
        if len(sys.argv) > 1:
            config = json.loads(sys.argv[1])
        else:
            config = json.load(sys.stdin)
        
        initial_state = config.get('initial_state')
        input_type = config.get('input_type', 'numerical')
        max_phase_transitions = config.get('max_phase_transitions', 10)
        max_total_iterations = config.get('max_total_iterations', 100)
        
        # Create meta-resolver with standard framework
        meta = MetaResolver()
        meta.create_standard_framework()
        
        # Resolve
        result = meta.resolve(
            initial_state=initial_state,
            input_type=input_type,
            max_phase_transitions=max_phase_transitions,
            max_total_iterations=max_total_iterations
        )
        
        # Prepare output
        output = {
            'success': True,
            'final_state': result['final_state'] if not hasattr(result['final_state'], 'tolist') else result['final_state'].tolist(),
            'converged': result['converged'],
            'total_iterations': result['total_iterations'],
            'phase_transitions': result['phase_transitions'],
            'phase_history': result['phase_history'],
            'phase_results': result['phase_results'],
            'input_type': input_type
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
