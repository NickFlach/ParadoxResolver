#!/usr/bin/env python3
"""
Resource allocation optimization endpoint for ParadoxResolver service
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crypto_paradox_optimizer import CryptoParadoxOptimizer

def main():
    try:
        # Parse input from command line
        if len(sys.argv) > 1:
            config = json.loads(sys.argv[1])
        else:
            config = json.load(sys.stdin)
        
        resources = config.get('resources', [])
        stakeholders = config.get('stakeholders', [])
        
        # Create optimizer
        optimizer = CryptoParadoxOptimizer()
        
        # Prepare resources and stakeholders for optimizer
        optimizer_config = {
            'resources': resources,
            'stakeholders': stakeholders
        }
        
        # Run optimization
        result = optimizer.optimize_allocation(optimizer_config)
        
        # Prepare output
        output = {
            'success': True,
            'allocation': result.get('allocation', {}),
            'total_satisfaction': result.get('total_satisfaction', 0),
            'fairness_score': result.get('fairness_score', 0),
            'iterations': result.get('iterations', 0),
            'converged': result.get('converged', False),
            'stakeholder_satisfaction': result.get('stakeholder_satisfaction', {}),
            'resource_utilization': result.get('resource_utilization', {})
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
