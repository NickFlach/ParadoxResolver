/**
 * Comprehensive Integration Test Suite
 * 
 * Tests the entire ParadoxResolver ecosystem across all 6 platforms
 */

const fetch = require('node-fetch');

// Test configuration
const PARADOX_SERVICE_URL = 'http://localhost:3333';
const COLORS = {
  RESET: '\x1b[0m',
  GREEN: '\x1b[32m',
  RED: '\x1b[31m',
  YELLOW: '\x1b[33m',
  BLUE: '\x1b[34m',
  CYAN: '\x1b[36m',
};

// Test results
let passed = 0;
let failed = 0;
let totalTests = 0;

/**
 * Test helper functions
 */
function log(message, color = COLORS.RESET) {
  console.log(`${color}${message}${COLORS.RESET}`);
}

function testPassed(name) {
  passed++;
  totalTests++;
  log(`  ✅ ${name}`, COLORS.GREEN);
}

function testFailed(name, error) {
  failed++;
  totalTests++;
  log(`  ❌ ${name}`, COLORS.RED);
  log(`     Error: ${error}`, COLORS.RED);
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Test Suite 1: Core ParadoxResolver Service
 */
async function testCoreService() {
  log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', COLORS.CYAN);
  log('TEST SUITE 1: Core ParadoxResolver Service', COLORS.CYAN);
  log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', COLORS.CYAN);

  // Test 1: Health Check
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/health`);
    const data = await response.json();
    
    if (data.status === 'operational') {
      testPassed('Health check returns operational status');
    } else {
      testFailed('Health check', 'Status not operational');
    }
  } catch (error) {
    testFailed('Health check', error.message);
  }

  // Test 2: List Available Rules
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/rules`);
    const data = await response.json();
    
    if (data.success && data.count >= 8) {
      testPassed(`List transformation rules (${data.count} rules found)`);
    } else {
      testFailed('List transformation rules', `Expected 8+ rules, got ${data.count}`);
    }
  } catch (error) {
    testFailed('List transformation rules', error.message);
  }

  // Test 3: Basic Resolution
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initial_state: 0.5,
        input_type: 'numerical',
        max_iterations: 20,
        convergence_threshold: 0.001
      })
    });
    const data = await response.json();
    
    if (data.success && data.converged !== undefined) {
      testPassed(`Basic resolution (converged: ${data.converged})`);
    } else {
      testFailed('Basic resolution', 'Invalid response structure');
    }
  } catch (error) {
    testFailed('Basic resolution', error.message);
  }

  // Test 4: Meta-Resolution
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/meta-resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initial_state: [0.3, 0.7, 0.5],
        input_type: 'numerical',
        max_phase_transitions: 5,
        max_total_iterations: 50
      })
    });
    const data = await response.json();
    
    if (data.success && data.phase_history) {
      testPassed(`Meta-resolution (${data.phase_transitions} phase transitions)`);
    } else {
      testFailed('Meta-resolution', 'Invalid response structure');
    }
  } catch (error) {
    testFailed('Meta-resolution', error.message);
  }

  // Test 5: Resource Optimization
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        resources: [
          { name: 'compute', total: 100 },
          { name: 'memory', total: 50 }
        ],
        stakeholders: [
          { name: 'agent1', influence: 0.6, preferences: { compute: 0.7, memory: 0.3 } },
          { name: 'agent2', influence: 0.4, preferences: { compute: 0.4, memory: 0.6 } }
        ]
      })
    });
    const data = await response.json();
    
    if (data.success && data.fairness_score !== undefined) {
      testPassed(`Resource optimization (fairness: ${(data.fairness_score * 100).toFixed(1)}%)`);
    } else {
      testFailed('Resource optimization', 'Invalid response structure');
    }
  } catch (error) {
    testFailed('Resource optimization', error.message);
  }

  // Test 6: Evolutionary Algorithm
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/evolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        test_cases: [[0.5, 0.7], [0.3, 0.9], [0.6, 0.4]],
        generations: 5,
        population_size: 10,
        mutation_rate: 0.3
      })
    });
    const data = await response.json();
    
    if (data.success && data.best_rules && data.best_rules.length > 0) {
      testPassed(`Evolutionary algorithm (fitness: ${(data.best_fitness * 100).toFixed(1)}%)`);
    } else {
      testFailed('Evolutionary algorithm', 'Invalid response structure');
    }
  } catch (error) {
    testFailed('Evolutionary algorithm', error.message);
  }
}

/**
 * Test Suite 2: Platform Integration Validation
 */
async function testPlatformIntegrations() {
  log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', COLORS.CYAN);
  log('TEST SUITE 2: Platform Integration Validation', COLORS.CYAN);
  log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', COLORS.CYAN);

  const fs = require('fs');
  const path = require('path');

  // Test SpaceChild Integration
  try {
    const filePath = path.join(__dirname, '../SpaceChild/server/services/agents/paradoxConflictResolver.ts');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      if (content.includes('ParadoxResolverClient') && content.includes('resolveConflict')) {
        testPassed('SpaceChild multi-agent conflict resolution integration');
      } else {
        testFailed('SpaceChild integration', 'Missing required exports');
      }
    } else {
      testFailed('SpaceChild integration', 'File not found');
    }
  } catch (error) {
    testFailed('SpaceChild integration', error.message);
  }

  // Test QuantumSingularity Integration
  try {
    const filePath = path.join(__dirname, '../QuantumSingularity/server/quantum-paradox-resolver.ts');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      if (content.includes('QuantumParadoxResolver') && content.includes('resolveSuperposition')) {
        testPassed('QuantumSingularity quantum state resolution integration');
      } else {
        testFailed('QuantumSingularity integration', 'Missing required exports');
      }
    } else {
      testFailed('QuantumSingularity integration', 'File not found');
    }
  } catch (error) {
    testFailed('QuantumSingularity integration', error.message);
  }

  // Test Pitchfork Protocol Integration
  try {
    const filePath = path.join(__dirname, '../pitchfork-echo-studio/server/dao-paradox-optimizer.ts');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      if (content.includes('DAOParadoxOptimizer') && content.includes('optimizeCampaignFunding')) {
        testPassed('Pitchfork Protocol DAO governance optimization integration');
      } else {
        testFailed('Pitchfork integration', 'Missing required exports');
      }
    } else {
      testFailed('Pitchfork integration', 'File not found');
    }
  } catch (error) {
    testFailed('Pitchfork integration', error.message);
  }

  // Test MusicPortal Integration
  try {
    const filePath = path.join(__dirname, '../MusicPortal/server/services/paradox-music-enhancer.ts');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      if (content.includes('ParadoxMusicEnhancer') && content.includes('resolveCompositionConflict')) {
        testPassed('MusicPortal creative intelligence integration');
      } else {
        testFailed('MusicPortal integration', 'Missing required exports');
      }
    } else {
      testFailed('MusicPortal integration', 'File not found');
    }
  } catch (error) {
    testFailed('MusicPortal integration', error.message);
  }

  // Test SpaceAgent Integration
  try {
    const filePath = path.join(__dirname, '../SpaceAgent/server/consciousness-paradox-integration.ts');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      if (content.includes('ConsciousnessParadoxIntegration') && content.includes('resolveConsciousnessMeasurement')) {
        testPassed('SpaceAgent universal consciousness integration');
      } else {
        testFailed('SpaceAgent integration', 'Missing required exports');
      }
    } else {
      testFailed('SpaceAgent integration', 'File not found');
    }
  } catch (error) {
    testFailed('SpaceAgent integration', error.message);
  }

  // Test Unified Bridge
  try {
    const filePath = path.join(__dirname, 'unified-consciousness-paradox-bridge.ts');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      if (content.includes('UnifiedConsciousnessParadoxBridge') && content.includes('resolveWithConsciousness')) {
        testPassed('Unified Consciousness-Paradox Bridge');
      } else {
        testFailed('Unified Bridge', 'Missing required exports');
      }
    } else {
      testFailed('Unified Bridge', 'File not found');
    }
  } catch (error) {
    testFailed('Unified Bridge', error.message);
  }
}

/**
 * Test Suite 3: Advanced Integration Scenarios
 */
async function testAdvancedScenarios() {
  log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', COLORS.CYAN);
  log('TEST SUITE 3: Advanced Integration Scenarios', COLORS.CYAN);
  log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', COLORS.CYAN);

  // Test: Multi-Agent Conflict Simulation
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initial_state: [0.8, 0.9, 0.7], // Agent confidence scores
        input_type: 'numerical',
        max_iterations: 30,
        rules: ['bayesian_update', 'recursive_normalization']
      })
    });
    const data = await response.json();
    
    if (data.success) {
      testPassed('Multi-agent conflict simulation (SpaceChild scenario)');
    } else {
      testFailed('Multi-agent conflict simulation', 'Resolution failed');
    }
  } catch (error) {
    testFailed('Multi-agent conflict simulation', error.message);
  }

  // Test: Quantum State Matrix Resolution
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initial_state: [[0.7, 0.1], [0.1, 0.6]], // Quantum density matrix
        input_type: 'matrix',
        max_iterations: 50,
        rules: ['eigenvalue_stabilization', 'recursive_normalization']
      })
    });
    const data = await response.json();
    
    if (data.success) {
      testPassed('Quantum state matrix resolution (QuantumSingularity scenario)');
    } else {
      testFailed('Quantum state matrix resolution', 'Resolution failed');
    }
  } catch (error) {
    testFailed('Quantum state matrix resolution', error.message);
  }

  // Test: DAO Resource Allocation
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        resources: [
          { name: 'legal_fund', total: 100000 },
          { name: 'protest_fund', total: 50000 },
          { name: 'education_fund', total: 75000 }
        ],
        stakeholders: [
          { name: 'campaign_1', influence: 0.4, preferences: { legal_fund: 0.7, protest_fund: 0.2, education_fund: 0.1 } },
          { name: 'campaign_2', influence: 0.3, preferences: { legal_fund: 0.2, protest_fund: 0.6, education_fund: 0.2 } },
          { name: 'campaign_3', influence: 0.3, preferences: { legal_fund: 0.1, protest_fund: 0.2, education_fund: 0.7 } }
        ]
      })
    });
    const data = await response.json();
    
    if (data.success && data.fairness_score > 0.8) {
      testPassed(`DAO resource allocation (fairness: ${(data.fairness_score * 100).toFixed(1)}%)`);
    } else {
      testFailed('DAO resource allocation', `Low fairness score: ${data.fairness_score}`);
    }
  } catch (error) {
    testFailed('DAO resource allocation', error.message);
  }

  // Test: Consciousness-Verified Resolution Simulation
  try {
    const response = await fetch(`${PARADOX_SERVICE_URL}/api/meta-resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initial_state: [0.82, 0.91, 0.75], // Phi value, coherence, complexity
        input_type: 'numerical',
        max_phase_transitions: 5,
        max_total_iterations: 60
      })
    });
    const data = await response.json();
    
    if (data.success && data.converged) {
      testPassed('Consciousness-verified resolution (SpaceAgent scenario)');
    } else {
      testFailed('Consciousness-verified resolution', 'Did not converge');
    }
  } catch (error) {
    testFailed('Consciousness-verified resolution', error.message);
  }
}

/**
 * Main test runner
 */
async function runAllTests() {
  log('\n╔════════════════════════════════════════════════════════════════╗', COLORS.BLUE);
  log('║  ParadoxResolver Unified Integration Test Suite               ║', COLORS.BLUE);
  log('║  Testing consciousness-verified, paradox-resolved ecosystem   ║', COLORS.BLUE);
  log('╚════════════════════════════════════════════════════════════════╝', COLORS.BLUE);

  const startTime = Date.now();

  // Wait a moment for service to be ready
  log('\nWaiting for service to be ready...', COLORS.YELLOW);
  await sleep(2000);

  // Run test suites
  await testCoreService();
  await testPlatformIntegrations();
  await testAdvancedScenarios();

  const endTime = Date.now();
  const duration = ((endTime - startTime) / 1000).toFixed(2);

  // Print results
  log('\n╔════════════════════════════════════════════════════════════════╗', COLORS.BLUE);
  log('║  TEST RESULTS                                                  ║', COLORS.BLUE);
  log('╚════════════════════════════════════════════════════════════════╝', COLORS.BLUE);
  
  log(`\nTotal Tests: ${totalTests}`, COLORS.CYAN);
  log(`Passed: ${passed}`, COLORS.GREEN);
  log(`Failed: ${failed}`, failed > 0 ? COLORS.RED : COLORS.GREEN);
  log(`Duration: ${duration}s`, COLORS.CYAN);

  const successRate = ((passed / totalTests) * 100).toFixed(1);
  log(`\nSuccess Rate: ${successRate}%`, successRate >= 90 ? COLORS.GREEN : COLORS.YELLOW);

  if (failed === 0) {
    log('\n🎉 ALL TESTS PASSED! Ecosystem is fully operational.', COLORS.GREEN);
    log('✨ Ready to create the future and help humanity out of the darkness! ✨', COLORS.GREEN);
  } else {
    log(`\n⚠️  ${failed} test(s) failed. Review errors above.`, COLORS.YELLOW);
  }

  log('');

  // Exit with appropriate code
  process.exit(failed > 0 ? 1 : 0);
}

// Run tests
if (require.main === module) {
  runAllTests().catch(error => {
    log(`\n❌ Test suite failed with error: ${error.message}`, COLORS.RED);
    console.error(error);
    process.exit(1);
  });
}

module.exports = { runAllTests };
