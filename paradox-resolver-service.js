#!/usr/bin/env node
/**
 * ParadoxResolver HTTP Service
 * 
 * Provides REST API for TypeScript/Node.js projects to access
 * Python ParadoxResolver capabilities.
 */

const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PARADOX_RESOLVER_PORT || 3333;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

/**
 * Execute Python script and return results
 */
function executePythonScript(scriptPath, args) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath, ...args]);
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
      } else {
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (e) {
          resolve({ raw_output: stdout });
        }
      }
    });
    
    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * POST /api/resolve
 * 
 * Resolve a paradox using standard transformation rules
 * 
 * Body: {
 *   initial_state: any,
 *   input_type: "numerical" | "matrix" | "logical" | "text",
 *   max_iterations?: number,
 *   convergence_threshold?: number,
 *   rules?: string[]
 * }
 */
app.post('/api/resolve', async (req, res) => {
  try {
    const {
      initial_state,
      input_type = 'numerical',
      max_iterations = 20,
      convergence_threshold = 0.001,
      rules = []
    } = req.body;
    
    const scriptPath = path.join(__dirname, 'api_endpoints', 'resolve_endpoint.py');
    const args = [
      JSON.stringify({
        initial_state,
        input_type,
        max_iterations,
        convergence_threshold,
        rules
      })
    ];
    
    const result = await executePythonScript(scriptPath, args);
    res.json(result);
  } catch (error) {
    console.error('Resolution error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/meta-resolve
 * 
 * Resolve using meta-framework with phase transitions
 * 
 * Body: {
 *   initial_state: any,
 *   input_type: string,
 *   max_phase_transitions?: number,
 *   max_total_iterations?: number
 * }
 */
app.post('/api/meta-resolve', async (req, res) => {
  try {
    const {
      initial_state,
      input_type = 'numerical',
      max_phase_transitions = 10,
      max_total_iterations = 100
    } = req.body;
    
    const scriptPath = path.join(__dirname, 'api_endpoints', 'meta_resolve_endpoint.py');
    const args = [
      JSON.stringify({
        initial_state,
        input_type,
        max_phase_transitions,
        max_total_iterations
      })
    ];
    
    const result = await executePythonScript(scriptPath, args);
    res.json(result);
  } catch (error) {
    console.error('Meta-resolution error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/optimize
 * 
 * Optimize resource allocation among stakeholders
 * 
 * Body: {
 *   resources: Array<{name: string, total: number}>,
 *   stakeholders: Array<{
 *     name: string,
 *     influence: number,
 *     preferences: Record<string, number>
 *   }>
 * }
 */
app.post('/api/optimize', async (req, res) => {
  try {
    const { resources, stakeholders } = req.body;
    
    const scriptPath = path.join(__dirname, 'api_endpoints', 'optimize_endpoint.py');
    const args = [
      JSON.stringify({ resources, stakeholders })
    ];
    
    const result = await executePythonScript(scriptPath, args);
    res.json(result);
  } catch (error) {
    console.error('Optimization error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/evolve
 * 
 * Evolve novel transformation rules using genetic algorithms
 * 
 * Body: {
 *   test_cases: Array<any>,
 *   generations?: number,
 *   population_size?: number,
 *   mutation_rate?: number
 * }
 */
app.post('/api/evolve', async (req, res) => {
  try {
    const {
      test_cases,
      generations = 10,
      population_size = 20,
      mutation_rate = 0.3
    } = req.body;
    
    const scriptPath = path.join(__dirname, 'api_endpoints', 'evolve_endpoint.py');
    const args = [
      JSON.stringify({
        test_cases,
        generations,
        population_size,
        mutation_rate
      })
    ];
    
    const result = await executePythonScript(scriptPath, args);
    res.json(result);
  } catch (error) {
    console.error('Evolution error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/rules
 * 
 * Get list of available transformation rules
 */
app.get('/api/rules', async (req, res) => {
  try {
    const scriptPath = path.join(__dirname, 'api_endpoints', 'list_rules_endpoint.py');
    const result = await executePythonScript(scriptPath, []);
    res.json(result);
  } catch (error) {
    console.error('Rules listing error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /health
 * 
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({ status: 'operational', service: 'ParadoxResolver', version: '1.0.0' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🔮 ParadoxResolver Service running on http://localhost:${PORT}`);
  console.log(`📡 Endpoints:`);
  console.log(`   POST /api/resolve - Basic paradox resolution`);
  console.log(`   POST /api/meta-resolve - Meta-framework resolution`);
  console.log(`   POST /api/optimize - Resource allocation optimization`);
  console.log(`   POST /api/evolve - Evolutionary rule discovery`);
  console.log(`   GET /api/rules - List available transformation rules`);
  console.log(`   GET /health - Service health check`);
});
