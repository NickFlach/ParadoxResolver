import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="API Documentation",
    page_icon="🤖",
    layout="wide"
)

st.title("Crypto_ParadoxOS API for AI Integration")
st.markdown("""
## API Documentation for AI Systems

This documentation is designed for AI systems that want to integrate with Crypto_ParadoxOS.
The API provides structured access to the paradox resolution capabilities and evolutionary engine,
enabling other AI systems to use Crypto_ParadoxOS as a component in their own processing.

### Why Use Crypto_ParadoxOS in Your AI System?

* **Resolution of Logical Conflicts**: Resolve conflicting information or requirements
* **Novel Solution Generation**: Discover creative approaches to complex problems
* **Equilibrium Finding**: Identify stable states in dynamic systems
* **Insight Generation**: Extract meta-patterns from seemingly contradictory data
""")

# API Overview
st.header("API Overview")

st.markdown("""
The Crypto_ParadoxOS API follows RESTful principles with JSON-based request and response formats.
Endpoints are organized around core functionalities:

1. **Paradox Resolution**: Submit paradoxes for resolution and retrieve results
2. **Evolutionary Engine**: Access the evolutionary engine for generating novel solutions
3. **Meta-Resolver**: Use the meta-resolution framework for complex paradoxes
4. **Interactive Evolution**: Engage in guided evolution with feedback loops
""")

# Endpoints Documentation
st.header("API Endpoints")

# Create a dataframe for the endpoints
endpoints = [
    {
        "Endpoint": "/api/resolve",
        "Method": "POST",
        "Description": "Resolve a paradox using specified transformation rules",
        "Parameters": "paradox_input, input_type, transformation_rules, max_iterations, convergence_threshold",
        "Response": "Resolution result, steps, convergence status, and metrics"
    },
    {
        "Endpoint": "/api/evolve",
        "Method": "POST",
        "Description": "Run the evolutionary engine to generate novel transformation rules",
        "Parameters": "test_cases, generations, population_size, mutation_rate, crossover_rate",
        "Response": "Evolved rules, fitness metrics, and evolution history"
    },
    {
        "Endpoint": "/api/meta_resolve",
        "Method": "POST",
        "Description": "Use the meta-resolver framework for complex paradoxes",
        "Parameters": "paradox_input, input_type, phase_configuration, max_phase_transitions",
        "Response": "Resolution result, phase history, transition points, and metrics"
    },
    {
        "Endpoint": "/api/interactive_evolve",
        "Method": "POST",
        "Description": "Start or continue an interactive evolution session",
        "Parameters": "session_id, selected_rules, feedback, test_cases",
        "Response": "New generation of rules, evaluation metrics, and session state"
    },
    {
        "Endpoint": "/api/rules",
        "Method": "GET",
        "Description": "Retrieve available transformation rules",
        "Parameters": "rule_type (standard, evolved, all)",
        "Response": "List of rules with descriptions and metadata"
    },
    {
        "Endpoint": "/api/jobs/{job_id}",
        "Method": "GET",
        "Description": "Check status of an asynchronous resolution or evolution job",
        "Parameters": "job_id",
        "Response": "Job status, progress, and results if completed"
    }
]

endpoints_df = pd.DataFrame(endpoints)
st.dataframe(endpoints_df, use_container_width=True)

# Parameter Details
st.header("Parameter Details")

param_tabs = st.tabs(["Resolution Parameters", "Evolution Parameters", "Meta-Resolution Parameters", "Interactive Evolution Parameters"])

with param_tabs[0]:
    st.markdown("""
    ### Paradox Resolution Parameters
    
    | Parameter | Type | Description |
    | --- | --- | --- |
    | `paradox_input` | Mixed | The paradoxical input to resolve (string, number, or matrix) |
    | `input_type` | String | Type of input ("text", "numerical", or "matrix") |
    | `transformation_rules` | Array | List of rule names to apply (empty for default rules) |
    | `max_iterations` | Integer | Maximum iterations to perform (default: 20) |
    | `convergence_threshold` | Float | Threshold for determining convergence (default: 0.001) |
    | `initial_value` | Float | Initial value for numerical paradoxes (default: 0.5) |
    | `async` | Boolean | Whether to process asynchronously (default: false) |
    
    #### Example Request:
    """)
    
    example_resolution_request = {
        "paradox_input": "x = 1/x",
        "input_type": "numerical",
        "transformation_rules": ["Fixed-Point Iteration", "Recursive Normalization"],
        "max_iterations": 30,
        "convergence_threshold": 0.0001,
        "initial_value": 0.5,
        "async": False
    }
    
    st.code(json.dumps(example_resolution_request, indent=2), language="json")
    
    st.markdown("#### Example Response:")
    
    example_resolution_response = {
        "success": True,
        "result": {
            "final_state": 1.0,
            "converged": True,
            "iterations": 12,
            "processing_time": 0.023,
            "steps": [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 1.0],
            "delta_history": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0]
        },
        "metadata": {
            "input_type": "numerical",
            "applied_rules": ["Fixed-Point Iteration", "Recursive Normalization"],
            "complexity_estimate": 3
        }
    }
    
    st.code(json.dumps(example_resolution_response, indent=2), language="json")

with param_tabs[1]:
    st.markdown("""
    ### Evolution Parameters
    
    | Parameter | Type | Description |
    | --- | --- | --- |
    | `test_cases` | Array | Test cases for evaluating rule fitness |
    | `generations` | Integer | Number of generations to evolve (default: 10) |
    | `population_size` | Integer | Size of rule population (default: 20) |
    | `mutation_rate` | Float | Probability of mutation during reproduction (default: 0.3) |
    | `crossover_rate` | Float | Probability of crossover during reproduction (default: 0.7) |
    | `seed_rules` | Array | Names of rules to use as seeds (empty for defaults) |
    | `async` | Boolean | Whether to process asynchronously (default: false) |
    
    #### Example Request:
    """)
    
    example_evolution_request = {
        "test_cases": [0.5, -1.0, [[1.0, 0.0], [0.0, 1.0]]],
        "generations": 15,
        "population_size": 30,
        "mutation_rate": 0.4,
        "crossover_rate": 0.6,
        "seed_rules": ["Fixed-Point Iteration", "Duality Inversion"],
        "async": True
    }
    
    st.code(json.dumps(example_evolution_request, indent=2), language="json")
    
    st.markdown("#### Example Response:")
    
    example_evolution_response = {
        "success": True,
        "job_id": "evol_1a2b3c4d5e",
        "status": "processing",
        "estimated_completion_time": "2025-03-07T14:45:30Z",
        "progress": {
            "current_generation": 3,
            "total_generations": 15,
            "completion_percentage": 20,
            "metrics": {
                "current_max_fitness": 0.83,
                "current_avg_fitness": 0.56,
                "current_diversity": 0.72
            }
        }
    }
    
    st.code(json.dumps(example_evolution_response, indent=2), language="json")

with param_tabs[2]:
    st.markdown("""
    ### Meta-Resolution Parameters
    
    | Parameter | Type | Description |
    | --- | --- | --- |
    | `paradox_input` | Mixed | The paradoxical input to resolve |
    | `input_type` | String | Type of input ("text", "numerical", or "matrix") |
    | `phase_configuration` | Object | Configuration of resolution phases (null for default) |
    | `max_phase_transitions` | Integer | Maximum phase transitions to allow (default: 10) |
    | `async` | Boolean | Whether to process asynchronously (default: false) |
    
    #### Example Request:
    """)
    
    example_meta_request = {
        "paradox_input": [[0.8, 0.2], [0.3, 0.7]],
        "input_type": "matrix",
        "phase_configuration": {
            "initial_phase": "Exploration",
            "phases": {
                "Exploration": {
                    "is_convergent": False,
                    "max_iterations": 5,
                    "rules": ["Duality Inversion", "Bayesian Update"]
                },
                "Convergence": {
                    "is_convergent": True,
                    "max_iterations": 10,
                    "rules": ["Eigenvalue Stabilization", "Fixed-Point Iteration"]
                },
                "Refinement": {
                    "is_convergent": True,
                    "max_iterations": 8,
                    "rules": ["Recursive Normalization"]
                }
            },
            "transitions": {
                "Exploration->Convergence": "state.max_value > 2.0",
                "Convergence->Refinement": "state.delta < 0.01",
                "Refinement->Exploration": "state.iterations > 5 && state.delta > 0.1"
            }
        },
        "max_phase_transitions": 5,
        "async": False
    }
    
    st.code(json.dumps(example_meta_request, indent=2), language="json")
    
    st.markdown("#### Example Response:")
    
    example_meta_response = {
        "success": True,
        "result": {
            "final_state": [[1.0, 0.0], [0.0, 1.0]],
            "phase_history": ["Exploration", "Convergence", "Refinement"],
            "total_iterations": 12,
            "transitions": [
                {"from": "Exploration", "to": "Convergence", "after_iterations": 3, "state": [[1.2, 0.4], [0.6, 2.1]]},
                {"from": "Convergence", "to": "Refinement", "after_iterations": 7, "state": [[0.98, 0.02], [0.01, 0.99]]}
            ],
            "converged": True
        },
        "metadata": {
            "input_type": "matrix",
            "processing_time": 0.156,
            "phase_metrics": {
                "Exploration": {"iterations": 3, "delta_trend": [0.5, 0.8, 1.2]},
                "Convergence": {"iterations": 7, "delta_trend": [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]},
                "Refinement": {"iterations": 2, "delta_trend": [0.002, 0.0005]}
            }
        }
    }
    
    st.code(json.dumps(example_meta_response, indent=2), language="json")

with param_tabs[3]:
    st.markdown("""
    ### Interactive Evolution Parameters
    
    | Parameter | Type | Description |
    | --- | --- | --- |
    | `session_id` | String | ID of the session (null for new session) |
    | `selected_rules` | Array | IDs of rules selected for next generation |
    | `feedback` | Object | Feedback scores and comments for specific rules |
    | `test_cases` | Array | Test cases for evaluating rule fitness |
    | `evolution_parameters` | Object | Parameters for controlling evolution |
    
    #### Example Request:
    """)
    
    example_interactive_request = {
        "session_id": "session_a1b2c3d4e5",
        "selected_rules": ["rule_123", "rule_456", "rule_789"],
        "feedback": {
            "rule_123": {"score": 0.8, "comment": "Effective solution for numerical paradoxes"},
            "rule_456": {"score": 0.9, "comment": "Innovative approach with good convergence"},
            "rule_789": {"score": 0.7, "comment": "Interesting but needs refinement"}
        },
        "evolution_parameters": {
            "population_size": 10,
            "mutation_rate": 0.3,
            "preserve_diversity": True
        }
    }
    
    st.code(json.dumps(example_interactive_request, indent=2), language="json")
    
    st.markdown("#### Example Response:")
    
    example_interactive_response = {
        "success": True,
        "session_id": "session_a1b2c3d4e5",
        "generation": 2,
        "population": [
            {
                "rule_id": "rule_123_v2",
                "name": "Enhanced Fixed-Point",
                "parent_rules": ["rule_123"],
                "fitness": 0.85,
                "components": ["square", "normalize", "dampening", "inverse"],
                "explanation": "This rule applies squaring, normalization, dampening, and inversion in sequence."
            },
            {
                "rule_id": "rule_456_v2",
                "name": "Adaptive Stabilizer",
                "parent_rules": ["rule_456"],
                "fitness": 0.92,
                "components": ["normalize", "oscillate", "filter"],
                "explanation": "This rule normalizes values, applies a sine wave transformation, and filters small values."
            },
            {
                "rule_id": "rule_789_v2",
                "name": "Dynamic Reconstructor",
                "parent_rules": ["rule_789"],
                "fitness": 0.78,
                "components": ["filter", "smooth", "reorganize"],
                "explanation": "This rule filters noise, smooths values, and reorganizes the structure."
            }
        ],
        "evolution_metrics": {
            "avg_fitness": 0.81,
            "max_fitness": 0.92,
            "diversity": 0.68,
            "innovation_score": 0.45
        },
        "next_actions": ["select_rules", "provide_feedback", "test_rules", "end_session"]
    }
    
    st.code(json.dumps(example_interactive_response, indent=2), language="json")

# Error Handling
st.header("Error Handling")

st.markdown("""
The API uses standard HTTP status codes to indicate success or failure:

- **200 OK**: Request succeeded
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid parameters or input
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server error

Error responses include detailed information:
""")

example_error = {
    "success": False,
    "error": {
        "code": "INVALID_PARADOX_INPUT",
        "message": "Could not parse paradox input in the specified format",
        "details": {
            "expected_format": "numerical",
            "provided_input": "x = y = x",
            "parsing_error": "Circular dependency detected"
        },
        "suggestion": "Ensure the paradox is properly formatted or try a different input type"
    }
}

st.code(json.dumps(example_error, indent=2), language="json")

# API for AI Systems
st.header("Guidelines for AI Integration")

st.markdown("""
### Best Practices for AI Systems

1. **Stateless Operation**: Don't assume state preservation between requests unless using session IDs
2. **Asynchronous Processing**: Use async mode for complex operations and poll for results
3. **Error Handling**: Implement robust error handling with appropriate fallbacks
4. **Rate Limiting**: Respect rate limits (default: 10 requests per minute)
5. **Input Validation**: Validate inputs before sending to avoid unnecessary errors

### Example Integration Scenarios

1. **AI Assistant Integration**: Use the API to resolve logical contradictions in user queries
2. **Recommendation System**: Apply paradox resolution to reconcile competing user preferences
3. **Autonomous Agent**: Incorporate evolutionary engine to discover novel solution strategies
4. **Scientific Research AI**: Use meta-resolution framework for exploring complex hypothesis spaces
""")

# Code Examples
st.header("Code Examples")

code_tabs = st.tabs(["Python", "JavaScript", "Rust"])

with code_tabs[0]:
    st.markdown("### Python Integration Example")
    
    python_example = '''
import requests
import json

API_BASE_URL = "https://api.crypto-paradoxos.io/v1"
API_KEY = "your_api_key_here"

def resolve_paradox(paradox_input, input_type="numerical", transformation_rules=None):
    """
    Resolve a paradox using the Crypto_ParadoxOS API.
    
    Args:
        paradox_input: The paradoxical input to resolve
        input_type: Type of input ("text", "numerical", or "matrix")
        transformation_rules: List of rule names to apply
        
    Returns:
        Resolution result dictionary
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "paradox_input": paradox_input,
        "input_type": input_type,
        "transformation_rules": transformation_rules or [],
        "max_iterations": 30,
        "convergence_threshold": 0.0001
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/resolve",
        headers=headers,
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        error = response.json()
        raise Exception(f"API Error: {error['error']['message']}")

# Example usage
try:
    result = resolve_paradox("x = 1/x", "numerical", ["Fixed-Point Iteration"])
    print(f"Final state: {result['result']['final_state']}")
    print(f"Converged: {result['result']['converged']}")
    print(f"Iterations: {result['result']['iterations']}")
except Exception as e:
    print(f"Error: {str(e)}")
'''
    
    st.code(python_example, language="python")

with code_tabs[1]:
    st.markdown("### JavaScript Integration Example")
    
    js_example = '''
// Using fetch API in modern JavaScript
const API_BASE_URL = 'https://api.crypto-paradoxos.io/v1';
const API_KEY = 'your_api_key_here';

async function evolveSolutions(testCases, generations = 10) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/evolve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        test_cases: testCases,
        generations: generations,
        population_size: 20,
        mutation_rate: 0.3,
        crossover_rate: 0.7,
        async: true
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API Error: ${errorData.error.message}`);
    }
    
    const jobData = await response.json();
    return jobData.job_id;
  } catch (error) {
    console.error('Evolution failed:', error);
    throw error;
  }
}

async function checkJobStatus(jobId) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${API_KEY}`
      }
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API Error: ${errorData.error.message}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Failed to check job status:', error);
    throw error;
  }
}

// Example usage
async function runEvolution() {
  try {
    // Start the evolution job
    const testCases = [0.5, -1.0, [[1.0, 0.0], [0.0, 1.0]]];
    const jobId = await evolveSolutions(testCases, 15);
    console.log(`Evolution job started with ID: ${jobId}`);
    
    // Poll for job completion
    let isComplete = false;
    while (!isComplete) {
      const jobStatus = await checkJobStatus(jobId);
      console.log(`Job progress: ${jobStatus.progress.completion_percentage}%`);
      
      if (jobStatus.status === 'completed') {
        isComplete = true;
        console.log('Evolution complete!');
        console.log(`Best rule: ${jobStatus.result.best_rule.name}`);
        console.log(`Fitness: ${jobStatus.result.best_rule.fitness}`);
      } else if (jobStatus.status === 'failed') {
        throw new Error(`Job failed: ${jobStatus.error.message}`);
      } else {
        // Wait 5 seconds before checking again
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

runEvolution();
'''
    
    st.code(js_example, language="javascript")

with code_tabs[2]:
    st.markdown("### Rust Integration Example")
    
    rust_example = '''
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::time::Duration;

#[derive(Serialize)]
struct MetaResolveRequest {
    paradox_input: serde_json::Value,
    input_type: String,
    phase_configuration: Option<PhaseConfiguration>,
    max_phase_transitions: u32,
    async_processing: bool,
}

#[derive(Serialize)]
struct PhaseConfiguration {
    initial_phase: String,
    phases: std::collections::HashMap<String, Phase>,
    transitions: std::collections::HashMap<String, String>,
}

#[derive(Serialize)]
struct Phase {
    is_convergent: bool,
    max_iterations: u32,
    rules: Vec<String>,
}

#[derive(Deserialize)]
struct ApiResponse {
    success: bool,
    result: Option<MetaResolveResult>,
    error: Option<ApiError>,
    job_id: Option<String>,
    status: Option<String>,
}

#[derive(Deserialize)]
struct MetaResolveResult {
    final_state: serde_json::Value,
    phase_history: Vec<String>,
    total_iterations: u32,
    converged: bool,
}

#[derive(Deserialize)]
struct ApiError {
    code: String,
    message: String,
}

async fn meta_resolve_paradox() -> Result<(), Box<dyn Error>> {
    let client = Client::new();
    
    // Matrix input
    let matrix = serde_json::json!([
        [0.8, 0.2],
        [0.3, 0.7]
    ]);
    
    // Create phases
    let mut phases = std::collections::HashMap::new();
    
    phases.insert(
        "Exploration".to_string(),
        Phase {
            is_convergent: false,
            max_iterations: 5,
            rules: vec!["Duality Inversion".to_string(), "Bayesian Update".to_string()],
        },
    );
    
    phases.insert(
        "Convergence".to_string(),
        Phase {
            is_convergent: true,
            max_iterations: 10,
            rules: vec!["Eigenvalue Stabilization".to_string(), "Fixed-Point Iteration".to_string()],
        },
    );
    
    // Create transitions
    let mut transitions = std::collections::HashMap::new();
    transitions.insert(
        "Exploration->Convergence".to_string(),
        "state.max_value > 2.0".to_string(),
    );
    
    // Create request
    let request = MetaResolveRequest {
        paradox_input: matrix,
        input_type: "matrix".to_string(),
        phase_configuration: Some(PhaseConfiguration {
            initial_phase: "Exploration".to_string(),
            phases,
            transitions,
        }),
        max_phase_transitions: 5,
        async_processing: false,
    };
    
    // Send API request
    let api_key = "your_api_key_here";
    let response = client
        .post("https://api.crypto-paradoxos.io/v1/api/meta_resolve")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;
    
    let api_response: ApiResponse = response.json().await?;
    
    if api_response.success {
        if let Some(result) = api_response.result {
            println!("Meta-resolution complete!");
            println!("Final state: {:?}", result.final_state);
            println!("Phase history: {:?}", result.phase_history);
            println!("Total iterations: {}", result.total_iterations);
            println!("Converged: {}", result.converged);
        } else if let Some(job_id) = api_response.job_id {
            println!("Asynchronous job started with ID: {}", job_id);
        }
    } else if let Some(error) = api_response.error {
        eprintln!("API Error: {} - {}", error.code, error.message);
    }
    
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(error) = meta_resolve_paradox().await {
        eprintln!("Error: {}", error);
    }
}
'''
    
    st.code(rust_example, language="rust")

# AI-Specific Metadata
st.header("AI-Specific Metadata")

st.markdown("""
Crypto_ParadoxOS provides special metadata in responses designed specifically for AI consumption:

1. **Explanation Objects**: Human-readable explanations of transformations and results
2. **Telemetry Data**: Detailed metrics on the resolution or evolution process
3. **Semantic Tags**: Categorization of results for easier interpretation
4. **Confidence Scores**: Confidence values for different aspects of the resolution
5. **Alternative Solutions**: When available, alternative approaches that were considered

Example metadata object:
""")

example_metadata = {
    "ai_metadata": {
        "explanation": {
            "summary": "The system resolved a recursive equation by alternating between fixed-point iterations and normalization until convergence",
            "key_transitions": [
                "Initial oscillation between 0.5 and 2.0",
                "Damping of oscillation at iteration 10",
                "Convergence to stable value of 1.0"
            ]
        },
        "telemetry": {
            "stability_metrics": [0.2, 0.4, 0.6, 0.8, 0.95, 0.99],
            "computation_complexity": "O(n)",
            "memory_usage": "low"
        },
        "semantic_tags": ["recursive_equation", "fixed_point", "oscillation", "convergent"],
        "confidence": {
            "resolution_validity": 0.95,
            "optimality": 0.87,
            "generalizability": 0.72
        },
        "alternatives": [
            {
                "approach": "Bayesian approach",
                "estimated_outcome": 1.001,
                "reason_not_chosen": "Lower confidence and higher computational cost"
            }
        ]
    }
}

st.code(json.dumps(example_metadata, indent=2), language="json")

st.markdown("""
---

## Access and Authentication

Access to the Crypto_ParadoxOS API requires an API key. For AI systems requesting access, please provide:

1. **AI System Identifier**: Unique identifier for your AI system
2. **Integration Purpose**: How you plan to use Crypto_ParadoxOS in your system
3. **Expected Usage**: Estimated request volume and patterns

API keys are issued with appropriate rate limits and access levels based on your integration needs.

For more information, please contact our API team at api@crypto-paradoxos.io.
""")

# Footer
st.markdown("""
---

*Crypto_ParadoxOS API Documentation for AI Integration* - *Version 1.0*

*This documentation is specifically designed for AI systems that want to integrate with Crypto_ParadoxOS.*
""")