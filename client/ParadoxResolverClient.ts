/**
 * ParadoxResolver TypeScript Client
 * 
 * Universal client for all Node.js/TypeScript projects to interact
 * with the ParadoxResolver service.
 */

export interface ParadoxResolverConfig {
  serviceUrl?: string;
  timeout?: number;
}

export interface ResolutionRequest {
  initial_state: any;
  input_type?: 'numerical' | 'matrix' | 'logical' | 'text';
  max_iterations?: number;
  convergence_threshold?: number;
  rules?: string[];
}

export interface ResolutionResult {
  success: boolean;
  final_state: any;
  converged: boolean;
  iterations: number;
  states_history: any[];
  input_type: string;
  error?: string;
}

export interface MetaResolutionRequest {
  initial_state: any;
  input_type?: string;
  max_phase_transitions?: number;
  max_total_iterations?: number;
}

export interface PhaseResult {
  phase: string;
  iterations: number;
  is_convergent: boolean;
  type: string;
}

export interface MetaResolutionResult {
  success: boolean;
  final_state: any;
  converged: boolean;
  total_iterations: number;
  phase_transitions: number;
  phase_history: string[];
  phase_results: PhaseResult[];
  input_type: string;
  error?: string;
}

export interface Resource {
  name: string;
  total: number;
}

export interface Stakeholder {
  name: string;
  influence: number;
  preferences: Record<string, number>;
}

export interface OptimizationRequest {
  resources: Resource[];
  stakeholders: Stakeholder[];
}

export interface OptimizationResult {
  success: boolean;
  allocation: Record<string, Record<string, number>>;
  total_satisfaction: number;
  fairness_score: number;
  iterations: number;
  converged: boolean;
  stakeholder_satisfaction: Record<string, number>;
  resource_utilization: Record<string, number>;
  error?: string;
}

export interface EvolutionRequest {
  test_cases: any[];
  generations?: number;
  population_size?: number;
  mutation_rate?: number;
}

export interface EvolvedRule {
  name: string;
  fitness: number;
  complexity: number;
  components: string[];
}

export interface EvolutionResult {
  success: boolean;
  best_rules: EvolvedRule[];
  best_fitness: number;
  avg_fitness: number;
  diversity: number;
  generations: number;
  population_size: number;
  history: {
    max_fitness: number[];
    avg_fitness: number[];
    diversity: number[];
    novelty: number[];
  };
  error?: string;
}

export interface RuleInfo {
  name: string;
  description: string;
  available: boolean;
}

export interface RulesResult {
  success: boolean;
  rules: Record<string, RuleInfo>;
  count: number;
  error?: string;
}

export class ParadoxResolverClient {
  private serviceUrl: string;
  private timeout: number;

  constructor(config: ParadoxResolverConfig = {}) {
    this.serviceUrl = config.serviceUrl || 'http://localhost:3333';
    this.timeout = config.timeout || 30000;
  }

  private async request<T>(endpoint: string, method: string = 'GET', body?: any): Promise<T> {
    const url = `${this.serviceUrl}${endpoint}`;
    const options: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    if (body) {
      options.body = JSON.stringify(body);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error(`Request timeout after ${this.timeout}ms`);
        }
        throw error;
      }
      throw new Error('Unknown error occurred');
    }
  }

  /**
   * Resolve a paradox using standard transformation rules
   */
  async resolve(request: ResolutionRequest): Promise<ResolutionResult> {
    return this.request<ResolutionResult>('/api/resolve', 'POST', request);
  }

  /**
   * Resolve using meta-framework with phase transitions
   */
  async metaResolve(request: MetaResolutionRequest): Promise<MetaResolutionResult> {
    return this.request<MetaResolutionResult>('/api/meta-resolve', 'POST', request);
  }

  /**
   * Optimize resource allocation among stakeholders
   */
  async optimize(request: OptimizationRequest): Promise<OptimizationResult> {
    return this.request<OptimizationResult>('/api/optimize', 'POST', request);
  }

  /**
   * Evolve novel transformation rules using genetic algorithms
   */
  async evolve(request: EvolutionRequest): Promise<EvolutionResult> {
    return this.request<EvolutionResult>('/api/evolve', 'POST', request);
  }

  /**
   * Get list of available transformation rules
   */
  async getRules(): Promise<RulesResult> {
    return this.request<RulesResult>('/api/rules', 'GET');
  }

  /**
   * Check service health
   */
  async healthCheck(): Promise<{ status: string; service: string; version: string }> {
    return this.request('/health', 'GET');
  }
}

// Export convenience functions
export const createParadoxResolverClient = (config?: ParadoxResolverConfig) => {
  return new ParadoxResolverClient(config);
};

export default ParadoxResolverClient;
