"""
Crypto_ParadoxOS Integration Core

This module provides the core functionality for integrating Crypto_ParadoxOS with
external applications and systems. It defines the base integration classes and
common utilities.
"""

import logging
import json
import os
import sys
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np
from pathlib import Path

# Ensure access to root module
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import core modules (with error handling)
try:
    from crypto_paradox_api import CryptoParadoxAPI, ParadoxResult
    from meta_resolver import MetaResolver
    from evolutionary_engine import EvolutionaryEngine
    
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    # Create placeholder classes if core modules not available
    # This allows the integration framework to be developed independently
    
    class CryptoParadoxAPI:
        """Placeholder API class."""
        def __init__(self): pass
        def register_rule(self, name, fn, desc=""): pass
        def resolve_paradox(self, *args, **kwargs): return {}
    
    class MetaResolver:
        """Placeholder MetaResolver class."""
        def __init__(self): pass
        def resolve(self, *args, **kwargs): return {}
    
    class EvolutionaryEngine:
        """Placeholder EvolutionaryEngine class."""
        def __init__(self): pass
        def evolve(self, *args, **kwargs): return {}


class IntegrationConfig:
    """Configuration settings for Crypto_ParadoxOS integration."""
    
    def __init__(
        self,
        application_name: str,
        module_name: str,
        use_evolution: bool = False,
        use_meta_resolution: bool = True,
        custom_rules: Dict[str, Callable] = None,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
        log_level: str = "INFO"
    ):
        """
        Initialize integration configuration.
        
        Args:
            application_name: Name of the target application
            module_name: Specific module within the application to integrate with
            use_evolution: Whether to use the evolutionary engine
            use_meta_resolution: Whether to use the meta-resolver
            custom_rules: Dictionary of custom transformation rules
            max_iterations: Maximum iterations for resolution processes
            convergence_threshold: Convergence threshold for resolution
            log_level: Logging level for the integration
        """
        self.application_name = application_name
        self.module_name = module_name
        self.use_evolution = use_evolution
        self.use_meta_resolution = use_meta_resolution
        self.custom_rules = custom_rules or {}
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.log_level = log_level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = self.__dict__.copy()
        
        # Remove callable objects (can't be serialized)
        if 'custom_rules' in config_dict:
            config_dict['custom_rules'] = list(config_dict['custom_rules'].keys())
            
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IntegrationConfig':
        """Create configuration from dictionary."""
        # Create copy to avoid modifying the original
        config = config_dict.copy()
        
        # Handle custom_rules
        if 'custom_rules' in config and isinstance(config['custom_rules'], list):
            config['custom_rules'] = {rule: None for rule in config['custom_rules']}
            
        return cls(**config)
    
    def save_to_file(self, filename: str) -> None:
        """Save configuration to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'IntegrationConfig':
        """Load configuration from JSON file."""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class ParadoxIntegration:
    """
    Base class for Crypto_ParadoxOS integration with external applications.
    
    This class provides the core functionality for integrating the paradox resolution
    capabilities with other systems.
    """
    
    def __init__(self, config: IntegrationConfig):
        """
        Initialize the integration.
        
        Args:
            config: Integration configuration
        """
        self.config = config
        
        # Set up logging
        self.logger = logging.getLogger(f"{config.application_name}.{config.module_name}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"Initializing {config.application_name}.{config.module_name} integration")
        
        # Check if core modules are available
        if not CORE_MODULES_AVAILABLE:
            self.logger.warning("Core Crypto_ParadoxOS modules not available. Using placeholder implementation.")
        
        # Initialize components
        self.api = CryptoParadoxAPI()
        self.meta_resolver = None
        self.evolutionary_engine = None
        
        # Set up meta-resolver if enabled
        if config.use_meta_resolution:
            self.meta_resolver = MetaResolver(self.api)
            self.meta_resolver.create_standard_framework()
            self.logger.info("Meta-resolver initialized with standard framework")
        
        # Set up evolutionary engine if enabled
        if config.use_evolution:
            self.evolutionary_engine = EvolutionaryEngine()
            self.logger.info("Evolutionary engine initialized")
        
        # Register custom rules
        self._register_custom_rules()
        
        self.logger.info(f"Integration initialized successfully")
    
    def _register_custom_rules(self) -> None:
        """Register custom transformation rules from the configuration."""
        if self.config.custom_rules:
            for name, rule_fn in self.config.custom_rules.items():
                if callable(rule_fn):
                    self.api.register_rule(name, rule_fn)
                    self.logger.info(f"Registered custom rule: {name}")
    
    def resolve_paradox(self, 
                       input_data: Any, 
                       input_type: str = "unknown",
                       use_meta: bool = None,
                       custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Resolve a paradox.
        
        Args:
            input_data: The paradoxical input to resolve
            input_type: Type of the input data
            use_meta: Whether to use meta-resolution (overrides config)
            custom_config: Custom configuration for this resolution
            
        Returns:
            Dictionary with resolution results
        """
        self.logger.info(f"Resolving paradox of type '{input_type}'")
        
        # Determine whether to use meta-resolution
        use_meta_resolution = use_meta if use_meta is not None else self.config.use_meta_resolution
        
        # Create configuration for resolution
        resolution_config = {
            "max_iterations": self.config.max_iterations,
            "convergence_threshold": self.config.convergence_threshold,
            "integration_context": {
                "application": self.config.application_name,
                "module": self.config.module_name
            }
        }
        
        # Add custom configuration if provided
        if custom_config:
            resolution_config.update(custom_config)
        
        try:
            # Use meta-resolver if enabled
            if use_meta_resolution and self.meta_resolver:
                self.logger.info("Using meta-resolver for complex resolution")
                result = self.meta_resolver.resolve(
                    initial_state=input_data,
                    input_type=input_type,
                    max_phase_transitions=10,
                    max_total_iterations=resolution_config["max_iterations"]
                )
            else:
                # Use standard resolution
                self.logger.info("Using standard resolver")
                result = self.api.resolve_paradox(
                    paradox_input=input_data,
                    input_type=input_type,
                    config=resolution_config
                )
            
            self.logger.info("Paradox resolution completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during paradox resolution: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "input_type": input_type,
                "final_state": input_data  # Return original data unchanged
            }
    
    def evolve_rules(self, 
                    test_cases: List[Any],
                    generations: int = 10,
                    custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evolve transformation rules.
        
        Args:
            test_cases: List of test cases to evaluate rule fitness
            generations: Number of generations to evolve
            custom_config: Custom configuration for evolution
            
        Returns:
            Dictionary with evolution results
        """
        if not self.evolutionary_engine:
            self.logger.warning("Evolutionary engine not enabled. Enable with use_evolution=True")
            return {"status": "failed", "error": "Evolutionary engine not enabled"}
        
        self.logger.info(f"Evolving rules using {len(test_cases)} test cases")
        
        # Configure evolution
        evolution_config = {
            "generations": generations,
            "integration_context": {
                "application": self.config.application_name,
                "module": self.config.module_name
            }
        }
        
        # Add custom configuration if provided
        if custom_config:
            evolution_config.update(custom_config)
            
        try:
            # Evolve rules
            result = self.evolutionary_engine.evolve(
                test_cases=test_cases,
                generations=generations
            )
            
            # Register best evolved rules
            best_rules = result.get("best_rules", [])
            for name, rule_fn in best_rules:
                self.api.register_rule(name, rule_fn, f"Evolved rule for {self.config.module_name}")
                self.logger.info(f"Registered evolved rule: {name}")
            
            self.logger.info(f"Rule evolution completed successfully with {len(best_rules)} new rules")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during rule evolution: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the current status of the integration.
        
        Returns:
            Dictionary with status information
        """
        return {
            "application": self.config.application_name,
            "module": self.config.module_name,
            "meta_resolution_enabled": self.meta_resolver is not None,
            "evolution_enabled": self.evolutionary_engine is not None,
            "core_modules_available": CORE_MODULES_AVAILABLE,
            "max_iterations": self.config.max_iterations,
            "convergence_threshold": self.config.convergence_threshold
        }


# Factory function for creating integrations
def create_integration(application_name: str, module_name: str, **config_kwargs) -> ParadoxIntegration:
    """
    Create a basic integration for the specified application and module.
    
    Args:
        application_name: Name of the application
        module_name: Name of the module
        **config_kwargs: Additional configuration parameters
        
    Returns:
        ParadoxIntegration instance
    """
    config = IntegrationConfig(application_name, module_name, **config_kwargs)
    return ParadoxIntegration(config)