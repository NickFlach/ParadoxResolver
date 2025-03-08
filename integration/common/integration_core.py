"""
Crypto_ParadoxOS Integration Core

This module provides the core functionality for integrating Crypto_ParadoxOS with
external applications and systems. It defines the base integration classes and
common utilities.
"""

import sys
import logging
from typing import Any, Dict, List, Optional, Type, Union, Callable
import importlib
import json

# Ensure access to core Crypto_ParadoxOS modules
sys.path.append("/home/runner/workspace")

# Core Crypto_ParadoxOS imports
from meta_resolver import MetaResolver
from evolutionary_engine import EvolutionaryEngine
from crypto_paradox_api import CryptoParadoxAPI
from transformation_rules import get_available_rules

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
        
        # Set up logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        
        self.logger = logging.getLogger(f"paradox_integration.{application_name}.{module_name}")
        self.logger.setLevel(numeric_level)
        
        # Create console handler if not already set up
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "application_name": self.application_name,
            "module_name": self.module_name,
            "use_evolution": self.use_evolution,
            "use_meta_resolution": self.use_meta_resolution,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "log_level": self.log_level,
            "custom_rules_count": len(self.custom_rules)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IntegrationConfig':
        """Create configuration from dictionary."""
        # Filter out custom_rules_count which is just informational
        if "custom_rules_count" in config_dict:
            del config_dict["custom_rules_count"]
        
        return cls(**config_dict)
    
    def save_to_file(self, filename: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        # Can't serialize custom_rules functions, so remove them
        if "custom_rules" in config_dict:
            del config_dict["custom_rules"]
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
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
        self.logger = config.logger
        
        # Initialize API
        self.api = CryptoParadoxAPI()
        
        # Add custom rules if provided
        if config.custom_rules:
            for name, rule_fn in config.custom_rules.items():
                self.api.register_rule(name, rule_fn)
        
        # Initialize MetaResolver if requested
        self.meta_resolver = None
        if config.use_meta_resolution:
            self.meta_resolver = MetaResolver(self.api)
            self.meta_resolver = self.meta_resolver.create_standard_framework()
        
        # Initialize EvolutionaryEngine if requested
        self.evolutionary_engine = None
        if config.use_evolution:
            standard_rules = get_available_rules()
            self.evolutionary_engine = EvolutionaryEngine(seed_rules=standard_rules)
        
        self.logger.info(f"Initialized ParadoxIntegration for {config.application_name}/{config.module_name}")

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
        use_meta_resolver = use_meta if use_meta is not None else self.config.use_meta_resolution
        
        try:
            if use_meta_resolver and self.meta_resolver:
                self.logger.info(f"Using meta-resolver for {input_type} input")
                result = self.meta_resolver.resolve(
                    input_data, 
                    input_type,
                    max_total_iterations=self.config.max_iterations
                )
                return result
            else:
                self.logger.info(f"Using direct API for {input_type} input")
                result = self.api.resolve_paradox(
                    input_data,
                    input_type,
                    config=custom_config or {
                        "max_iterations": self.config.max_iterations,
                        "convergence_threshold": self.config.convergence_threshold
                    }
                )
                return result
        except Exception as e:
            self.logger.error(f"Error resolving paradox: {str(e)}")
            raise

    def evolve_rules(self, 
                    test_cases: List[Any],
                    generations: int = 5) -> Dict[str, Any]:
        """
        Evolve new transformation rules.
        
        Args:
            test_cases: Test cases to use for fitness evaluation
            generations: Number of generations to evolve
            
        Returns:
            Dictionary with evolution results
        """
        if not self.evolutionary_engine:
            self.logger.error("Cannot evolve rules: evolutionary engine not initialized")
            raise ValueError("Evolutionary engine not initialized")
        
        try:
            self.logger.info(f"Evolving rules for {generations} generations")
            result = self.evolutionary_engine.evolve(test_cases, generations)
            
            # Register the best evolved rules with the API
            best_rules = self.evolutionary_engine.get_best_rules(3)
            for name, rule_fn in best_rules:
                self.api.register_rule(name, rule_fn, "Evolved rule")
            
            return result
        except Exception as e:
            self.logger.error(f"Error evolving rules: {str(e)}")
            raise

    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the current status of the integration.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "config": self.config.to_dict(),
            "available_rules": len(self.api.get_available_rules()),
            "meta_resolver_active": self.meta_resolver is not None,
            "evolutionary_engine_active": self.evolutionary_engine is not None,
        }
        
        # Add evolutionary engine info if available
        if self.evolutionary_engine:
            status["population_diversity"] = self.evolutionary_engine._calculate_diversity()
            status["generation_count"] = getattr(self.evolutionary_engine, "current_generation", 0)
        
        return status


def create_integration(application: str, module: str, **config_kwargs) -> ParadoxIntegration:
    """
    Factory function to create appropriate integration for the given application.
    
    Args:
        application: Target application name
        module: Specific module within the application
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured ParadoxIntegration instance
    """
    config = IntegrationConfig(application, module, **config_kwargs)
    
    # Try to import specialized integration if available
    try:
        module_path = f"integration.{application.lower()}.{module.lower()}_integration"
        integration_module = importlib.import_module(module_path)
        
        # Look for specialized integration class
        for attr_name in dir(integration_module):
            if attr_name.endswith("Integration") and attr_name != "ParadoxIntegration":
                integration_class = getattr(integration_module, attr_name)
                if issubclass(integration_class, ParadoxIntegration):
                    config.logger.info(f"Using specialized integration: {attr_name}")
                    return integration_class(config)
    except (ImportError, AttributeError) as e:
        config.logger.warning(f"Could not load specialized integration: {str(e)}")
    
    # Fall back to base integration
    return ParadoxIntegration(config)