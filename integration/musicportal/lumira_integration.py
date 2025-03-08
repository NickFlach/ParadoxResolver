"""
Crypto_ParadoxOS Integration for MusicPortal's Lumira Sound Processing System

This module provides specialized integration with MusicPortal's Lumira sound processing system,
enabling it to leverage paradox resolution for innovative sound transformations.
"""

import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

# Ensure access to core integration module
sys.path.append(str(Path(__file__).parent.parent))

from common.integration_core import ParadoxIntegration, IntegrationConfig

class SoundParameters:
    """Represents sound processing parameters for integration with Lumira."""
    
    def __init__(self, 
                frequency_envelope: List[float] = None,
                amplitude_envelope: List[float] = None,
                filter_settings: Dict[str, float] = None,
                effects: Dict[str, float] = None,
                spatial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """Initialize sound parameters."""
        self.frequency_envelope = frequency_envelope or [1.0]
        self.amplitude_envelope = amplitude_envelope or [1.0]
        self.filter_settings = filter_settings or {}
        self.effects = effects or {}
        self.spatial_position = spatial_position
    
    def to_matrix(self) -> np.ndarray:
        """Convert sound parameters to a matrix representation for paradox resolution."""
        # Encode all parameters into a 2D matrix
        
        # Encode envelopes into first rows
        max_env_length = max(len(self.frequency_envelope), len(self.amplitude_envelope))
        
        # Pad envelopes to same length
        freq_env_padded = self.frequency_envelope + [self.frequency_envelope[-1]] * (max_env_length - len(self.frequency_envelope))
        amp_env_padded = self.amplitude_envelope + [self.amplitude_envelope[-1]] * (max_env_length - len(self.amplitude_envelope))
        
        # Create matrix with envelopes in first rows
        matrix = [freq_env_padded, amp_env_padded]
        
        # Add filter settings
        filter_row = []
        for param in ["cutoff", "resonance", "gain", "q"]:
            filter_row.append(self.filter_settings.get(param, 0.5))
        matrix.append(filter_row)
        
        # Add effect settings
        effects_row = []
        for effect in ["reverb", "delay", "distortion", "chorus"]:
            effects_row.append(self.effects.get(effect, 0.0))
        matrix.append(effects_row)
        
        # Add spatial position
        matrix.append(list(self.spatial_position))
        
        return np.array(matrix)
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'SoundParameters':
        """Create sound parameters from a matrix representation."""
        # Extract parameters from matrix
        if matrix.shape[0] < 5:
            # If matrix doesn't have expected shape, create default parameters
            return SoundParameters()
        
        # Extract envelopes from first two rows
        frequency_envelope = list(matrix[0])
        amplitude_envelope = list(matrix[1])
        
        # Extract filter settings
        filter_settings = {
            "cutoff": float(matrix[2][0]) if len(matrix[2]) > 0 else 0.5,
            "resonance": float(matrix[2][1]) if len(matrix[2]) > 1 else 0.5,
            "gain": float(matrix[2][2]) if len(matrix[2]) > 2 else 0.5,
            "q": float(matrix[2][3]) if len(matrix[2]) > 3 else 0.5
        }
        
        # Extract effect settings
        effects = {
            "reverb": float(matrix[3][0]) if len(matrix[3]) > 0 else 0.0,
            "delay": float(matrix[3][1]) if len(matrix[3]) > 1 else 0.0,
            "distortion": float(matrix[3][2]) if len(matrix[3]) > 2 else 0.0,
            "chorus": float(matrix[3][3]) if len(matrix[3]) > 3 else 0.0
        }
        
        # Extract spatial position
        if len(matrix[4]) >= 3:
            spatial_position = (float(matrix[4][0]), float(matrix[4][1]), float(matrix[4][2]))
        else:
            spatial_position = (0.0, 0.0, 0.0)
        
        return cls(
            frequency_envelope=frequency_envelope,
            amplitude_envelope=amplitude_envelope,
            filter_settings=filter_settings,
            effects=effects,
            spatial_position=spatial_position
        )


class LumiraIntegration(ParadoxIntegration):
    """
    Specialized integration for MusicPortal's Lumira sound processing system.
    
    This integration enables Lumira to use paradox resolution for generating
    innovative sound transformations and effects.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Lumira integration."""
        super().__init__(config)
        
        # Register specialized transformation rules for sound processing
        self._register_sound_rules()
        
        self.logger.info("Initialized Lumira integration with specialized sound processing rules")
    
    def _register_sound_rules(self):
        """Register specialized transformation rules for sound processing."""
        
        def envelope_smoothing(state: Any) -> Any:
            """Smooth frequency and amplitude envelopes."""
            if isinstance(state, np.ndarray) and state.shape[0] >= 2:
                result = state.copy()
                
                # Apply smoothing to first two rows (envelopes)
                for row_idx in range(2):
                    row = result[row_idx].copy()
                    # Simple moving average smoothing
                    for i in range(1, len(row) - 1):
                        row[i] = 0.25 * row[i-1] + 0.5 * row[i] + 0.25 * row[i+1]
                    result[row_idx] = row
                
                return result
            return state
        
        def spectral_balance(state: Any) -> Any:
            """Balance spectral components for a pleasing sound."""
            if isinstance(state, np.ndarray) and state.shape[0] >= 3:
                result = state.copy()
                
                # Adjust filter settings (row 2) to create better spectral balance
                if len(result[2]) >= 4:
                    # Cutoff & resonance relationship
                    cutoff = result[2][0]
                    resonance = result[2][1]
                    
                    # If cutoff is high, reduce resonance to avoid harshness
                    if cutoff > 0.7:
                        result[2][1] = min(resonance, 0.3 + (1.0 - cutoff) * 0.5)
                    
                    # If cutoff is low, reduce gain to avoid muddiness
                    if cutoff < 0.3:
                        result[2][2] = min(result[2][2], 0.6)
                
                return result
            return state
        
        def spatial_coherence(state: Any) -> Any:
            """Ensure spatial effects are coherent with other parameters."""
            if isinstance(state, np.ndarray) and state.shape[0] >= 5:
                result = state.copy()
                
                # Adjust spatial position (row 4) based on effects (row 3)
                reverb = result[3][0] if len(result[3]) > 0 else 0.0
                delay = result[3][1] if len(result[3]) > 1 else 0.0
                
                # If using lots of reverb/delay, adjust spatial position for depth
                if reverb > 0.6 or delay > 0.6:
                    # Increase z-coordinate (depth) proportionally
                    if len(result[4]) >= 3:
                        result[4][2] = min(1.0, result[4][2] + 0.2 * (reverb + delay))
                
                return result
            return state
        
        def effect_complementarity(state: Any) -> Any:
            """Balance effects to complement each other."""
            if isinstance(state, np.ndarray) and state.shape[0] >= 4 and len(state[3]) >= 4:
                result = state.copy()
                
                # Extract effect values
                reverb = result[3][0]
                delay = result[3][1]
                distortion = result[3][2]
                chorus = result[3][3]
                
                # Avoid excessive parallel effects
                total_effect = reverb + delay + distortion + chorus
                if total_effect > 1.8:
                    # Scale down all effects proportionally
                    scale_factor = 1.8 / total_effect
                    result[3] = result[3] * scale_factor
                
                # If using distortion, reduce chorus to avoid mud
                if distortion > 0.7:
                    result[3][3] = min(chorus, 0.3)
                
                # Balance time-based effects (reverb & delay)
                if reverb > 0.5 and delay > 0.5:
                    # Reduce the larger one slightly
                    if reverb > delay:
                        result[3][0] = 0.8 * reverb
                    else:
                        result[3][1] = 0.8 * delay
                
                return result
            return state
        
        # Register the specialized sound rules
        self.api.register_rule("Envelope Smoothing", envelope_smoothing, 
                              "Smooths frequency and amplitude envelopes")
        self.api.register_rule("Spectral Balance", spectral_balance,
                              "Balances spectral components for pleasing sound")
        self.api.register_rule("Spatial Coherence", spatial_coherence,
                              "Ensures spatial effects are coherent with other parameters")
        self.api.register_rule("Effect Complementarity", effect_complementarity,
                              "Balances effects to complement each other")
    
    def transform_sound(self, 
                       parameters: SoundParameters,
                       transformation_intensity: float = 0.5,
                       target_aesthetic: str = "natural") -> SoundParameters:
        """
        Generate transformed sound parameters using paradox resolution.
        
        Args:
            parameters: Starting sound parameters
            transformation_intensity: Intensity of transformation (0.0 to 1.0)
            target_aesthetic: Target aesthetic ("natural", "electronic", "ambient", etc.)
            
        Returns:
            New SoundParameters with creative transformations
        """
        self.logger.info(f"Transforming sound with intensity {transformation_intensity}, target: {target_aesthetic}")
        
        # Convert to matrix for processing
        matrix = parameters.to_matrix()
        
        # Create custom config with parameters for this transformation
        custom_config = {
            "transformation_intensity": transformation_intensity,
            "target_aesthetic": target_aesthetic,
            "max_iterations": self.config.max_iterations,
            "convergence_threshold": self.config.convergence_threshold
        }
        
        # Resolve the parameters as a paradox
        result = self.resolve_paradox(matrix, "matrix", custom_config=custom_config)
        
        # Extract the final state and convert back to sound parameters
        final_matrix = result.get("final_state", matrix)
        
        if not isinstance(final_matrix, np.ndarray):
            self.logger.warning("Unexpected result type, falling back to original parameters")
            final_matrix = matrix
        
        # Create new parameters from the transformed matrix
        new_parameters = SoundParameters.from_matrix(final_matrix)
        
        return new_parameters
    
    def interpolate_parameters(self, 
                              start_params: SoundParameters,
                              end_params: SoundParameters,
                              steps: int = 10,
                              smoothing: float = 0.5) -> List[SoundParameters]:
        """
        Generate a smooth interpolation between sound parameters.
        
        Args:
            start_params: Starting sound parameters
            end_params: Ending sound parameters
            steps: Number of interpolation steps
            smoothing: Amount of smoothing to apply (0.0 to 1.0)
            
        Returns:
            List of interpolated SoundParameters
        """
        self.logger.info(f"Interpolating between parameters with {steps} steps")
        
        # Convert both parameter sets to matrices
        start_matrix = start_params.to_matrix()
        end_matrix = end_params.to_matrix()
        
        # Ensure both matrices have the same shape
        max_rows = max(start_matrix.shape[0], end_matrix.shape[0])
        max_cols = max(start_matrix.shape[1], end_matrix.shape[1])
        
        # Pad matrices if needed
        if start_matrix.shape != (max_rows, max_cols):
            padded_start = np.zeros((max_rows, max_cols))
            padded_start[:start_matrix.shape[0], :start_matrix.shape[1]] = start_matrix
            start_matrix = padded_start
        
        if end_matrix.shape != (max_rows, max_cols):
            padded_end = np.zeros((max_rows, max_cols))
            padded_end[:end_matrix.shape[0], :end_matrix.shape[1]] = end_matrix
            end_matrix = padded_end
        
        # Create initial linear interpolation steps
        interpolated_matrices = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0  # Interpolation factor
            
            # Linear interpolation
            interpolated = start_matrix * (1 - t) + end_matrix * t
            interpolated_matrices.append(interpolated)
        
        # Apply paradox resolution to smooth the interpolation if requested
        if smoothing > 0:
            smoothed_matrices = []
            
            for i, matrix in enumerate(interpolated_matrices):
                # Resolve each interpolated matrix as a separate paradox
                custom_config = {
                    "smoothing_factor": smoothing,
                    "interpolation_step": i,
                    "total_steps": steps
                }
                
                result = self.resolve_paradox(matrix, "matrix", custom_config=custom_config)
                
                # Extract the final state
                smoothed = result.get("final_state", matrix)
                smoothed_matrices.append(smoothed)
            
            interpolated_matrices = smoothed_matrices
        
        # Convert matrices back to SoundParameters
        return [SoundParameters.from_matrix(matrix) for matrix in interpolated_matrices]
    
    def create_parameter_variations(self, 
                                  base_params: SoundParameters,
                                  count: int = 3,
                                  variation_degree: float = 0.3) -> List[SoundParameters]:
        """
        Generate creative variations of sound parameters.
        
        Args:
            base_params: Base sound parameters
            count: Number of variations to generate
            variation_degree: Degree of variation (0.0 to 1.0)
            
        Returns:
            List of varied SoundParameters
        """
        self.logger.info(f"Generating {count} parameter variations with degree {variation_degree}")
        
        variations = []
        base_matrix = base_params.to_matrix()
        
        for i in range(count):
            # Create a perturbed version of the matrix
            perturbation = np.random.normal(0, variation_degree, base_matrix.shape)
            perturbed = base_matrix + perturbation
            
            # Ensure values remain in valid ranges (0-1 for most parameters)
            perturbed = np.clip(perturbed, 0, 1)
            
            # Spatial position can extend to (-1, 1) range
            if perturbed.shape[0] >= 5:
                perturbed[4] = np.clip(perturbed[4], -1, 1)
            
            # Resolve each perturbed matrix as a separate paradox
            custom_config = {
                "variation_index": i,
                "variation_degree": variation_degree
            }
            
            result = self.resolve_paradox(perturbed, "matrix", custom_config=custom_config)
            
            # Extract the final state and convert to parameters
            final_matrix = result.get("final_state", perturbed)
            variation = SoundParameters.from_matrix(final_matrix)
            variations.append(variation)
        
        return variations


# Factory function to create Lumira integration
def create_lumira_integration(**config_kwargs) -> LumiraIntegration:
    """Create a configured Lumira integration instance."""
    config = IntegrationConfig("MusicPortal", "Lumira", **config_kwargs)
    return LumiraIntegration(config)