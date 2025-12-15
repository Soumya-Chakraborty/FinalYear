"""
Tonic normalization utilities for pitch frequency adjustment.

This module provides functions to normalize pitch frequencies by shifting
the tonic to a reference frequency (C4 = 261.63 Hz) for consistent analysis.
"""

import numpy as np
from typing import Union
from ..exceptions import QuantizationError


def normalize_by_tonic(f0_hz: Union[float, np.ndarray], 
                      tonic_hz: float) -> Union[float, np.ndarray]:
    """
    Normalize pitch frequencies by shifting tonic to C4 (261.63 Hz).
    
    This function shifts all frequencies proportionally so that the tonic
    frequency becomes C4 (261.63 Hz), maintaining the relative pitch
    relationships while standardizing the tonal center.
    
    Args:
        f0_hz: Pitch frequencies in Hz (scalar or array)
        tonic_hz: Tonic frequency in Hz to normalize to C4
        
    Returns:
        Normalized frequencies with tonic shifted to C4
        
    Raises:
        QuantizationError: If tonic frequency is invalid
        
    Examples:
        >>> normalize_by_tonic(440.0, 220.0)  # A4 with tonic at A3
        523.26  # A5 (octave higher due to normalization)
        >>> normalize_by_tonic(220.0, 220.0)  # Tonic itself
        261.63  # C4
    """
    if tonic_hz <= 0:
        raise QuantizationError("Tonic frequency must be positive")
    
    # Validate tonic frequency range (reasonable musical range)
    if not (80.0 <= tonic_hz <= 800.0):
        raise QuantizationError(f"Tonic frequency {tonic_hz:.2f} Hz outside valid range [80, 800] Hz")
    
    f0_hz = np.asarray(f0_hz)
    
    # Check for non-positive frequencies
    if np.any(f0_hz <= 0):
        raise QuantizationError("All frequencies must be positive")
    
    # Reference C4 frequency
    C4_HZ = 261.63
    
    # Normalize by scaling all frequencies
    # f_normalized = f_original * (C4 / tonic)
    normalized = f0_hz * (C4_HZ / tonic_hz)
    
    # Return scalar if input was scalar
    if normalized.ndim == 0:
        return float(normalized)
    return normalized


class TonicNormalizer:
    """
    Tonic normalizer for consistent pitch frequency adjustment.
    
    Provides a stateful interface for tonic normalization with
    validation and caching of normalization parameters.
    """
    
    def __init__(self, reference_hz: float = 261.63):
        """
        Initialize tonic normalizer.
        
        Args:
            reference_hz: Reference frequency for normalization (default: C4 = 261.63 Hz)
        """
        if reference_hz <= 0:
            raise QuantizationError("Reference frequency must be positive")
        
        self.reference_hz = reference_hz
        self._cached_tonic = None
        self._cached_scale_factor = None
    
    def normalize(self, f0_hz: Union[float, np.ndarray], 
                 tonic_hz: float) -> Union[float, np.ndarray]:
        """
        Normalize frequencies using the configured reference.
        
        Args:
            f0_hz: Pitch frequencies in Hz (scalar or array)
            tonic_hz: Tonic frequency in Hz
            
        Returns:
            Normalized frequencies
            
        Raises:
            QuantizationError: If tonic frequency is invalid
        """
        # Cache scale factor for repeated calls with same tonic
        if self._cached_tonic != tonic_hz:
            if tonic_hz <= 0:
                raise QuantizationError("Tonic frequency must be positive")
            if not (80.0 <= tonic_hz <= 800.0):
                raise QuantizationError(f"Tonic frequency {tonic_hz:.2f} Hz outside valid range [80, 800] Hz")
            
            self._cached_tonic = tonic_hz
            self._cached_scale_factor = self.reference_hz / tonic_hz
        
        f0_hz = np.asarray(f0_hz)
        
        # Check for non-positive frequencies
        if np.any(f0_hz <= 0):
            raise QuantizationError("All frequencies must be positive")
        
        # Apply cached scale factor
        normalized = f0_hz * self._cached_scale_factor
        
        # Return scalar if input was scalar
        if normalized.ndim == 0:
            return float(normalized)
        return normalized
    
    def get_scale_factor(self, tonic_hz: float) -> float:
        """
        Get the scale factor for a given tonic frequency.
        
        Args:
            tonic_hz: Tonic frequency in Hz
            
        Returns:
            Scale factor for normalization
            
        Raises:
            QuantizationError: If tonic frequency is invalid
        """
        if tonic_hz <= 0:
            raise QuantizationError("Tonic frequency must be positive")
        if not (80.0 <= tonic_hz <= 800.0):
            raise QuantizationError(f"Tonic frequency {tonic_hz:.2f} Hz outside valid range [80, 800] Hz")
        
        return self.reference_hz / tonic_hz
    
    def denormalize(self, normalized_hz: Union[float, np.ndarray], 
                   tonic_hz: float) -> Union[float, np.ndarray]:
        """
        Convert normalized frequencies back to original tonic.
        
        Args:
            normalized_hz: Normalized frequencies in Hz
            tonic_hz: Original tonic frequency in Hz
            
        Returns:
            Frequencies in original tonic
        """
        if tonic_hz <= 0:
            raise QuantizationError("Tonic frequency must be positive")
        
        normalized_hz = np.asarray(normalized_hz)
        
        # Reverse the normalization
        original = normalized_hz * (tonic_hz / self.reference_hz)
        
        # Return scalar if input was scalar
        if original.ndim == 0:
            return float(original)
        return original