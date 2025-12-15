"""
Chromatic quantization utilities for pitch frequency conversion.

This module provides functions to convert pitch frequencies to MIDI values
and quantize them to chromatic bins for HMM training and inference.
"""

import numpy as np
from typing import Union
from ..exceptions import QuantizationError


def hz_to_midi(f_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert frequency in Hz to MIDI note number using logarithmic mapping.
    
    Uses the standard formula: MIDI = 12 * log2(f_hz / 440.0) + 69
    where A4 (440 Hz) corresponds to MIDI note 69.
    
    Args:
        f_hz: Frequency in Hz (scalar or array)
        
    Returns:
        MIDI note number(s) as float (can be fractional)
        
    Raises:
        QuantizationError: If frequency is non-positive
        
    Examples:
        >>> hz_to_midi(440.0)  # A4
        69.0
        >>> hz_to_midi(261.63)  # C4
        60.0
    """
    f_hz = np.asarray(f_hz)
    
    # Check for non-positive frequencies
    if np.any(f_hz <= 0):
        raise QuantizationError("Frequency must be positive")
    
    # Standard MIDI conversion formula
    midi = 12.0 * np.log2(f_hz / 440.0) + 69.0
    
    # Return scalar if input was scalar
    if midi.ndim == 0:
        return float(midi)
    return midi


def nearest_chromatic_bin(midi: Union[float, np.ndarray], 
                         base_midi: float = 48) -> Union[int, np.ndarray]:
    """
    Quantize MIDI note numbers to nearest chromatic bin indices.
    
    Maps MIDI values to integer bin indices in the range [0, 35] representing
    chromatic pitches from C3 (MIDI 48) to B5 (MIDI 83).
    
    Args:
        midi: MIDI note number(s) (can be fractional)
        base_midi: Base MIDI note for bin 0 (default: 48 = C3)
        
    Returns:
        Chromatic bin indices [0, 35] (int or array of ints)
        Values outside range are clamped to boundaries
        
    Examples:
        >>> nearest_chromatic_bin(48.0)  # C3
        0
        >>> nearest_chromatic_bin(60.0)  # C4
        12
        >>> nearest_chromatic_bin(83.0)  # B5
        35
    """
    midi = np.asarray(midi)
    
    # Convert to bin indices relative to base_midi
    bin_indices = np.round(midi - base_midi).astype(int)
    
    # Clamp to valid range [0, 35]
    bin_indices = np.clip(bin_indices, 0, 35)
    
    # Return scalar if input was scalar
    if bin_indices.ndim == 0:
        return int(bin_indices)
    return bin_indices


class ChromaticQuantizer:
    """
    Chromatic quantizer for converting frequencies to discrete bins.
    
    Provides a stateful interface for frequency-to-bin conversion with
    configurable parameters and boundary handling.
    """
    
    def __init__(self, n_bins: int = 36, base_midi: float = 48):
        """
        Initialize chromatic quantizer.
        
        Args:
            n_bins: Number of chromatic bins (default: 36 for C3-B5)
            base_midi: MIDI note for bin 0 (default: 48 = C3)
        """
        if n_bins <= 0:
            raise QuantizationError("Number of bins must be positive")
        if base_midi < 0:
            raise QuantizationError("Base MIDI note must be non-negative")
            
        self.n_bins = n_bins
        self.base_midi = base_midi
        self.max_midi = base_midi + n_bins - 1
        
    def quantize(self, f_hz: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Quantize frequencies to chromatic bin indices.
        
        Args:
            f_hz: Frequency in Hz (scalar or array)
            
        Returns:
            Chromatic bin indices [0, n_bins-1]
            
        Raises:
            QuantizationError: If frequency is non-positive
        """
        # Convert to MIDI
        midi = hz_to_midi(f_hz)
        
        # Quantize to bins
        return nearest_chromatic_bin(midi, self.base_midi)
    
    def get_bin_frequency(self, bin_idx: int) -> float:
        """
        Get the center frequency for a given bin index.
        
        Args:
            bin_idx: Bin index [0, n_bins-1]
            
        Returns:
            Center frequency in Hz
            
        Raises:
            QuantizationError: If bin index is out of range
        """
        if not (0 <= bin_idx < self.n_bins):
            raise QuantizationError(f"Bin index {bin_idx} out of range [0, {self.n_bins-1}]")
        
        midi = self.base_midi + bin_idx
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    
    def get_bin_boundaries(self) -> np.ndarray:
        """
        Get frequency boundaries for all bins.
        
        Returns:
            Array of boundary frequencies [n_bins + 1]
        """
        midi_boundaries = np.arange(self.base_midi - 0.5, 
                                   self.base_midi + self.n_bins + 0.5)
        return 440.0 * (2.0 ** ((midi_boundaries - 69.0) / 12.0))