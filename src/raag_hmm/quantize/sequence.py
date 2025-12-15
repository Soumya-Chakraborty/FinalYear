"""
Sequence quantization utilities combining normalization and binning.

This module provides functions to process pitch sequences through
tonic normalization and chromatic quantization in a single pipeline.
"""

import numpy as np
from typing import Union
from .tonic import normalize_by_tonic
from .chromatic import ChromaticQuantizer
from ..exceptions import QuantizationError


def quantize_sequence(f0_hz: np.ndarray, 
                     tonic_hz: float,
                     n_bins: int = 36,
                     base_midi: float = 48) -> np.ndarray:
    """
    Quantize a pitch sequence by normalizing tonic and binning to chromatic scale.
    
    This function combines tonic normalization and chromatic quantization
    into a single pipeline for processing pitch contours extracted from audio.
    
    Args:
        f0_hz: Pitch frequencies in Hz (1D array)
        tonic_hz: Tonic frequency in Hz for normalization
        n_bins: Number of chromatic bins (default: 36 for C3-B5)
        base_midi: Base MIDI note for bin 0 (default: 48 = C3)
        
    Returns:
        Quantized sequence as integer array [0, n_bins-1]
        
    Raises:
        QuantizationError: If input parameters are invalid
        
    Examples:
        >>> f0 = np.array([220.0, 246.94, 261.63])  # A3, B3, C4 with A3 tonic
        >>> quantize_sequence(f0, tonic_hz=220.0)
        array([12, 14, 15])  # Normalized and quantized
    """
    f0_hz = np.asarray(f0_hz)
    
    if f0_hz.ndim != 1:
        raise QuantizationError("Input frequency array must be 1-dimensional")
    
    if len(f0_hz) == 0:
        return np.array([], dtype=int)
    
    # Step 1: Normalize by tonic (shift tonic to C4)
    normalized_hz = normalize_by_tonic(f0_hz, tonic_hz)
    
    # Step 2: Quantize to chromatic bins
    quantizer = ChromaticQuantizer(n_bins=n_bins, base_midi=base_midi)
    quantized = quantizer.quantize(normalized_hz)
    
    return quantized