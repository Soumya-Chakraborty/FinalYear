"""
Quantization module.

Convert pitch frequencies to discrete chromatic bins normalized by tonic.
"""

from .chromatic import hz_to_midi, nearest_chromatic_bin, ChromaticQuantizer
from .tonic import normalize_by_tonic, TonicNormalizer
from .sequence import quantize_sequence

__all__ = [
    "hz_to_midi",
    "nearest_chromatic_bin",
    "ChromaticQuantizer",
    "normalize_by_tonic", 
    "TonicNormalizer",
    "quantize_sequence"
]