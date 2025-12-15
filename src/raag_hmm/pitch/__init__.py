"""
Pitch extraction module.

Multi-method pitch detection with robust fallback mechanisms.
"""

from .extractor import (
    extract_pitch_praat, extract_pitch_librosa, extract_pitch_with_fallback,
    PraatExtractor, LibrosaExtractor
)

__all__ = [
    "extract_pitch_praat",
    "extract_pitch_librosa",
    "extract_pitch_with_fallback",
    "PraatExtractor",
    "LibrosaExtractor"
]

# Import smoother
from .smoother import smooth_pitch, PitchSmoother
__all__.extend(["smooth_pitch", "PitchSmoother"])