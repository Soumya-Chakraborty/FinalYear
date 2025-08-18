"""
Pitch extraction implementations using multiple methods.

This module provides robust pitch extraction with Praat as primary method
and librosa as fallback, following the design specifications.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union
from pathlib import Path

try:
    import parselmouth
    PRAAT_AVAILABLE = True
except ImportError:
    PRAAT_AVAILABLE = False
    logging.warning("Praat parselmouth not available. Falling back to librosa only.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.error("Librosa not available. Pitch extraction will fail.")

from ..exceptions import PitchExtractionError
from ..config import get_config

logger = logging.getLogger(__name__)


class PraatExtractor:
    """
    Praat-based pitch extraction using parselmouth library.
    
    Implements the primary pitch extraction method with voicing probability
    and configurable frame/hop parameters as specified in the design.
    """
    
    def __init__(self, 
                 frame_sec: float = 0.0464,
                 hop_sec: float = 0.01,
                 voicing_threshold: float = 0.5,
                 f0_min: float = 75.0,
                 f0_max: float = 600.0):
        """
        Initialize Praat extractor with specified parameters.
        
        Args:
            frame_sec: Frame size in seconds (default: 0.0464s)
            hop_sec: Hop size in seconds (default: 0.01s) 
            voicing_threshold: Minimum voicing probability (default: 0.5)
            f0_min: Minimum F0 frequency in Hz (default: 75.0)
            f0_max: Maximum F0 frequency in Hz (default: 600.0)
        """
        if not PRAAT_AVAILABLE:
            raise PitchExtractionError("Praat parselmouth library not available")
            
        self.frame_sec = frame_sec
        self.hop_sec = hop_sec
        self.voicing_threshold = voicing_threshold
        self.f0_min = f0_min
        self.f0_max = f0_max
        
    def extract(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using Praat's autocorrelation method.
        
        Args:
            y: Audio signal array
            sr: Sample rate in Hz
            
        Returns:
            Tuple of (f0_hz, voicing_prob) arrays
            
        Raises:
            PitchExtractionError: If extraction fails
        """
        try:
            # Create Praat Sound object
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            
            # Extract pitch with specified parameters
            pitch = sound.to_pitch_ac(
                time_step=self.hop_sec,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max
            )
            
            # Get time points and F0 values
            times = pitch.xs()
            f0_values = pitch.selected_array['frequency']
            
            # Get voicing strength (strength of periodicity)
            # Praat's strength is similar to voicing probability
            voicing_strength = pitch.selected_array['strength']
            
            # Convert undefined values (0.0) to NaN
            f0_hz = np.where(f0_values == 0.0, np.nan, f0_values)
            voicing_prob = np.where(f0_values == 0.0, 0.0, voicing_strength)
            
            # Apply voicing threshold
            f0_hz = np.where(voicing_prob >= self.voicing_threshold, f0_hz, np.nan)
            
            logger.debug(f"Praat extraction: {len(f0_hz)} frames, "
                        f"{np.sum(~np.isnan(f0_hz))} voiced")
            
            return f0_hz, voicing_prob
            
        except Exception as e:
            raise PitchExtractionError(f"Praat pitch extraction failed: {str(e)}")


class LibrosaExtractor:
    """
    Librosa-based pitch extraction with pyin and yin algorithms.
    
    Provides fallback pitch extraction when Praat fails, with consistent
    output format and parameter mapping.
    """
    
    def __init__(self,
                 frame_sec: float = 0.0464,
                 hop_sec: float = 0.01,
                 voicing_threshold: float = 0.5,
                 f0_min: float = 75.0,
                 f0_max: float = 600.0):
        """
        Initialize librosa extractor with specified parameters.
        
        Args:
            frame_sec: Frame size in seconds (default: 0.0464s)
            hop_sec: Hop size in seconds (default: 0.01s)
            voicing_threshold: Minimum voicing probability (default: 0.5)
            f0_min: Minimum F0 frequency in Hz (default: 75.0)
            f0_max: Maximum F0 frequency in Hz (default: 600.0)
        """
        if not LIBROSA_AVAILABLE:
            raise PitchExtractionError("Librosa library not available")
            
        self.frame_sec = frame_sec
        self.hop_sec = hop_sec
        self.voicing_threshold = voicing_threshold
        self.f0_min = f0_min
        self.f0_max = f0_max
        
    def extract(self, y: np.ndarray, sr: int, method: str = 'pyin') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using librosa pyin or yin algorithms.
        
        Args:
            y: Audio signal array
            sr: Sample rate in Hz
            method: Algorithm to use ('pyin' or 'yin')
            
        Returns:
            Tuple of (f0_hz, voicing_prob) arrays
            
        Raises:
            PitchExtractionError: If extraction fails
        """
        try:
            # Calculate hop length in samples
            hop_length = int(self.hop_sec * sr)
            
            if method == 'pyin':
                # Use pyin which provides voicing probabilities
                f0_hz, voiced_flag, voiced_prob = librosa.pyin(
                    y,
                    fmin=self.f0_min,
                    fmax=self.f0_max,
                    sr=sr,
                    hop_length=hop_length,
                    resolution=0.01  # Cents resolution
                )
                
                # Convert voiced probabilities to our format
                voicing_prob = np.where(voiced_flag, voiced_prob, 0.0)
                
            elif method == 'yin':
                # Use yin algorithm (no voicing probabilities)
                f0_hz = librosa.yin(
                    y,
                    fmin=self.f0_min,
                    fmax=self.f0_max,
                    sr=sr,
                    hop_length=hop_length
                )
                
                # Estimate voicing probability based on pitch continuity
                voicing_prob = self._estimate_voicing_from_continuity(f0_hz)
                
            else:
                raise ValueError(f"Unknown method: {method}. Use 'pyin' or 'yin'")
            
            # Apply voicing threshold
            f0_hz = np.where(voicing_prob >= self.voicing_threshold, f0_hz, np.nan)
            
            logger.debug(f"Librosa {method} extraction: {len(f0_hz)} frames, "
                        f"{np.sum(~np.isnan(f0_hz))} voiced")
            
            return f0_hz, voicing_prob
            
        except Exception as e:
            raise PitchExtractionError(f"Librosa {method} pitch extraction failed: {str(e)}")
    
    def _estimate_voicing_from_continuity(self, f0_hz: np.ndarray) -> np.ndarray:
        """
        Estimate voicing probability from pitch continuity for yin algorithm.
        
        Args:
            f0_hz: F0 frequency array
            
        Returns:
            Estimated voicing probability array
        """
        voicing_prob = np.zeros_like(f0_hz)
        
        # Simple heuristic: high probability for non-zero, continuous pitch
        valid_mask = ~np.isnan(f0_hz) & (f0_hz > 0)
        
        if np.any(valid_mask):
            # Calculate local continuity, handling NaN values
            f0_clean = np.where(valid_mask, f0_hz, 0.0)
            f0_diff = np.abs(np.diff(f0_clean, prepend=f0_clean[0]))
            continuity = np.exp(-f0_diff / 50.0)  # Exponential decay with pitch jumps
            
            # Assign high probability to continuous regions
            voicing_prob[valid_mask] = np.minimum(continuity[valid_mask], 1.0)
        
        # Ensure no NaN values in output
        voicing_prob = np.nan_to_num(voicing_prob, nan=0.0)
        
        return voicing_prob


def extract_pitch_with_fallback(y: np.ndarray,
                               sr: int,
                               frame_sec: float = 0.0464,
                               hop_sec: float = 0.01,
                               voicing_threshold: float = 0.5,
                               primary_method: str = 'praat',
                               fallback_methods: list = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pitch with automatic fallback between methods.
    
    Args:
        y: Audio signal array
        sr: Sample rate in Hz
        frame_sec: Frame size in seconds (default: 0.0464s)
        hop_sec: Hop size in seconds (default: 0.01s)
        voicing_threshold: Minimum voicing probability (default: 0.5)
        primary_method: Primary extraction method ('praat' or 'librosa_pyin')
        fallback_methods: List of fallback methods to try if primary fails
        
    Returns:
        Tuple of (f0_hz, voicing_prob) arrays
        
    Raises:
        PitchExtractionError: If all methods fail
    """
    if fallback_methods is None:
        fallback_methods = ['librosa_pyin', 'librosa_yin']
    
    # Try primary method first
    methods_to_try = [primary_method] + fallback_methods
    
    for method in methods_to_try:
        try:
            if method == 'praat':
                if not PRAAT_AVAILABLE:
                    logger.warning("Praat not available, trying fallback methods")
                    continue
                return extract_pitch_praat(y, sr, frame_sec, hop_sec, voicing_threshold)
                
            elif method == 'librosa_pyin':
                if not LIBROSA_AVAILABLE:
                    logger.warning("Librosa not available, trying other methods")
                    continue
                return extract_pitch_librosa(y, sr, 'pyin', frame_sec, hop_sec, voicing_threshold)
                
            elif method == 'librosa_yin':
                if not LIBROSA_AVAILABLE:
                    logger.warning("Librosa not available, trying other methods")
                    continue
                return extract_pitch_librosa(y, sr, 'yin', frame_sec, hop_sec, voicing_threshold)
                
            else:
                logger.warning(f"Unknown method: {method}, skipping")
                continue
                
        except PitchExtractionError as e:
            logger.warning(f"Method {method} failed: {str(e)}, trying next method")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error with method {method}: {str(e)}, trying next method")
            continue
    
    # If all methods failed
    raise PitchExtractionError(f"All pitch extraction methods failed for the given audio")


# Convenience functions for direct usage
def extract_pitch_praat(y: np.ndarray, 
                       sr: int,
                       frame_sec: float = 0.0464,
                       hop_sec: float = 0.01,
                       voicing_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pitch using Praat parselmouth library.
    
    Args:
        y: Audio signal array
        sr: Sample rate in Hz
        frame_sec: Frame size in seconds (default: 0.0464s)
        hop_sec: Hop size in seconds (default: 0.01s)
        voicing_threshold: Minimum voicing probability (default: 0.5)
        
    Returns:
        Tuple of (f0_hz, voicing_prob) arrays
        
    Raises:
        PitchExtractionError: If Praat is unavailable or extraction fails
    """
    extractor = PraatExtractor(
        frame_sec=frame_sec,
        hop_sec=hop_sec,
        voicing_threshold=voicing_threshold
    )
    return extractor.extract(y, sr)


def extract_pitch_librosa(y: np.ndarray,
                         sr: int,
                         method: str = 'pyin',
                         frame_sec: float = 0.0464,
                         hop_sec: float = 0.01,
                         voicing_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pitch using librosa pyin or yin algorithms.
    
    Args:
        y: Audio signal array
        sr: Sample rate in Hz
        method: Algorithm to use ('pyin' or 'yin')
        frame_sec: Frame size in seconds (default: 0.0464s)
        hop_sec: Hop size in seconds (default: 0.01s)
        voicing_threshold: Minimum voicing probability (default: 0.5)
        
    Returns:
        Tuple of (f0_hz, voicing_prob) arrays
        
    Raises:
        PitchExtractionError: If librosa is unavailable or extraction fails
    """
    extractor = LibrosaExtractor(
        frame_sec=frame_sec,
        hop_sec=hop_sec,
        voicing_threshold=voicing_threshold
    )
    return extractor.extract(y, sr, method=method)