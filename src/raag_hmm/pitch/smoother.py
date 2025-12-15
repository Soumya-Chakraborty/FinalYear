"""
Pitch smoothing and post-processing functionality.

Implements median filtering, Gaussian smoothing, gap filling, and octave error correction
for robust pitch contour processing.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from scipy import ndimage
from scipy.signal import medfilt
from scipy.interpolate import interp1d

from ..config import get_config

logger = logging.getLogger(__name__)


class PitchSmoother:
    """
    Pitch smoothing and post-processing with configurable parameters.
    
    Implements the smoothing pipeline specified in the design document:
    1. Median filtering for spike removal
    2. Gaussian smoothing for continuity
    3. Gap filling for short unvoiced segments
    4. Octave error correction using continuity constraints
    """
    
    def __init__(self,
                 median_window: int = 5,
                 gaussian_sigma: float = 1.0,
                 gap_fill_threshold_ms: float = 100.0,
                 octave_tolerance: float = 0.3):
        """
        Initialize pitch smoother with specified parameters.
        
        Args:
            median_window: Window size for median filtering (default: 5)
            gaussian_sigma: Standard deviation for Gaussian smoothing (default: 1.0)
            gap_fill_threshold_ms: Maximum gap length to fill in milliseconds (default: 100.0)
            octave_tolerance: Tolerance for octave error detection as fraction (default: 0.3)
        """
        self.median_window = median_window
        self.gaussian_sigma = gaussian_sigma
        self.gap_fill_threshold_ms = gap_fill_threshold_ms
        self.octave_tolerance = octave_tolerance
        
    def smooth(self, 
               f0_hz: np.ndarray, 
               voicing_prob: Optional[np.ndarray] = None,
               hop_sec: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply complete smoothing pipeline to pitch contour.
        
        Args:
            f0_hz: F0 frequency array (may contain NaN for unvoiced frames)
            voicing_prob: Voicing probability array (optional)
            hop_sec: Hop size in seconds for gap filling calculations
            
        Returns:
            Tuple of (smoothed_f0_hz, updated_voicing_prob)
        """
        if voicing_prob is None:
            voicing_prob = np.where(np.isnan(f0_hz), 0.0, 1.0)
        
        # Step 1: Median filtering to remove spikes
        f0_median = self._median_filter(f0_hz)
        
        # Step 2: Octave error correction
        f0_octave_corrected = self._correct_octave_errors(f0_median)
        
        # Step 3: Gap filling for short unvoiced segments
        f0_gap_filled, voicing_updated = self._fill_gaps(
            f0_octave_corrected, voicing_prob, hop_sec
        )
        
        # Step 4: Gaussian smoothing for final continuity
        f0_smoothed = self._gaussian_smooth(f0_gap_filled)
        
        logger.debug(f"Smoothing: {np.sum(~np.isnan(f0_hz))} -> {np.sum(~np.isnan(f0_smoothed))} voiced frames")
        
        return f0_smoothed, voicing_updated
    
    def _median_filter(self, f0_hz: np.ndarray) -> np.ndarray:
        """
        Apply median filtering to remove pitch spikes.
        
        Args:
            f0_hz: Input F0 array
            
        Returns:
            Median-filtered F0 array
        """
        # Create a copy to avoid modifying original
        f0_filtered = f0_hz.copy()
        
        # Get voiced frames mask
        voiced_mask = ~np.isnan(f0_hz)
        
        if np.sum(voiced_mask) < self.median_window:
            # Not enough voiced frames for filtering
            return f0_filtered
        
        # Extract voiced segments and their indices
        voiced_indices = np.where(voiced_mask)[0]
        voiced_values = f0_hz[voiced_mask]
        
        # Apply median filter to voiced values only
        if len(voiced_values) >= self.median_window:
            filtered_values = medfilt(voiced_values, kernel_size=self.median_window)
            f0_filtered[voiced_indices] = filtered_values
        
        return f0_filtered
    
    def _correct_octave_errors(self, f0_hz: np.ndarray) -> np.ndarray:
        """
        Correct octave errors using pitch continuity constraints.
        
        Args:
            f0_hz: Input F0 array
            
        Returns:
            Octave-corrected F0 array
        """
        f0_corrected = f0_hz.copy()
        voiced_mask = ~np.isnan(f0_hz)
        
        if np.sum(voiced_mask) < 3:
            # Need at least 3 points for octave correction
            return f0_corrected
        
        voiced_indices = np.where(voiced_mask)[0]
        voiced_values = f0_hz[voiced_mask]
        
        # Process each voiced segment
        corrected_values = voiced_values.copy()
        
        for i in range(1, len(voiced_values) - 1):
            current_f0 = voiced_values[i]
            prev_f0 = corrected_values[i - 1]  # Use corrected previous value
            next_f0 = voiced_values[i + 1]
            
            # Check for octave errors (2x or 0.5x frequency)
            octave_up = current_f0 * 2.0
            octave_down = current_f0 * 0.5
            
            # Calculate distances to neighbors
            dist_original = abs(current_f0 - prev_f0) + abs(current_f0 - next_f0)
            dist_octave_up = abs(octave_up - prev_f0) + abs(octave_up - next_f0)
            dist_octave_down = abs(octave_down - prev_f0) + abs(octave_down - next_f0)
            
            # Choose the frequency that minimizes distance to neighbors
            min_dist = min(dist_original, dist_octave_up, dist_octave_down)
            
            if min_dist == dist_octave_up and dist_octave_up < dist_original * (1 - self.octave_tolerance):
                corrected_values[i] = octave_up
                logger.debug(f"Octave correction: {current_f0:.1f} -> {octave_up:.1f} Hz")
            elif min_dist == dist_octave_down and dist_octave_down < dist_original * (1 - self.octave_tolerance):
                corrected_values[i] = octave_down
                logger.debug(f"Octave correction: {current_f0:.1f} -> {octave_down:.1f} Hz")
        
        # Update the corrected values
        f0_corrected[voiced_indices] = corrected_values
        
        return f0_corrected
    
    def _fill_gaps(self, 
                   f0_hz: np.ndarray, 
                   voicing_prob: np.ndarray,
                   hop_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill short unvoiced gaps using interpolation.
        
        Args:
            f0_hz: Input F0 array
            voicing_prob: Voicing probability array
            hop_sec: Hop size in seconds
            
        Returns:
            Tuple of (gap-filled F0, updated voicing probabilities)
        """
        f0_filled = f0_hz.copy()
        voicing_updated = voicing_prob.copy()
        
        voiced_mask = ~np.isnan(f0_hz)
        
        if np.sum(voiced_mask) < 2:
            # Need at least 2 voiced frames for interpolation
            return f0_filled, voicing_updated
        
        # Calculate maximum gap length in frames
        max_gap_frames = int(self.gap_fill_threshold_ms / 1000.0 / hop_sec)
        
        # Find voiced segments
        voiced_indices = np.where(voiced_mask)[0]
        
        # Look for gaps between voiced segments
        for i in range(len(voiced_indices) - 1):
            start_idx = voiced_indices[i]
            end_idx = voiced_indices[i + 1]
            gap_length = end_idx - start_idx - 1
            
            if 0 < gap_length <= max_gap_frames:
                # Fill this gap with interpolation
                start_f0 = f0_hz[start_idx]
                end_f0 = f0_hz[end_idx]
                
                # Linear interpolation
                gap_indices = np.arange(start_idx + 1, end_idx)
                interpolated_values = np.linspace(start_f0, end_f0, len(gap_indices) + 2)[1:-1]
                
                # Update F0 and voicing probability
                f0_filled[gap_indices] = interpolated_values
                voicing_updated[gap_indices] = 0.5  # Medium confidence for interpolated values
                
                logger.debug(f"Gap filled: {gap_length} frames between {start_f0:.1f} and {end_f0:.1f} Hz")
        
        return f0_filled, voicing_updated
    
    def _gaussian_smooth(self, f0_hz: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to voiced pitch values.
        
        Args:
            f0_hz: Input F0 array
            
        Returns:
            Gaussian-smoothed F0 array
        """
        f0_smoothed = f0_hz.copy()
        voiced_mask = ~np.isnan(f0_hz)
        
        if np.sum(voiced_mask) < 3 or self.gaussian_sigma <= 0:
            # Not enough points for smoothing or smoothing disabled
            return f0_smoothed
        
        # Extract voiced segments and smooth them
        voiced_indices = np.where(voiced_mask)[0]
        voiced_values = f0_hz[voiced_mask]
        
        # Apply Gaussian filter to voiced values
        smoothed_values = ndimage.gaussian_filter1d(
            voiced_values, 
            sigma=self.gaussian_sigma,
            mode='nearest'
        )
        
        # Update smoothed values
        f0_smoothed[voiced_indices] = smoothed_values
        
        return f0_smoothed


def smooth_pitch(f0_hz: np.ndarray,
                voicing_prob: Optional[np.ndarray] = None,
                hop_sec: float = 0.01,
                median_window: int = 5,
                gaussian_sigma: float = 1.0,
                gap_fill_threshold_ms: float = 100.0,
                octave_tolerance: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for pitch smoothing with default parameters.
    
    Args:
        f0_hz: F0 frequency array (may contain NaN for unvoiced frames)
        voicing_prob: Voicing probability array (optional)
        hop_sec: Hop size in seconds for gap filling calculations (default: 0.01)
        median_window: Window size for median filtering (default: 5)
        gaussian_sigma: Standard deviation for Gaussian smoothing (default: 1.0)
        gap_fill_threshold_ms: Maximum gap length to fill in milliseconds (default: 100.0)
        octave_tolerance: Tolerance for octave error detection as fraction (default: 0.3)
        
    Returns:
        Tuple of (smoothed_f0_hz, updated_voicing_prob)
    """
    smoother = PitchSmoother(
        median_window=median_window,
        gaussian_sigma=gaussian_sigma,
        gap_fill_threshold_ms=gap_fill_threshold_ms,
        octave_tolerance=octave_tolerance
    )
    
    return smoother.smooth(f0_hz, voicing_prob, hop_sec)