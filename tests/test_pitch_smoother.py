"""
Unit tests for pitch smoothing and post-processing functionality.

Tests median filtering, Gaussian smoothing, gap filling, and octave error correction.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from raag_hmm.pitch.smoother import PitchSmoother, smooth_pitch


class TestPitchSmoother:
    """Test cases for PitchSmoother class."""
    
    def test_smoother_initialization(self):
        """Test PitchSmoother initialization with default parameters."""
        smoother = PitchSmoother()
        
        assert smoother.median_window == 5
        assert smoother.gaussian_sigma == 1.0
        assert smoother.gap_fill_threshold_ms == 100.0
        assert smoother.octave_tolerance == 0.3
    
    def test_smoother_custom_parameters(self):
        """Test PitchSmoother initialization with custom parameters."""
        smoother = PitchSmoother(
            median_window=7,
            gaussian_sigma=2.0,
            gap_fill_threshold_ms=50.0,
            octave_tolerance=0.2
        )
        
        assert smoother.median_window == 7
        assert smoother.gaussian_sigma == 2.0
        assert smoother.gap_fill_threshold_ms == 50.0
        assert smoother.octave_tolerance == 0.2
    
    def test_median_filter_basic(self):
        """Test median filtering removes spikes."""
        smoother = PitchSmoother(median_window=3)
        
        # Create F0 with spike
        f0_hz = np.array([220.0, 220.0, 440.0, 220.0, 220.0])  # 440 is a spike
        
        f0_filtered = smoother._median_filter(f0_hz)
        
        # Spike should be completely removed by median filter (median of [220, 440, 220] = 220)
        assert f0_filtered[2] == 220.0
    
    def test_median_filter_with_nans(self):
        """Test median filtering handles NaN values correctly."""
        smoother = PitchSmoother(median_window=3)
        
        # Create F0 with NaN (unvoiced) frames
        f0_hz = np.array([220.0, np.nan, 440.0, np.nan, 220.0])
        
        f0_filtered = smoother._median_filter(f0_hz)
        
        # NaN values should remain NaN
        assert np.isnan(f0_filtered[1])
        assert np.isnan(f0_filtered[3])
        
        # Voiced values should be processed
        assert not np.isnan(f0_filtered[0])
        assert not np.isnan(f0_filtered[2])
        assert not np.isnan(f0_filtered[4])
    
    def test_octave_error_correction(self):
        """Test octave error correction."""
        smoother = PitchSmoother(octave_tolerance=0.3)
        
        # Create F0 with octave error (220 -> 440 -> 220, middle should be corrected to 220)
        f0_hz = np.array([220.0, 440.0, 220.0])
        
        f0_corrected = smoother._correct_octave_errors(f0_hz)
        
        # Middle value should be corrected to 220 (octave down from 440)
        assert abs(f0_corrected[1] - 220.0) < abs(f0_corrected[1] - 440.0)
    
    def test_octave_error_correction_up(self):
        """Test octave error correction upward."""
        smoother = PitchSmoother(octave_tolerance=0.3)
        
        # Create F0 with octave error (440 -> 220 -> 440, middle should be corrected to 440)
        f0_hz = np.array([440.0, 220.0, 440.0])
        
        f0_corrected = smoother._correct_octave_errors(f0_hz)
        
        # Middle value should be corrected to 440 (octave up from 220)
        assert abs(f0_corrected[1] - 440.0) < abs(f0_corrected[1] - 220.0)
    
    def test_octave_error_no_correction_needed(self):
        """Test octave error correction when no correction is needed."""
        smoother = PitchSmoother(octave_tolerance=0.3)
        
        # Create smooth F0 contour
        f0_hz = np.array([220.0, 225.0, 230.0, 225.0, 220.0])
        
        f0_corrected = smoother._correct_octave_errors(f0_hz)
        
        # Values should remain largely unchanged
        np.testing.assert_allclose(f0_corrected, f0_hz, rtol=0.1)
    
    def test_gap_filling_short_gap(self):
        """Test gap filling for short unvoiced segments."""
        smoother = PitchSmoother(gap_fill_threshold_ms=100.0)
        
        # Create F0 with short gap (2 frames at 10ms hop = 20ms gap)
        f0_hz = np.array([220.0, np.nan, np.nan, 240.0])
        voicing_prob = np.array([1.0, 0.0, 0.0, 1.0])
        
        f0_filled, voicing_updated = smoother._fill_gaps(f0_hz, voicing_prob, hop_sec=0.01)
        
        # Gap should be filled
        assert not np.isnan(f0_filled[1])
        assert not np.isnan(f0_filled[2])
        
        # Interpolated values should be between start and end
        assert 220.0 < f0_filled[1] < 240.0
        assert 220.0 < f0_filled[2] < 240.0
        
        # Voicing probabilities should be updated
        assert voicing_updated[1] > 0.0
        assert voicing_updated[2] > 0.0
    
    def test_gap_filling_long_gap(self):
        """Test gap filling skips long unvoiced segments."""
        smoother = PitchSmoother(gap_fill_threshold_ms=50.0)  # 50ms threshold
        
        # Create F0 with long gap (6 frames at 10ms hop = 60ms gap)
        f0_hz = np.array([220.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 240.0])
        voicing_prob = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        f0_filled, voicing_updated = smoother._fill_gaps(f0_hz, voicing_prob, hop_sec=0.01)
        
        # Gap should NOT be filled (too long)
        assert np.isnan(f0_filled[1])
        assert np.isnan(f0_filled[2])
        assert np.isnan(f0_filled[3])
        
        # Voicing probabilities should remain unchanged
        assert voicing_updated[1] == 0.0
        assert voicing_updated[2] == 0.0
    
    def test_gaussian_smoothing(self):
        """Test Gaussian smoothing."""
        smoother = PitchSmoother(gaussian_sigma=1.0)
        
        # Create noisy F0 contour
        f0_hz = np.array([220.0, 225.0, 215.0, 230.0, 220.0])
        
        f0_smoothed = smoother._gaussian_smooth(f0_hz)
        
        # Smoothed values should be less variable
        original_variance = np.var(f0_hz)
        smoothed_variance = np.var(f0_smoothed)
        
        assert smoothed_variance <= original_variance
    
    def test_gaussian_smoothing_with_nans(self):
        """Test Gaussian smoothing handles NaN values."""
        smoother = PitchSmoother(gaussian_sigma=1.0)
        
        # Create F0 with NaN values
        f0_hz = np.array([220.0, np.nan, 225.0, np.nan, 220.0])
        
        f0_smoothed = smoother._gaussian_smooth(f0_hz)
        
        # NaN values should remain NaN
        assert np.isnan(f0_smoothed[1])
        assert np.isnan(f0_smoothed[3])
        
        # Voiced values should be smoothed
        assert not np.isnan(f0_smoothed[0])
        assert not np.isnan(f0_smoothed[2])
        assert not np.isnan(f0_smoothed[4])
    
    def test_complete_smoothing_pipeline(self):
        """Test complete smoothing pipeline."""
        smoother = PitchSmoother(
            median_window=3,
            gaussian_sigma=1.0,
            gap_fill_threshold_ms=50.0,
            octave_tolerance=0.3
        )
        
        # Create complex F0 with various issues
        f0_hz = np.array([220.0, 440.0, 220.0, np.nan, 225.0, 215.0, 230.0])  # spike, gap, noise
        voicing_prob = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        
        f0_smoothed, voicing_updated = smoother.smooth(f0_hz, voicing_prob, hop_sec=0.01)
        
        # Check output format
        assert len(f0_smoothed) == len(f0_hz)
        assert len(voicing_updated) == len(voicing_prob)
        
        # Octave error should be corrected
        assert abs(f0_smoothed[1] - 220.0) < abs(f0_hz[1] - 220.0)
        
        # Gap should be filled
        assert not np.isnan(f0_smoothed[3])
        
        # Noise should be reduced
        noise_original = np.var(f0_hz[4:7])
        noise_smoothed = np.var(f0_smoothed[4:7])
        assert noise_smoothed <= noise_original
    
    def test_smoothing_insufficient_data(self):
        """Test smoothing with insufficient data."""
        smoother = PitchSmoother()
        
        # Single voiced frame
        f0_hz = np.array([220.0])
        voicing_prob = np.array([1.0])
        
        f0_smoothed, voicing_updated = smoother.smooth(f0_hz, voicing_prob)
        
        # Should not crash and return reasonable output
        assert len(f0_smoothed) == 1
        assert len(voicing_updated) == 1
        assert f0_smoothed[0] == 220.0
    
    def test_smoothing_all_unvoiced(self):
        """Test smoothing with all unvoiced frames."""
        smoother = PitchSmoother()
        
        # All unvoiced frames
        f0_hz = np.array([np.nan, np.nan, np.nan])
        voicing_prob = np.array([0.0, 0.0, 0.0])
        
        f0_smoothed, voicing_updated = smoother.smooth(f0_hz, voicing_prob)
        
        # Should remain all unvoiced
        assert np.all(np.isnan(f0_smoothed))
        assert np.all(voicing_updated == 0.0)


class TestSmoothPitchFunction:
    """Test cases for smooth_pitch convenience function."""
    
    def test_smooth_pitch_basic(self):
        """Test basic smooth_pitch function usage."""
        # Create test data
        f0_hz = np.array([220.0, 440.0, 220.0, np.nan, 225.0])
        voicing_prob = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
        
        f0_smoothed, voicing_updated = smooth_pitch(f0_hz, voicing_prob)
        
        # Check output format
        assert isinstance(f0_smoothed, np.ndarray)
        assert isinstance(voicing_updated, np.ndarray)
        assert len(f0_smoothed) == len(f0_hz)
        assert len(voicing_updated) == len(voicing_prob)
    
    def test_smooth_pitch_no_voicing_prob(self):
        """Test smooth_pitch without voicing probabilities."""
        f0_hz = np.array([220.0, 225.0, np.nan, 230.0])
        
        f0_smoothed, voicing_updated = smooth_pitch(f0_hz)
        
        # Should generate voicing probabilities automatically
        assert len(voicing_updated) == len(f0_hz)
        # NaN frame may be filled by gap filling, so check if it's either 0.0 or 0.5 (interpolated)
        assert voicing_updated[2] in [0.0, 0.5]  # Could be filled by gap filling
        assert voicing_updated[0] > 0.0   # Voiced frame should have > 0 voicing
    
    def test_smooth_pitch_custom_parameters(self):
        """Test smooth_pitch with custom parameters."""
        f0_hz = np.array([220.0, 225.0, 230.0, 225.0, 220.0])
        
        f0_smoothed, voicing_updated = smooth_pitch(
            f0_hz,
            median_window=3,
            gaussian_sigma=2.0,
            gap_fill_threshold_ms=50.0,
            octave_tolerance=0.2
        )
        
        # Should complete without error
        assert len(f0_smoothed) == len(f0_hz)
        assert len(voicing_updated) == len(f0_hz)


class TestSmoothingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test smoothing with empty arrays."""
        smoother = PitchSmoother()
        
        f0_hz = np.array([])
        voicing_prob = np.array([])
        
        f0_smoothed, voicing_updated = smoother.smooth(f0_hz, voicing_prob)
        
        assert len(f0_smoothed) == 0
        assert len(voicing_updated) == 0
    
    def test_mismatched_array_lengths(self):
        """Test smoothing with mismatched array lengths."""
        smoother = PitchSmoother()
        
        f0_hz = np.array([220.0, 225.0, 230.0])
        voicing_prob = np.array([1.0, 1.0])  # Different length
        
        # Should handle gracefully or raise appropriate error
        try:
            f0_smoothed, voicing_updated = smoother.smooth(f0_hz, voicing_prob)
            # If it doesn't raise an error, check that it produces reasonable output
            assert len(f0_smoothed) == len(f0_hz)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for mismatched lengths
            pass
    
    def test_extreme_parameter_values(self):
        """Test smoothing with extreme parameter values."""
        # Zero sigma (no smoothing)
        smoother = PitchSmoother(gaussian_sigma=0.0)
        f0_hz = np.array([220.0, 225.0, 230.0])
        
        f0_smoothed, _ = smoother.smooth(f0_hz)
        np.testing.assert_allclose(f0_smoothed, f0_hz)
        
        # Very large sigma
        smoother = PitchSmoother(gaussian_sigma=100.0)
        f0_smoothed, _ = smoother.smooth(f0_hz)
        
        # Should be very smooth (low variance)
        assert np.var(f0_smoothed) < np.var(f0_hz)
    
    def test_parameter_sensitivity(self):
        """Test parameter sensitivity for smoothing effectiveness."""
        # Create noisy signal
        np.random.seed(42)  # For reproducibility
        base_f0 = 220.0
        noise = np.random.normal(0, 5, 20)
        f0_hz = base_f0 + noise
        
        # Test different smoothing strengths
        smoother_light = PitchSmoother(gaussian_sigma=0.5)
        smoother_heavy = PitchSmoother(gaussian_sigma=2.0)
        
        f0_light, _ = smoother_light.smooth(f0_hz)
        f0_heavy, _ = smoother_heavy.smooth(f0_hz)
        
        # Heavy smoothing should produce smoother result
        variance_light = np.var(f0_light)
        variance_heavy = np.var(f0_heavy)
        
        assert variance_heavy < variance_light


if __name__ == "__main__":
    pytest.main([__file__])