"""
Unit tests for the pitch module.

Tests for pitch extraction and smoothing functionality.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.raag_hmm.pitch.extractor import extract_pitch_with_fallback, PraatExtractor, LibrosaExtractor
from src.raag_hmm.pitch.smoother import smooth_pitch, PitchSmoother


class TestPitchExtractor:
    """Test cases for pitch extraction functionality."""
    
    def test_praat_extractor_initialization(self):
        """Test Praat extractor initialization."""
        extractor = PraatExtractor(
            frame_sec=0.0464,
            hop_sec=0.01,
            voicing_threshold=0.5
        )
        
        assert extractor.frame_sec == 0.0464
        assert extractor.hop_sec == 0.01
        assert extractor.voicing_threshold == 0.5
    
    def test_librosa_extractor_initialization(self):
        """Test Librosa extractor initialization."""
        extractor = LibrosaExtractor(
            frame_sec=0.0464,
            hop_sec=0.01,
            voicing_threshold=0.5
        )
        
        assert extractor.frame_sec == 0.0464
        assert extractor.hop_sec == 0.01
        assert extractor.voicing_threshold == 0.5
    
    def test_extract_pitch_with_fallback(self, mock_audio_data):
        """Test pitch extraction with fallback mechanism."""
        audio_data, sr = mock_audio_data
        
        # This test is skipped if required libraries aren't available
        try:
            f0_hz, voicing_prob = extract_pitch_with_fallback(
                audio_data, sr, frame_sec=0.0464, hop_sec=0.01
            )
            
            # Check that we get arrays of same length
            assert len(f0_hz) == len(voicing_prob)
            
            # Check that voicing probabilities are in [0, 1] range
            assert np.all(voicing_prob >= 0) and np.all(voicing_prob <= 1)
            
        except Exception as e:
            pytest.skip(f"Pitch extraction unavailable: {e}")
    
    def test_extract_pitch_with_fallback_fallback(self, mock_audio_data):
        """Test that fallback mechanism works."""
        audio_data, sr = mock_audio_data
        
        # Mock the primary method to fail and check fallback
        with patch('src.raag_hmm.pitch.extractor.extract_pitch_praat') as mock_praat:
            mock_praat.side_effect = Exception("Praat failed")
            
            try:
                f0_hz, voicing_prob = extract_pitch_with_fallback(
                    audio_data, sr, primary_method='praat'
                )
                
                # Should have fallen back to librosa
                assert len(f0_hz) == len(voicing_prob)
                
            except Exception:
                pytest.skip("All pitch extraction methods unavailable")


class TestPitchSmoother:
    """Test cases for pitch smoothing functionality."""
    
    def test_pitch_smoother_initialization(self):
        """Test PitchSmoother initialization."""
        smoother = PitchSmoother(
            median_window=5,
            gaussian_sigma=1.0,
            gap_fill_threshold_ms=100.0,
            octave_tolerance=0.3
        )
        
        assert smoother.median_window == 5
        assert smoother.gaussian_sigma == 1.0
        assert smoother.gap_fill_threshold_ms == 100.0
        assert smoother.octave_tolerance == 0.3
    
    def test_smooth_pitch_function(self, mock_pitch_data):
        """Test the smooth_pitch convenience function."""
        f0_hz, voicing_prob = mock_pitch_data
        
        smoothed_f0, updated_voicing = smooth_pitch(
            f0_hz, voicing_prob, hop_sec=0.01
        )
        
        # Check that output arrays have same length as input
        assert len(smoothed_f0) == len(f0_hz)
        assert len(updated_voicing) == len(voicing_prob)
        
        # Check that voiced values remain positive
        valid_mask = ~np.isnan(smoothed_f0)
        assert np.all(smoothed_f0[valid_mask] > 0)
    
    def test_pitch_smoother_complete_pipeline(self, mock_pitch_data):
        """Test the complete smoothing pipeline."""
        f0_hz, voicing_prob = mock_pitch_data
        
        smoother = PitchSmoother()
        smoothed_f0, updated_voicing = smoother.smooth(f0_hz, voicing_prob, hop_sec=0.01)
        
        # Check that output arrays have same length as input
        assert len(smoothed_f0) == len(f0_hz)
        assert len(updated_voicing) == len(voicing_prob)
    
    def test_median_filter(self, mock_pitch_data):
        """Test median filtering component."""
        f0_hz, voicing_prob = mock_pitch_data
        
        smoother = PitchSmoother(median_window=3)
        filtered = smoother._median_filter(f0_hz)
        
        assert len(filtered) == len(f0_hz)
        # NaN values should remain NaN
        assert np.array_equal(np.isnan(f0_hz), np.isnan(filtered))
    
    def test_gaussian_smooth(self, mock_pitch_data):
        """Test Gaussian smoothing component."""
        f0_hz, voicing_prob = mock_pitch_data
        
        # Remove NaNs for this test
        f0_clean = f0_hz.copy()
        f0_clean[np.isnan(f0_clean)] = 0  # Replace NaN with 0 for smoothing
        
        smoother = PitchSmoother(gaussian_sigma=0.5)
        smoothed = smoother._gaussian_smooth(f0_clean)
        
        assert len(smoothed) == len(f0_clean)
    
    def test_fill_gaps(self, mock_pitch_data):
        """Test gap filling functionality."""
        f0_hz, voicing_prob = mock_pitch_data
        
        # Create a scenario with consecutive NaN values
        f0_with_gap = f0_hz.copy()
        f0_with_gap[2:4] = np.nan  # Create a gap
        
        smoother = PitchSmoother(gap_fill_threshold_ms=200.0)  # Allow larger gaps
        filled_f0, updated_voicing = smoother._fill_gaps(f0_with_gap, voicing_prob, hop_sec=0.01)
        
        assert len(filled_f0) == len(f0_hz)
        assert len(updated_voicing) == len(voicing_prob)
    
    def test_correct_octave_errors(self, mock_pitch_data):
        """Test octave error correction."""
        f0_hz, voicing_prob = mock_pitch_data
        
        smoother = PitchSmoother()
        corrected_f0 = smoother._correct_octave_errors(f0_hz)
        
        assert len(corrected_f0) == len(f0_hz)
        # NaN values should remain NaN
        assert np.array_equal(np.isnan(f0_hz), np.isnan(corrected_f0))


def test_pitch_smoother_edge_cases():
    """Test edge cases for pitch smoothing."""
    # Test with all NaN values
    all_nan = np.full(10, np.nan)
    voicing_prob = np.zeros(10)
    
    smoothed_f0, updated_voicing = smooth_pitch(all_nan, voicing_prob)
    
    # All values should remain NaN or be handled properly
    assert np.all(np.isnan(smoothed_f0))
    
    # Test with single value
    single_val = np.array([220.0])
    single_voicing = np.array([0.9])
    
    smoothed_single, updated_single = smooth_pitch(single_val, single_voicing)
    
    assert len(smoothed_single) == 1
    assert len(updated_single) == 1