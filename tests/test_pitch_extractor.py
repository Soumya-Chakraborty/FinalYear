"""
Unit tests for pitch extraction functionality.

Tests Praat and librosa-based pitch extraction methods with various
edge cases and accuracy validation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from raag_hmm.pitch.extractor import (
    PraatExtractor, LibrosaExtractor,
    extract_pitch_praat, extract_pitch_librosa, extract_pitch_with_fallback,
    PRAAT_AVAILABLE, LIBROSA_AVAILABLE
)
from raag_hmm.exceptions import PitchExtractionError


class TestPraatExtractor:
    """Test cases for Praat-based pitch extraction."""
    
    @pytest.fixture
    def sample_audio(self):
        """Generate synthetic audio signal for testing."""
        sr = 22050
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a simple sine wave at 220 Hz (A3)
        frequency = 220.0
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        return audio, sr, frequency
    
    @pytest.fixture
    def noisy_audio(self):
        """Generate noisy audio signal for testing."""
        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        
        # Mix of sine wave and noise
        signal = 0.3 * np.sin(2 * np.pi * 220 * t)
        noise = 0.1 * np.random.randn(len(t))
        
        return signal + noise, sr
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_praat_extractor_initialization(self):
        """Test PraatExtractor initialization with default parameters."""
        extractor = PraatExtractor()
        
        assert extractor.frame_sec == 0.0464
        assert extractor.hop_sec == 0.01
        assert extractor.voicing_threshold == 0.5
        assert extractor.f0_min == 75.0
        assert extractor.f0_max == 600.0
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_praat_extractor_custom_parameters(self):
        """Test PraatExtractor initialization with custom parameters."""
        extractor = PraatExtractor(
            frame_sec=0.05,
            hop_sec=0.02,
            voicing_threshold=0.7,
            f0_min=100.0,
            f0_max=500.0
        )
        
        assert extractor.frame_sec == 0.05
        assert extractor.hop_sec == 0.02
        assert extractor.voicing_threshold == 0.7
        assert extractor.f0_min == 100.0
        assert extractor.f0_max == 500.0
    
    def test_praat_extractor_unavailable(self):
        """Test PraatExtractor raises error when Praat is unavailable."""
        with patch('raag_hmm.pitch.extractor.PRAAT_AVAILABLE', False):
            with pytest.raises(PitchExtractionError, match="Praat parselmouth library not available"):
                PraatExtractor()
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_praat_extract_clean_signal(self, sample_audio):
        """Test Praat extraction on clean synthetic signal."""
        audio, sr, expected_freq = sample_audio
        extractor = PraatExtractor(voicing_threshold=0.3)  # Lower threshold for synthetic signal
        
        f0_hz, voicing_prob = extractor.extract(audio, sr)
        
        # Check output format
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)
        assert len(f0_hz) == len(voicing_prob)
        
        # Check that we detected some voiced frames
        voiced_frames = ~np.isnan(f0_hz)
        assert np.sum(voiced_frames) > 0, "Should detect some voiced frames"
        
        # Check frequency accuracy (within 10% tolerance)
        if np.sum(voiced_frames) > 0:
            median_f0 = np.median(f0_hz[voiced_frames])
            assert abs(median_f0 - expected_freq) / expected_freq < 0.1, \
                f"Expected ~{expected_freq} Hz, got {median_f0} Hz"
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_praat_extract_silence(self):
        """Test Praat extraction on silence."""
        sr = 22050
        duration = 0.5
        audio = np.zeros(int(sr * duration))
        
        extractor = PraatExtractor()
        f0_hz, voicing_prob = extractor.extract(audio, sr)
        
        # Should detect mostly unvoiced frames
        voiced_frames = ~np.isnan(f0_hz)
        voiced_ratio = np.sum(voiced_frames) / len(f0_hz)
        assert voiced_ratio < 0.1, "Silence should have very few voiced frames"
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_praat_extract_noisy_signal(self, noisy_audio):
        """Test Praat extraction on noisy signal."""
        audio, sr = noisy_audio
        extractor = PraatExtractor()
        
        f0_hz, voicing_prob = extractor.extract(audio, sr)
        
        # Should still detect some voiced frames despite noise
        voiced_frames = ~np.isnan(f0_hz)
        assert len(f0_hz) > 0
        assert len(voicing_prob) > 0
        
        # Voicing probabilities should be in valid range
        assert np.all((voicing_prob >= 0.0) & (voicing_prob <= 1.0))
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_praat_voicing_threshold(self, sample_audio):
        """Test voicing threshold application."""
        audio, sr, _ = sample_audio
        
        # Test with different thresholds
        low_threshold = PraatExtractor(voicing_threshold=0.1)
        high_threshold = PraatExtractor(voicing_threshold=0.9)
        
        f0_low, _ = low_threshold.extract(audio, sr)
        f0_high, _ = high_threshold.extract(audio, sr)
        
        # Lower threshold should detect more voiced frames
        voiced_low = np.sum(~np.isnan(f0_low))
        voiced_high = np.sum(~np.isnan(f0_high))
        
        assert voiced_low >= voiced_high, "Lower threshold should detect more voiced frames"
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_praat_extract_error_handling(self):
        """Test Praat extraction error handling."""
        extractor = PraatExtractor()
        
        # Test with invalid input
        with pytest.raises(PitchExtractionError):
            extractor.extract(np.array([]), 22050)  # Empty array
    
    @pytest.mark.skipif(not PRAAT_AVAILABLE, reason="Praat parselmouth not available")
    def test_extract_pitch_praat_function(self, sample_audio):
        """Test convenience function extract_pitch_praat."""
        audio, sr, expected_freq = sample_audio
        
        f0_hz, voicing_prob = extract_pitch_praat(
            audio, sr, voicing_threshold=0.3
        )
        
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)
        assert len(f0_hz) == len(voicing_prob)
        
        # Should detect some voiced frames
        voiced_frames = ~np.isnan(f0_hz)
        assert np.sum(voiced_frames) > 0


class TestLibrosaExtractor:
    """Test cases for librosa-based pitch extraction."""
    
    @pytest.fixture
    def sample_audio(self):
        """Generate synthetic audio signal for testing."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a simple sine wave at 220 Hz (A3)
        frequency = 220.0
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        return audio, sr, frequency
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_librosa_extractor_initialization(self):
        """Test LibrosaExtractor initialization."""
        extractor = LibrosaExtractor()
        
        assert extractor.frame_sec == 0.0464
        assert extractor.hop_sec == 0.01
        assert extractor.voicing_threshold == 0.5
    
    def test_librosa_extractor_unavailable(self):
        """Test LibrosaExtractor raises error when librosa is unavailable."""
        with patch('raag_hmm.pitch.extractor.LIBROSA_AVAILABLE', False):
            with pytest.raises(PitchExtractionError, match="Librosa library not available"):
                LibrosaExtractor()
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_librosa_extract_pyin(self, sample_audio):
        """Test librosa extraction using pyin method."""
        audio, sr, expected_freq = sample_audio
        extractor = LibrosaExtractor(voicing_threshold=0.3)
        
        f0_hz, voicing_prob = extractor.extract(audio, sr, method='pyin')
        
        # Check output format
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)
        assert len(f0_hz) == len(voicing_prob)
        
        # Check that we detected some voiced frames
        voiced_frames = ~np.isnan(f0_hz)
        assert np.sum(voiced_frames) > 0, "Should detect some voiced frames"
        
        # Voicing probabilities should be in valid range
        assert np.all((voicing_prob >= 0.0) & (voicing_prob <= 1.0))
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_librosa_extract_yin(self, sample_audio):
        """Test librosa extraction using yin method."""
        audio, sr, expected_freq = sample_audio
        extractor = LibrosaExtractor(voicing_threshold=0.3)
        
        f0_hz, voicing_prob = extractor.extract(audio, sr, method='yin')
        
        # Check output format
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)
        assert len(f0_hz) == len(voicing_prob)
        
        # YIN should still produce reasonable results
        voiced_frames = ~np.isnan(f0_hz)
        assert len(f0_hz) > 0
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_librosa_invalid_method(self, sample_audio):
        """Test librosa extraction with invalid method."""
        audio, sr, _ = sample_audio
        extractor = LibrosaExtractor()
        
        with pytest.raises(PitchExtractionError, match="Unknown method"):
            extractor.extract(audio, sr, method='invalid_method')
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_librosa_voicing_estimation(self):
        """Test voicing probability estimation for yin method."""
        extractor = LibrosaExtractor()
        
        # Test with synthetic F0 sequence
        f0_hz = np.array([220.0, 220.5, 221.0, np.nan, np.nan, 220.0])
        voicing_prob = extractor._estimate_voicing_from_continuity(f0_hz)
        
        assert len(voicing_prob) == len(f0_hz)
        assert np.all((voicing_prob >= 0.0) & (voicing_prob <= 1.0))
        
        # NaN values should have zero voicing probability
        nan_mask = np.isnan(f0_hz)
        assert np.all(voicing_prob[nan_mask] == 0.0)
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_extract_pitch_librosa_function(self, sample_audio):
        """Test convenience function extract_pitch_librosa."""
        audio, sr, _ = sample_audio
        
        # Test pyin method
        f0_hz, voicing_prob = extract_pitch_librosa(
            audio, sr, method='pyin', voicing_threshold=0.3
        )
        
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)
        
        # Test yin method
        f0_hz_yin, voicing_prob_yin = extract_pitch_librosa(
            audio, sr, method='yin', voicing_threshold=0.3
        )
        
        assert isinstance(f0_hz_yin, np.ndarray)
        assert isinstance(voicing_prob_yin, np.ndarray)


class TestPitchExtractionFallback:
    """Test cases for automatic fallback mechanism."""
    
    @pytest.fixture
    def sample_audio(self):
        """Generate synthetic audio signal for testing."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a simple sine wave at 220 Hz (A3)
        frequency = 220.0
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        return audio, sr, frequency
    
    @pytest.mark.skipif(not (PRAAT_AVAILABLE and LIBROSA_AVAILABLE), 
                       reason="Both Praat and librosa required")
    def test_fallback_praat_primary(self, sample_audio):
        """Test fallback with Praat as primary method."""
        audio, sr, _ = sample_audio
        
        f0_hz, voicing_prob = extract_pitch_with_fallback(
            audio, sr, primary_method='praat', voicing_threshold=0.3
        )
        
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)
        assert len(f0_hz) == len(voicing_prob)
        
        # Should detect some voiced frames
        voiced_frames = ~np.isnan(f0_hz)
        assert np.sum(voiced_frames) > 0
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_fallback_librosa_primary(self, sample_audio):
        """Test fallback with librosa as primary method."""
        audio, sr, _ = sample_audio
        
        f0_hz, voicing_prob = extract_pitch_with_fallback(
            audio, sr, primary_method='librosa_pyin', voicing_threshold=0.3
        )
        
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)
        assert len(f0_hz) == len(voicing_prob)
    
    def test_fallback_praat_unavailable(self, sample_audio):
        """Test fallback when Praat is unavailable."""
        audio, sr, _ = sample_audio
        
        with patch('raag_hmm.pitch.extractor.PRAAT_AVAILABLE', False):
            if LIBROSA_AVAILABLE:
                # Should fallback to librosa
                f0_hz, voicing_prob = extract_pitch_with_fallback(
                    audio, sr, primary_method='praat', voicing_threshold=0.3
                )
                assert isinstance(f0_hz, np.ndarray)
                assert isinstance(voicing_prob, np.ndarray)
            else:
                # Should raise error if no methods available
                with pytest.raises(PitchExtractionError):
                    extract_pitch_with_fallback(audio, sr, primary_method='praat')
    
    def test_fallback_all_methods_fail(self, sample_audio):
        """Test fallback when all methods fail."""
        audio, sr, _ = sample_audio
        
        # Mock all methods to fail
        with patch('raag_hmm.pitch.extractor.PRAAT_AVAILABLE', False), \
             patch('raag_hmm.pitch.extractor.LIBROSA_AVAILABLE', False):
            
            with pytest.raises(PitchExtractionError, match="All pitch extraction methods failed"):
                extract_pitch_with_fallback(audio, sr)
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_fallback_custom_methods(self, sample_audio):
        """Test fallback with custom method list."""
        audio, sr, _ = sample_audio
        
        f0_hz, voicing_prob = extract_pitch_with_fallback(
            audio, sr, 
            primary_method='librosa_pyin',
            fallback_methods=['librosa_yin'],
            voicing_threshold=0.3
        )
        
        assert isinstance(f0_hz, np.ndarray)
        assert isinstance(voicing_prob, np.ndarray)


class TestPitchExtractionIntegration:
    """Integration tests for pitch extraction methods."""
    
    @pytest.fixture
    def multi_tone_audio(self):
        """Generate audio with multiple frequency components."""
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create signal with frequency sweep
        f_start, f_end = 200.0, 400.0
        freq_sweep = f_start + (f_end - f_start) * t / duration
        audio = 0.5 * np.sin(2 * np.pi * freq_sweep * t)
        
        return audio, sr
    
    @pytest.mark.skipif(not (PRAAT_AVAILABLE and LIBROSA_AVAILABLE), 
                       reason="Both Praat and librosa required")
    def test_method_consistency(self, multi_tone_audio):
        """Test consistency between Praat and librosa methods."""
        audio, sr = multi_tone_audio
        
        # Extract with both methods
        f0_praat, _ = extract_pitch_praat(audio, sr, voicing_threshold=0.3)
        f0_librosa, _ = extract_pitch_librosa(audio, sr, method='pyin', voicing_threshold=0.3)
        
        # Both should detect voiced frames
        voiced_praat = ~np.isnan(f0_praat)
        voiced_librosa = ~np.isnan(f0_librosa)
        
        assert np.sum(voiced_praat) > 0
        assert np.sum(voiced_librosa) > 0
        
        # Methods should have similar output lengths (within hop size differences)
        length_diff = abs(len(f0_praat) - len(f0_librosa))
        max_expected_diff = 5  # Allow some difference due to implementation details
        assert length_diff <= max_expected_diff
    
    def test_parameter_validation(self):
        """Test parameter validation across methods."""
        # Test invalid parameters
        with pytest.raises((ValueError, PitchExtractionError)):
            PraatExtractor(frame_sec=-1.0)  # Negative frame size
        
        with pytest.raises((ValueError, PitchExtractionError)):
            LibrosaExtractor(hop_sec=0.0)  # Zero hop size
    
    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="Librosa not available")
    def test_edge_case_audio_lengths(self):
        """Test extraction with various audio lengths."""
        sr = 22050
        
        # Very short audio
        short_audio = np.sin(2 * np.pi * 220 * np.linspace(0, 0.1, int(sr * 0.1)))
        f0_short, _ = extract_pitch_librosa(short_audio, sr, voicing_threshold=0.3)
        assert len(f0_short) >= 0  # Should not crash
        
        # Single sample
        single_sample = np.array([0.5])
        try:
            f0_single, _ = extract_pitch_librosa(single_sample, sr)
            # If it doesn't raise an error, check the output
            assert isinstance(f0_single, np.ndarray)
        except PitchExtractionError:
            # It's acceptable to fail on single sample
            pass


if __name__ == "__main__":
    pytest.main([__file__])