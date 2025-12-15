"""
Unit tests for the quantize module.

Tests for pitch quantization and normalization functionality.
"""

import numpy as np
import pytest
from src.raag_hmm.quantize.chromatic import hz_to_midi, nearest_chromatic_bin, ChromaticQuantizer
from src.raag_hmm.quantize.tonic import normalize_by_tonic, TonicNormalizer
from src.raag_hmm.quantize.sequence import quantize_sequence


class TestChromaticQuantization:
    """Test cases for chromatic quantization functionality."""
    
    def test_hz_to_midi_basic(self):
        """Test basic Hz to MIDI conversion."""
        # Test A4 (440 Hz) -> MIDI 69
        assert abs(hz_to_midi(440.0) - 69.0) < 0.01
        
        # Test C4 (261.63 Hz) -> MIDI 60
        assert abs(hz_to_midi(261.63) - 60.0) < 0.01
        
        # Test C3 (130.81 Hz) -> MIDI 48
        assert abs(hz_to_midi(130.81) - 48.0) < 0.01
    
    def test_hz_to_midi_array(self):
        """Test Hz to MIDI conversion for arrays."""
        freqs = np.array([440.0, 261.63, 130.81])
        expected_midi = np.array([69.0, 60.0, 48.0])
        
        midi_vals = hz_to_midi(freqs)
        
        np.testing.assert_array_almost_equal(midi_vals, expected_midi, decimal=2)
    
    def test_hz_to_midi_invalid_input(self):
        """Test Hz to MIDI conversion with invalid input."""
        with pytest.raises(Exception):  # May raise ValueError or QuantizationError
            hz_to_midi(0.0)  # Non-positive frequency
        
        with pytest.raises(Exception):
            hz_to_midi(-100.0)  # Negative frequency
    
    def test_nearest_chromatic_bin_basic(self):
        """Test basic chromatic bin quantization."""
        # MIDI 60 (C4) with base 48 should give bin 12
        assert nearest_chromatic_bin(60.0, base_midi=48) == 12
        
        # MIDI 48 (C3) with base 48 should give bin 0
        assert nearest_chromatic_bin(48.0, base_midi=48) == 0
        
        # MIDI 83 (B5) with base 48 should give bin 35
        assert nearest_chromatic_bin(83.0, base_midi=48) == 35
    
    def test_nearest_chromatic_bin_clamping(self):
        """Test chromatic bin clamping to valid range."""
        # Below range should be clamped to 0
        assert nearest_chromatic_bin(47.0, base_midi=48) == 0  # -1 clamped to 0
        
        # Above range should be clamped to 35 (for 36 bins)
        assert nearest_chromatic_bin(84.0, base_midi=48) == 35  # 36 clamped to 35
    
    def test_chromatic_quantizer_basic(self):
        """Test ChromaticQuantizer functionality."""
        quantizer = ChromaticQuantizer(n_bins=36, base_midi=48)
        
        # Test quantization
        freqs = np.array([261.63, 277.18, 293.66])  # C4, C#4/D♭4, D4
        bins = quantizer.quantize(freqs)
        
        # These should map to adjacent bins around C4 (bin 12)
        assert len(bins) == 3
        assert bins[0] == 12  # C4 -> bin 12
    
    def test_chromatic_quantizer_get_bin_frequency(self):
        """Test getting frequency for a specific bin."""
        quantizer = ChromaticQuantizer(n_bins=36, base_midi=48)
        
        # Bin 12 should correspond to C4 (261.63 Hz)
        c4_freq = quantizer.get_bin_frequency(12)
        assert abs(c4_freq - 261.63) < 0.1
    
    def test_chromatic_quantizer_invalid_bin(self):
        """Test error handling for invalid bin indices."""
        quantizer = ChromaticQuantizer(n_bins=36, base_midi=48)
        
        with pytest.raises(Exception):  # May raise QuantizationError
            quantizer.get_bin_frequency(-1)  # Negative bin
        
        with pytest.raises(Exception):
            quantizer.get_bin_frequency(36)  # Out of range bin


class TestTonicNormalization:
    """Test cases for tonic normalization functionality."""
    
    def test_normalize_by_tonic_basic(self):
        """Test basic tonic normalization."""
        # Normalize A4 (440 Hz) with tonic A3 (220 Hz) -> should become A5 (554.37 Hz)
        normalized = normalize_by_tonic(440.0, 220.0)
        expected = 261.63 * (440.0 / 220.0)  # C4 * (A4 / A3 ratio)
        assert abs(normalized - expected) < 0.1
        
        # Normalize tonic itself -> should become C4 (261.63 Hz)
        normalized = normalize_by_tonic(220.0, 220.0)
        assert abs(normalized - 261.63) < 0.1
    
    def test_normalize_by_tonic_array(self):
        """Test tonic normalization for arrays."""
        freqs = np.array([220.0, 330.0, 440.0])  # A3, E4, A4 with A3 tonic
        tonic = 220.0
        normalized = normalize_by_tonic(freqs, tonic)
        
        expected = np.array([261.63, 261.63 * 1.5, 261.63 * 2.0])  # C4, G4, C5
        np.testing.assert_array_almost_equal(normalized, expected, decimal=1)
    
    def test_normalize_by_tonic_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(Exception):  # May raise QuantizationError
            normalize_by_tonic(440.0, 0.0)  # Zero tonic
        
        with pytest.raises(Exception):
            normalize_by_tonic(440.0, -100.0)  # Negative tonic
        
        with pytest.raises(Exception):
            normalize_by_tonic(0.0, 220.0)  # Zero frequency
        
        with pytest.raises(Exception):
            normalize_by_tonic(440.0, 50.0)  # Too low tonic (out of range)
    
    def test_tonic_normalizer_basic(self):
        """Test TonicNormalizer functionality."""
        normalizer = TonicNormalizer()
        
        # Normalize frequencies
        freqs = np.array([220.0, 440.0])
        normalized = normalizer.normalize(freqs, tonic_hz=220.0)
        
        expected = np.array([261.63, 261.63 * 2.0])  # C4, C5
        np.testing.assert_array_almost_equal(normalized, expected, decimal=1)
    
    def test_tonic_normalizer_get_scale_factor(self):
        """Test scale factor calculation."""
        normalizer = TonicNormalizer()
        
        scale_factor = normalizer.get_scale_factor(220.0)  # A3
        expected_factor = 261.63 / 220.0
        assert abs(scale_factor - expected_factor) < 0.001
    
    def test_tonic_normalizer_denormalize(self):
        """Test denormalization functionality."""
        normalizer = TonicNormalizer()
        
        # First normalize
        normalized = normalizer.normalize(440.0, tonic_hz=220.0)
        
        # Then denormalize back
        denormalized = normalizer.denormalize(normalized, tonic_hz=220.0)
        
        # Should be approximately the same as original
        assert abs(denormalized - 440.0) < 0.1


class TestSequenceQuantization:
    """Test cases for sequence quantization functionality."""
    
    def test_quantize_sequence_basic(self):
        """Test basic sequence quantization."""
        # Create a simple pitch sequence in A3 range
        f0_hz = np.array([220.0, 247.0, 261.63, 293.7, 329.6])  # A3 to E4
        tonic_hz = 220.0  # A3
        
        quantized = quantize_sequence(f0_hz, tonic_hz, n_bins=36, base_midi=48)
        
        # Check that we get integer array of same length
        assert len(quantized) == len(f0_hz)
        assert quantized.dtype in [np.int32, np.int64, int]
        
        # First element (220.0 Hz) should be normalized to C4 range
        # C4 is bin 12 with base_midi 48
        assert 10 <= quantized[0] <= 15  # Allow some tolerance
    
    def test_quantize_sequence_with_nan(self):
        """Test sequence quantization with NaN values."""
        f0_hz = np.array([220.0, np.nan, 261.63, 293.7])
        tonic_hz = 220.0
        
        # This should handle NaN values appropriately
        with pytest.raises(Exception):  # NaN handling may vary by implementation
            quantize_sequence(f0_hz, tonic_hz, n_bins=36, base_midi=48)
    
    def test_quantize_sequence_empty(self):
        """Test sequence quantization with empty input."""
        f0_hz = np.array([])
        tonic_hz = 220.0
        
        quantized = quantize_sequence(f0_hz, tonic_hz, n_bins=36, base_midi=48)
        
        assert len(quantized) == 0
        assert quantized.dtype in [np.int32, np.int64, int]
    
    def test_quantize_sequence_invalid_params(self):
        """Test error handling for invalid parameters."""
        f0_hz = np.array([220.0, 247.0])
        tonic_hz = 0.0  # Invalid tonic
        
        with pytest.raises(Exception):  # May raise QuantizationError
            quantize_sequence(f0_hz, tonic_hz)
        
        # Test with negative frequency
        f0_hz_negative = np.array([-220.0])
        with pytest.raises(Exception):
            quantize_sequence(f0_hz_negative, 220.0)