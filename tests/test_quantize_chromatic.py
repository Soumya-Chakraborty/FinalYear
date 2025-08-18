"""
Unit tests for chromatic quantization utilities.
"""

import pytest
import numpy as np
from raag_hmm.quantize.chromatic import (
    hz_to_midi, 
    nearest_chromatic_bin, 
    ChromaticQuantizer
)
from raag_hmm.exceptions import QuantizationError


class TestHzToMidi:
    """Test frequency to MIDI conversion."""
    
    def test_standard_frequencies(self):
        """Test conversion of standard musical frequencies."""
        # A4 = 440 Hz = MIDI 69
        assert abs(hz_to_midi(440.0) - 69.0) < 1e-10
        
        # C4 = 261.63 Hz ≈ MIDI 60
        assert abs(hz_to_midi(261.63) - 60.0) < 0.01
        
        # C3 = 130.81 Hz ≈ MIDI 48
        assert abs(hz_to_midi(130.81) - 48.0) < 0.01
        
        # B5 = 987.77 Hz ≈ MIDI 83
        assert abs(hz_to_midi(987.77) - 83.0) < 0.01
    
    def test_array_input(self):
        """Test conversion with numpy array input."""
        freqs = np.array([440.0, 261.63, 130.81])
        midis = hz_to_midi(freqs)
        
        assert isinstance(midis, np.ndarray)
        assert len(midis) == 3
        assert abs(midis[0] - 69.0) < 1e-10
        assert abs(midis[1] - 60.0) < 0.01
        assert abs(midis[2] - 48.0) < 0.01
    
    def test_scalar_return_type(self):
        """Test that scalar input returns scalar output."""
        result = hz_to_midi(440.0)
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)
    
    def test_negative_frequency_error(self):
        """Test error handling for negative frequencies."""
        with pytest.raises(QuantizationError, match="Frequency must be positive"):
            hz_to_midi(-100.0)
    
    def test_zero_frequency_error(self):
        """Test error handling for zero frequency."""
        with pytest.raises(QuantizationError, match="Frequency must be positive"):
            hz_to_midi(0.0)
    
    def test_array_with_invalid_values(self):
        """Test error handling for arrays containing invalid values."""
        with pytest.raises(QuantizationError):
            hz_to_midi(np.array([440.0, -100.0, 261.63]))


class TestNearestChromaticBin:
    """Test chromatic bin quantization."""
    
    def test_exact_midi_values(self):
        """Test quantization of exact MIDI note values."""
        # C3 = MIDI 48 = bin 0
        assert nearest_chromatic_bin(48.0) == 0
        
        # C4 = MIDI 60 = bin 12
        assert nearest_chromatic_bin(60.0) == 12
        
        # B5 = MIDI 83 = bin 35
        assert nearest_chromatic_bin(83.0) == 35
    
    def test_fractional_midi_rounding(self):
        """Test rounding of fractional MIDI values."""
        # Should round to nearest integer
        assert nearest_chromatic_bin(48.3) == 0  # Rounds down
        assert nearest_chromatic_bin(48.7) == 1  # Rounds up
        assert nearest_chromatic_bin(60.5) == 12  # 60.5 - 48 = 12.5, rounds to 12
    
    def test_boundary_clamping(self):
        """Test clamping of values outside valid range."""
        # Below range should clamp to 0
        assert nearest_chromatic_bin(40.0) == 0
        assert nearest_chromatic_bin(-10.0) == 0
        
        # Above range should clamp to 35
        assert nearest_chromatic_bin(90.0) == 35
        assert nearest_chromatic_bin(100.0) == 35
    
    def test_array_input(self):
        """Test quantization with array input."""
        midis = np.array([48.0, 60.0, 83.0, 40.0, 90.0])
        bins = nearest_chromatic_bin(midis)
        
        assert isinstance(bins, np.ndarray)
        np.testing.assert_array_equal(bins, [0, 12, 35, 0, 35])
    
    def test_scalar_return_type(self):
        """Test that scalar input returns scalar output."""
        result = nearest_chromatic_bin(60.0)
        assert isinstance(result, int)
        assert not isinstance(result, np.ndarray)
    
    def test_custom_base_midi(self):
        """Test quantization with custom base MIDI note."""
        # Use C4 (MIDI 60) as base instead of C3 (MIDI 48)
        assert nearest_chromatic_bin(60.0, base_midi=60) == 0
        assert nearest_chromatic_bin(72.0, base_midi=60) == 12


class TestChromaticQuantizer:
    """Test ChromaticQuantizer class."""
    
    def test_default_initialization(self):
        """Test default quantizer initialization."""
        quantizer = ChromaticQuantizer()
        assert quantizer.n_bins == 36
        assert quantizer.base_midi == 48
        assert quantizer.max_midi == 83
    
    def test_custom_initialization(self):
        """Test quantizer with custom parameters."""
        quantizer = ChromaticQuantizer(n_bins=24, base_midi=60)
        assert quantizer.n_bins == 24
        assert quantizer.base_midi == 60
        assert quantizer.max_midi == 83
    
    def test_invalid_initialization(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(QuantizationError, match="Number of bins must be positive"):
            ChromaticQuantizer(n_bins=0)
        
        with pytest.raises(QuantizationError, match="Base MIDI note must be non-negative"):
            ChromaticQuantizer(base_midi=-1)
    
    def test_quantize_method(self):
        """Test frequency quantization method."""
        quantizer = ChromaticQuantizer()
        
        # Test standard frequencies
        assert quantizer.quantize(130.81) == 0  # C3
        assert quantizer.quantize(261.63) == 12  # C4
        assert quantizer.quantize(987.77) == 35  # B5
    
    def test_quantize_array(self):
        """Test quantization of frequency arrays."""
        quantizer = ChromaticQuantizer()
        freqs = np.array([130.81, 261.63, 987.77])
        bins = quantizer.quantize(freqs)
        
        np.testing.assert_array_equal(bins, [0, 12, 35])
    
    def test_get_bin_frequency(self):
        """Test getting center frequency for bin indices."""
        quantizer = ChromaticQuantizer()
        
        # Test exact frequencies for known bins
        f0 = quantizer.get_bin_frequency(0)  # C3
        assert abs(f0 - 130.81) < 0.1
        
        f12 = quantizer.get_bin_frequency(12)  # C4
        assert abs(f12 - 261.63) < 0.1
    
    def test_get_bin_frequency_out_of_range(self):
        """Test error handling for invalid bin indices."""
        quantizer = ChromaticQuantizer()
        
        with pytest.raises(QuantizationError, match="Bin index -1 out of range"):
            quantizer.get_bin_frequency(-1)
        
        with pytest.raises(QuantizationError, match="Bin index 36 out of range"):
            quantizer.get_bin_frequency(36)
    
    def test_get_bin_boundaries(self):
        """Test getting frequency boundaries for all bins."""
        quantizer = ChromaticQuantizer()
        boundaries = quantizer.get_bin_boundaries()
        
        # Should have n_bins + 1 boundaries
        assert len(boundaries) == 37
        
        # Boundaries should be monotonically increasing
        assert np.all(np.diff(boundaries) > 0)
        
        # First boundary should be below C3, last above B5
        assert boundaries[0] < 130.81
        assert boundaries[-1] > 987.77


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_high_frequencies(self):
        """Test handling of very high frequencies."""
        quantizer = ChromaticQuantizer()
        
        # Should clamp to highest bin
        assert quantizer.quantize(10000.0) == 35
    
    def test_very_low_frequencies(self):
        """Test handling of very low frequencies."""
        quantizer = ChromaticQuantizer()
        
        # Should clamp to lowest bin
        assert quantizer.quantize(50.0) == 0
    
    def test_frequency_precision(self):
        """Test precision of frequency conversions."""
        quantizer = ChromaticQuantizer()
        
        # Round-trip conversion should be accurate
        for bin_idx in [0, 12, 24, 35]:
            freq = quantizer.get_bin_frequency(bin_idx)
            recovered_bin = quantizer.quantize(freq)
            assert recovered_bin == bin_idx
    
    def test_empty_array_input(self):
        """Test handling of empty arrays."""
        empty_freqs = np.array([])
        
        midi_result = hz_to_midi(empty_freqs)
        assert len(midi_result) == 0
        
        bin_result = nearest_chromatic_bin(empty_freqs)
        assert len(bin_result) == 0