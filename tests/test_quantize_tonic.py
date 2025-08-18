"""
Unit tests for tonic normalization utilities.
"""

import pytest
import numpy as np
from raag_hmm.quantize.tonic import normalize_by_tonic, TonicNormalizer
from raag_hmm.quantize.sequence import quantize_sequence
from raag_hmm.exceptions import QuantizationError


class TestNormalizeByTonic:
    """Test tonic normalization function."""
    
    def test_tonic_to_c4(self):
        """Test that tonic frequency is normalized to C4."""
        C4_HZ = 261.63
        
        # Various tonic frequencies should normalize to C4
        assert abs(normalize_by_tonic(220.0, 220.0) - C4_HZ) < 0.01  # A3 tonic
        assert abs(normalize_by_tonic(246.94, 246.94) - C4_HZ) < 0.01  # B3 tonic
        assert abs(normalize_by_tonic(293.66, 293.66) - C4_HZ) < 0.01  # D4 tonic
    
    def test_proportional_scaling(self):
        """Test that all frequencies are scaled proportionally."""
        tonic_hz = 220.0  # A3
        C4_HZ = 261.63
        scale_factor = C4_HZ / tonic_hz
        
        # Test multiple frequencies
        freqs = np.array([220.0, 440.0, 880.0])  # A3, A4, A5
        normalized = normalize_by_tonic(freqs, tonic_hz)
        expected = freqs * scale_factor
        
        np.testing.assert_allclose(normalized, expected, rtol=1e-10)
    
    def test_octave_relationships_preserved(self):
        """Test that octave relationships are preserved after normalization."""
        tonic_hz = 220.0  # A3
        
        # Test octave pairs
        f1 = 220.0  # A3
        f2 = 440.0  # A4 (octave higher)
        
        norm1 = normalize_by_tonic(f1, tonic_hz)
        norm2 = normalize_by_tonic(f2, tonic_hz)
        
        # Should maintain 2:1 ratio
        assert abs(norm2 / norm1 - 2.0) < 1e-10
    
    def test_array_input(self):
        """Test normalization with array input."""
        tonic_hz = 220.0
        freqs = np.array([220.0, 246.94, 261.63, 293.66])
        normalized = normalize_by_tonic(freqs, tonic_hz)
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(freqs)
        
        # First frequency (tonic) should become C4
        assert abs(normalized[0] - 261.63) < 0.01
    
    def test_scalar_return_type(self):
        """Test that scalar input returns scalar output."""
        result = normalize_by_tonic(220.0, 220.0)
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)
    
    def test_negative_tonic_error(self):
        """Test error handling for negative tonic frequency."""
        with pytest.raises(QuantizationError, match="Tonic frequency must be positive"):
            normalize_by_tonic(440.0, -220.0)
    
    def test_zero_tonic_error(self):
        """Test error handling for zero tonic frequency."""
        with pytest.raises(QuantizationError, match="Tonic frequency must be positive"):
            normalize_by_tonic(440.0, 0.0)
    
    def test_tonic_range_validation(self):
        """Test validation of tonic frequency range."""
        # Too low
        with pytest.raises(QuantizationError, match="outside valid range"):
            normalize_by_tonic(440.0, 50.0)
        
        # Too high
        with pytest.raises(QuantizationError, match="outside valid range"):
            normalize_by_tonic(440.0, 1000.0)
    
    def test_negative_frequency_error(self):
        """Test error handling for negative frequencies."""
        with pytest.raises(QuantizationError, match="All frequencies must be positive"):
            normalize_by_tonic(-440.0, 220.0)
    
    def test_array_with_negative_frequencies(self):
        """Test error handling for arrays with negative frequencies."""
        freqs = np.array([220.0, -440.0, 880.0])
        with pytest.raises(QuantizationError, match="All frequencies must be positive"):
            normalize_by_tonic(freqs, 220.0)


class TestTonicNormalizer:
    """Test TonicNormalizer class."""
    
    def test_default_initialization(self):
        """Test default normalizer initialization."""
        normalizer = TonicNormalizer()
        assert normalizer.reference_hz == 261.63  # C4
    
    def test_custom_reference(self):
        """Test normalizer with custom reference frequency."""
        normalizer = TonicNormalizer(reference_hz=440.0)  # A4 reference
        assert normalizer.reference_hz == 440.0
    
    def test_invalid_reference_error(self):
        """Test error handling for invalid reference frequency."""
        with pytest.raises(QuantizationError, match="Reference frequency must be positive"):
            TonicNormalizer(reference_hz=-261.63)
    
    def test_normalize_method(self):
        """Test normalization method."""
        normalizer = TonicNormalizer()
        
        # Should match standalone function
        freq = 440.0
        tonic = 220.0
        
        result1 = normalizer.normalize(freq, tonic)
        result2 = normalize_by_tonic(freq, tonic)
        
        assert abs(result1 - result2) < 1e-10
    
    def test_caching_behavior(self):
        """Test that scale factors are cached for repeated calls."""
        normalizer = TonicNormalizer()
        
        # First call should cache the tonic
        result1 = normalizer.normalize(440.0, 220.0)
        assert normalizer._cached_tonic == 220.0
        assert normalizer._cached_scale_factor is not None
        
        # Second call with same tonic should use cache
        result2 = normalizer.normalize(880.0, 220.0)
        assert result2 == result1 * 2  # Proportional scaling
    
    def test_cache_invalidation(self):
        """Test that cache is invalidated when tonic changes."""
        normalizer = TonicNormalizer()
        
        # First tonic
        normalizer.normalize(440.0, 220.0)
        old_scale = normalizer._cached_scale_factor
        
        # Different tonic should update cache
        normalizer.normalize(440.0, 440.0)
        new_scale = normalizer._cached_scale_factor
        
        assert old_scale != new_scale
        assert normalizer._cached_tonic == 440.0
    
    def test_get_scale_factor(self):
        """Test scale factor computation."""
        normalizer = TonicNormalizer(reference_hz=261.63)
        
        # Scale factor should be reference / tonic
        scale = normalizer.get_scale_factor(220.0)
        expected = 261.63 / 220.0
        assert abs(scale - expected) < 1e-10
    
    def test_denormalize_method(self):
        """Test denormalization (reverse operation)."""
        normalizer = TonicNormalizer()
        
        # Round-trip should recover original
        original = 440.0
        tonic = 220.0
        
        normalized = normalizer.normalize(original, tonic)
        recovered = normalizer.denormalize(normalized, tonic)
        
        assert abs(recovered - original) < 1e-10
    
    def test_array_operations(self):
        """Test normalizer with array inputs."""
        normalizer = TonicNormalizer()
        
        freqs = np.array([220.0, 440.0, 880.0])
        tonic = 220.0
        
        normalized = normalizer.normalize(freqs, tonic)
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(freqs)
        
        # First element (tonic) should become reference frequency
        assert abs(normalized[0] - normalizer.reference_hz) < 0.01


class TestQuantizeSequence:
    """Test sequence quantization function."""
    
    def test_basic_quantization(self):
        """Test basic sequence quantization."""
        # Simple sequence with known tonic
        freqs = np.array([261.63, 293.66, 329.63])  # C4, D4, E4
        tonic_hz = 261.63  # C4 tonic
        
        quantized = quantize_sequence(freqs, tonic_hz)
        
        # Should be integer array
        assert quantized.dtype == int
        assert len(quantized) == len(freqs)
        
        # C4 should map to bin 12 (C4 - C3 = 12 semitones)
        assert quantized[0] == 12
    
    def test_tonic_normalization_effect(self):
        """Test that tonic normalization affects quantization."""
        freqs = np.array([220.0, 246.94, 261.63])  # A3, B3, C4
        
        # With A3 tonic vs C4 tonic should give different results
        quant_a3 = quantize_sequence(freqs, tonic_hz=220.0)
        quant_c4 = quantize_sequence(freqs, tonic_hz=261.63)
        
        # Results should be different due to normalization
        assert not np.array_equal(quant_a3, quant_c4)
    
    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        empty_freqs = np.array([])
        result = quantize_sequence(empty_freqs, tonic_hz=261.63)
        
        assert len(result) == 0
        assert result.dtype == int
    
    def test_custom_parameters(self):
        """Test quantization with custom parameters."""
        freqs = np.array([261.63, 293.66])
        
        # Custom bin count and base MIDI
        result = quantize_sequence(freqs, tonic_hz=261.63, n_bins=24, base_midi=60)
        
        assert len(result) == len(freqs)
        assert np.all(result >= 0)
        assert np.all(result < 24)
    
    def test_multidimensional_error(self):
        """Test error handling for multidimensional input."""
        freqs_2d = np.array([[261.63, 293.66], [329.63, 349.23]])
        
        with pytest.raises(QuantizationError, match="must be 1-dimensional"):
            quantize_sequence(freqs_2d, tonic_hz=261.63)
    
    def test_invalid_tonic_propagation(self):
        """Test that tonic validation errors are propagated."""
        freqs = np.array([261.63, 293.66])
        
        with pytest.raises(QuantizationError, match="Tonic frequency must be positive"):
            quantize_sequence(freqs, tonic_hz=-261.63)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_extreme_tonic_frequencies(self):
        """Test normalization with extreme but valid tonic frequencies."""
        normalizer = TonicNormalizer()
        
        # Very low tonic (but within range)
        result_low = normalizer.normalize(440.0, 80.0)
        assert result_low > 0
        
        # Very high tonic (but within range)
        result_high = normalizer.normalize(440.0, 800.0)
        assert result_high > 0
    
    def test_precision_preservation(self):
        """Test that normalization preserves numerical precision."""
        normalizer = TonicNormalizer()
        
        # Use high-precision frequencies
        freq = 440.123456789
        tonic = 220.987654321
        
        normalized = normalizer.normalize(freq, tonic)
        denormalized = normalizer.denormalize(normalized, tonic)
        
        # Should recover original with high precision
        assert abs(denormalized - freq) < 1e-10
    
    def test_sequence_consistency(self):
        """Test that sequence quantization is consistent with individual calls."""
        freqs = np.array([261.63, 293.66, 329.63])
        tonic = 261.63
        
        # Quantize as sequence
        seq_result = quantize_sequence(freqs, tonic)
        
        # Quantize individually
        from raag_hmm.quantize.chromatic import ChromaticQuantizer
        from raag_hmm.quantize.tonic import normalize_by_tonic
        
        quantizer = ChromaticQuantizer()
        individual_results = []
        for freq in freqs:
            normalized = normalize_by_tonic(freq, tonic)
            quantized = quantizer.quantize(normalized)
            individual_results.append(quantized)
        
        np.testing.assert_array_equal(seq_result, individual_results)