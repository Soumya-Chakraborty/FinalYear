"""
Unit tests for audio I/O functionality.

Tests audio loading, format support, resampling, and error handling.
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

from raag_hmm.io.audio import load_audio, AudioLoader
from raag_hmm.exceptions import AudioProcessingError


class TestLoadAudio:
    """Test the load_audio function."""
    
    def test_load_audio_basic(self, tmp_path):
        """Test basic audio loading functionality."""
        # Create test audio file
        audio_path = tmp_path / "test.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 1 second 440Hz
        sf.write(audio_path, test_signal, 22050)
        
        # Load audio
        loaded = load_audio(str(audio_path))
        
        assert isinstance(loaded, np.ndarray)
        assert len(loaded) == 22050  # 1 second at 22050 Hz
        assert loaded.ndim == 1  # Mono
    
    def test_load_audio_resampling(self, tmp_path):
        """Test automatic resampling functionality."""
        # Create test audio at different sample rate
        audio_path = tmp_path / "test_44k.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))  # 1 second at 44.1kHz
        sf.write(audio_path, test_signal, 44100)
        
        # Load with target sample rate
        loaded = load_audio(str(audio_path), sr=22050)
        
        assert len(loaded) == 22050  # Resampled to 22050 Hz
    
    def test_load_audio_custom_sr(self, tmp_path):
        """Test loading with custom sample rate."""
        audio_path = tmp_path / "test.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        sf.write(audio_path, test_signal, 16000)
        
        loaded = load_audio(str(audio_path), sr=16000)
        assert len(loaded) == 16000
    
    def test_load_audio_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(AudioProcessingError, match="Audio file not found"):
            load_audio("nonexistent.wav")
    
    def test_load_audio_invalid_format(self, tmp_path):
        """Test error handling for invalid audio files."""
        # Create invalid audio file
        invalid_path = tmp_path / "invalid.wav"
        invalid_path.write_text("not audio data")
        
        with pytest.raises(AudioProcessingError, match="Failed to load audio file"):
            load_audio(str(invalid_path))


class TestAudioLoader:
    """Test the AudioLoader class."""
    
    def test_audio_loader_init(self):
        """Test AudioLoader initialization."""
        loader = AudioLoader(target_sr=16000, mono=False)
        assert loader.target_sr == 16000
        assert loader.mono == False
    
    def test_audio_loader_load(self, tmp_path):
        """Test AudioLoader.load method."""
        # Create test audio
        audio_path = tmp_path / "test.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        sf.write(audio_path, test_signal, 22050)
        
        loader = AudioLoader(target_sr=22050)
        audio, sr = loader.load(str(audio_path))
        
        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        assert len(audio) == 22050
    
    def test_audio_loader_resampling(self, tmp_path):
        """Test AudioLoader resampling."""
        # Create test audio at 44.1kHz
        audio_path = tmp_path / "test_44k.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        sf.write(audio_path, test_signal, 44100)
        
        loader = AudioLoader(target_sr=22050)
        audio, sr = loader.load(str(audio_path))
        
        assert sr == 22050
        assert len(audio) == 22050  # Resampled
    
    def test_validate_format_supported(self):
        """Test format validation for supported formats."""
        loader = AudioLoader()
        
        assert loader.validate_format("test.wav") == True
        assert loader.validate_format("test.flac") == True
        assert loader.validate_format("test.mp3") == True
        assert loader.validate_format("test.WAV") == True  # Case insensitive
    
    def test_validate_format_unsupported(self):
        """Test format validation for unsupported formats."""
        loader = AudioLoader()
        
        assert loader.validate_format("test.txt") == False
        assert loader.validate_format("test.pdf") == False
        assert loader.validate_format("test") == False  # No extension
    
    def test_load_unsupported_format(self, tmp_path):
        """Test error handling for unsupported formats."""
        # Create file with unsupported extension
        invalid_path = tmp_path / "test.txt"
        invalid_path.write_text("not audio")
        
        loader = AudioLoader()
        with pytest.raises(AudioProcessingError, match="Unsupported audio format"):
            loader.load(str(invalid_path))
    
    def test_load_empty_audio(self, tmp_path):
        """Test error handling for empty audio files."""
        # Create empty audio file
        audio_path = tmp_path / "empty.wav"
        sf.write(audio_path, np.array([]), 22050)
        
        loader = AudioLoader()
        with pytest.raises(AudioProcessingError, match="Audio file is empty"):
            loader.load(str(audio_path))
    
    def test_load_silent_audio(self, tmp_path):
        """Test handling of silent audio (should warn but not fail)."""
        # Create silent audio file
        audio_path = tmp_path / "silent.wav"
        silent_signal = np.zeros(22050)  # 1 second of silence
        sf.write(audio_path, silent_signal, 22050)
        
        loader = AudioLoader()
        with pytest.warns(UserWarning, match="contains only silence"):
            audio, sr = loader.load(str(audio_path))
        
        assert len(audio) == 22050
        assert np.all(audio == 0)
    
    def test_get_info(self, tmp_path):
        """Test audio file info extraction."""
        # Create test audio
        audio_path = tmp_path / "test.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 44100))  # 2 seconds
        sf.write(audio_path, test_signal, 22050)
        
        loader = AudioLoader()
        info = loader.get_info(str(audio_path))
        
        assert 'duration' in info
        assert 'sample_rate' in info
        assert 'channels' in info
        assert 'frames' in info
        assert info['sample_rate'] == 22050
        assert info['channels'] == 1
        assert abs(info['duration'] - 2.0) < 0.1  # Approximately 2 seconds
    
    def test_get_info_invalid_file(self, tmp_path):
        """Test error handling for invalid files in get_info."""
        invalid_path = tmp_path / "invalid.txt"
        invalid_path.write_text("not audio")
        
        loader = AudioLoader()
        with pytest.raises(AudioProcessingError, match="Failed to get info"):
            loader.get_info(str(invalid_path))


class TestAudioFormats:
    """Test various audio format support."""
    
    @pytest.mark.parametrize("format_ext,sf_format", [
        (".wav", "WAV"),
        (".flac", "FLAC"),
    ])
    def test_format_support(self, tmp_path, format_ext, sf_format):
        """Test loading different audio formats."""
        # Create test audio in specified format
        audio_path = tmp_path / f"test{format_ext}"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        
        try:
            sf.write(audio_path, test_signal, 22050, format=sf_format)
        except Exception:
            pytest.skip(f"Format {sf_format} not supported by soundfile")
        
        # Test with load_audio function
        loaded = load_audio(str(audio_path))
        assert len(loaded) == 22050
        
        # Test with AudioLoader class
        loader = AudioLoader()
        audio, sr = loader.load(str(audio_path))
        assert len(audio) == 22050
        assert sr == 22050


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_short_audio(self, tmp_path):
        """Test loading very short audio files."""
        audio_path = tmp_path / "short.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, 220))  # 10ms
        sf.write(audio_path, test_signal, 22050)
        
        loaded = load_audio(str(audio_path))
        assert len(loaded) == 220
    
    def test_very_long_audio_info(self, tmp_path):
        """Test getting info for long audio without loading."""
        # Create longer audio file
        audio_path = tmp_path / "long.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 10, 220500))  # 10 seconds
        sf.write(audio_path, test_signal, 22050)
        
        loader = AudioLoader()
        info = loader.get_info(str(audio_path))
        
        assert abs(info['duration'] - 10.0) < 0.1
        assert info['frames'] == 220500
    
    def test_extreme_resampling(self, tmp_path):
        """Test extreme resampling ratios."""
        # Create audio at very different sample rate
        audio_path = tmp_path / "test_8k.wav"
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 8000))
        sf.write(audio_path, test_signal, 8000)
        
        # Resample to much higher rate
        loaded = load_audio(str(audio_path), sr=48000)
        assert len(loaded) == 48000  # 6x upsampling