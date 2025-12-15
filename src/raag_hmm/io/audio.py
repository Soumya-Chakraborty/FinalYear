"""
Audio loading functionality with format support and resampling.

This module provides functions and classes for loading audio files in various formats
(wav, flac, mp3) with automatic resampling to the target sample rate.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import warnings

from ..exceptions import AudioProcessingError
from ..logger import get_logger

logger = get_logger(__name__)


def load_audio(path: str, sr: int = 22050) -> np.ndarray:
    """
    Load audio file with automatic resampling to target sample rate.
    
    Supports wav, flac, mp3 formats with automatic format detection.
    Always returns mono audio resampled to the specified sample rate.
    
    Args:
        path: Path to audio file
        sr: Target sample rate (default: 22050 Hz)
        
    Returns:
        Audio signal as numpy array (mono, resampled)
        
    Raises:
        AudioProcessingError: If file cannot be loaded or format unsupported
    """
    try:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")
            
        logger.debug(f"Loading audio file: {path}")
        
        # Use librosa for robust audio loading with automatic resampling
        y, original_sr = librosa.load(path, sr=None, mono=True)
        
        if original_sr != sr:
            logger.debug(f"Resampling from {original_sr} Hz to {sr} Hz")
            y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
            
        logger.debug(f"Loaded audio: {len(y)} samples at {sr} Hz ({len(y)/sr:.2f}s)")
        return y
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Failed to load audio file {path}: {str(e)}")


class AudioLoader:
    """
    Audio loading class with configurable parameters and format validation.
    
    Provides more control over audio loading process with support for
    different sample rates and format validation.
    """
    
    SUPPORTED_FORMATS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
    
    def __init__(self, target_sr: int = 22050, mono: bool = True):
        """
        Initialize AudioLoader with target parameters.
        
        Args:
            target_sr: Target sample rate for resampling
            mono: Whether to convert to mono (default: True)
        """
        self.target_sr = target_sr
        self.mono = mono
        logger.debug(f"AudioLoader initialized: sr={target_sr}, mono={mono}")
    
    def load(self, path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with format validation and resampling.
        
        Args:
            path: Path to audio file
            
        Returns:
            Tuple of (audio_signal, sample_rate)
            
        Raises:
            AudioProcessingError: If format unsupported or loading fails
        """
        path = Path(path)
        
        # Validate file format
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise AudioProcessingError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            # Load with librosa for robust format support
            y, original_sr = librosa.load(path, sr=None, mono=self.mono)
            
            # Resample if needed
            if original_sr != self.target_sr:
                logger.debug(f"Resampling {path.name}: {original_sr} -> {self.target_sr} Hz")
                y = librosa.resample(y, orig_sr=original_sr, target_sr=self.target_sr)
            
            # Validate audio content
            if len(y) == 0:
                raise AudioProcessingError(f"Audio file is empty: {path}")
                
            if np.all(y == 0):
                warnings.warn(f"Audio file contains only silence: {path}")
                
            logger.info(f"Loaded {path.name}: {len(y)} samples, {len(y)/self.target_sr:.2f}s")
            return y, self.target_sr
            
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            raise AudioProcessingError(f"Failed to load {path}: {str(e)}")
    
    def validate_format(self, path: str) -> bool:
        """
        Check if file format is supported without loading.
        
        Args:
            path: Path to audio file
            
        Returns:
            True if format is supported, False otherwise
        """
        return Path(path).suffix.lower() in self.SUPPORTED_FORMATS
    
    def get_info(self, path: str) -> dict:
        """
        Get audio file information without loading full content.
        
        Args:
            path: Path to audio file
            
        Returns:
            Dictionary with file information
            
        Raises:
            AudioProcessingError: If file cannot be read
        """
        try:
            path = Path(path)
            
            # Use soundfile for metadata extraction (faster than loading)
            info = sf.info(str(path))
            
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format': info.format,
                'subtype': info.subtype,
                'file_size': path.stat().st_size
            }
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to get info for {path}: {str(e)}")