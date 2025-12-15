"""
Test configuration and fixtures for RaagHMM.

This file contains pytest configuration and shared fixtures
for testing the RaagHMM system.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_audio_data():
    """Create mock audio data for testing."""
    # Create a simple sine wave at 440 Hz (A4)
    sr = 22050
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    frequency = 440.0
    audio_data = np.sin(2 * np.pi * frequency * t)
    return audio_data, sr


@pytest.fixture
def mock_pitch_data():
    """Create mock pitch data for testing."""
    # Create mock pitch array with some NaN values (unvoiced frames)
    f0_hz = np.array([220.0, 246.9, 261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9, 523.2])
    voicing_prob = np.array([0.9, 0.95, 0.8, 0.9, 0.85, 0.9, 0.8, 0.9, 0.85, 0.9])
    
    # Add some unvoiced frames (NaN in pitch)
    f0_hz_with_nan = f0_hz.copy()
    f0_hz_with_nan[2::3] = np.nan  # Every 3rd frame is unvoiced
    
    return f0_hz_with_nan, voicing_prob


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'recording_id': 'test_recording_001',
        'raag': 'Bihag',
        'tonic_hz': 261.63,
        'artist': 'Test Artist',
        'instrument': 'sitar',
        'split': 'test'
    }


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )