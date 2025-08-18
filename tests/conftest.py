"""
Pytest configuration and shared fixtures for RaagHMM tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from raag_hmm.config import reset_config


@pytest.fixture(autouse=True)
def reset_configuration():
    """Reset configuration to defaults before each test."""
    reset_config()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    sr = 22050
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration))
    # Simple sine wave at 440 Hz (A4)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    return {
        "recording_id": "test_001",
        "raag": "Yaman",
        "tonic_hz": 261.63,
        "artist": "Test Artist",
        "instrument": "Sitar",
        "split": "train"
    }


@pytest.fixture
def sample_pitch_sequence():
    """Generate sample pitch sequence for testing."""
    # Create a simple melodic pattern
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
    sequence = []
    for freq in frequencies:
        sequence.extend([freq] * 10)  # Hold each note for 10 frames
    return np.array(sequence)


@pytest.fixture
def sample_quantized_sequence():
    """Generate sample quantized sequence for testing."""
    # Simple chromatic pattern: C4, C#4, D4, D#4, E4, F4, F#4, G4
    return np.array([12, 13, 14, 15, 16, 17, 18, 19] * 5)  # Repeat pattern


@pytest.fixture
def mock_dataset_structure(temp_dir):
    """Create a mock dataset structure for testing."""
    dataset_dir = temp_dir / "dataset"
    
    # Create train and test directories
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    # Create sample metadata files
    raags = ["Yaman", "Bihag", "Darbari"]
    
    for split, split_dir in [("train", train_dir), ("test", test_dir)]:
        for i, raag in enumerate(raags):
            # Create metadata file
            metadata = {
                "recording_id": f"{split}_{raag}_{i:03d}",
                "raag": raag,
                "tonic_hz": 261.63 + i * 10,  # Vary tonic slightly
                "split": split
            }
            
            metadata_file = split_dir / f"{metadata['recording_id']}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
    
    return dataset_dir