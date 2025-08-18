"""
Integration tests for CLI dataset preparation commands.

Tests the dataset preparation, pitch extraction, and validation commands.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from typer.testing import CliRunner
import soundfile as sf

from raag_hmm.cli.main import app


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample dataset for testing."""
    # Create directory structure
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    
    # Create sample audio files (sine waves)
    sample_rate = 22050
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio files for different raags
    raags = ["bihag", "darbari", "desh"]
    
    for i, raag in enumerate(raags):
        # Create train files
        freq = 440 + i * 50  # Different frequencies
        audio_data = 0.5 * np.sin(2 * np.pi * freq * t)
        
        audio_file = train_dir / f"{raag}_train_{i}.wav"
        sf.write(str(audio_file), audio_data, sample_rate)
        
        # Create corresponding metadata
        metadata = {
            "recording_id": f"{raag}_train_{i}",
            "raag": raag,
            "tonic_hz": 261.63,
            "artist": "Test Artist",
            "split": "train"
        }
        
        metadata_file = train_dir / f"{raag}_train_{i}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create test files
        audio_file = test_dir / f"{raag}_test_{i}.wav"
        sf.write(str(audio_file), audio_data, sample_rate)
        
        metadata["recording_id"] = f"{raag}_test_{i}"
        metadata["split"] = "test"
        
        metadata_file = test_dir / f"{raag}_test_{i}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return tmp_path


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    sample_rate = 22050
    duration = 0.5  # 0.5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple sine wave
    freq = 440  # A4
    audio_data = 0.5 * np.sin(2 * np.pi * freq * t)
    
    audio_file = tmp_path / "test_audio.wav"
    sf.write(str(audio_file), audio_data, sample_rate)
    
    # Create metadata
    metadata = {
        "recording_id": "test_audio",
        "raag": "bihag",
        "tonic_hz": 261.63,
        "artist": "Test Artist"
    }
    
    metadata_file = tmp_path / "test_audio.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return audio_file, metadata_file


class TestDatasetPrepareCommand:
    """Test dataset prepare command."""
    
    def test_prepare_dataset_basic(self, cli_runner, sample_dataset, tmp_path):
        """Test basic dataset preparation."""
        output_dir = tmp_path / "output"
        
        result = cli_runner.invoke(app, [
            "dataset", "prepare",
            str(sample_dataset),
            str(output_dir),
            "--sample-rate", "22050"
        ])
        
        assert result.exit_code == 0
        assert "Dataset preparation completed" in result.stdout
        
        # Check output structure
        assert (output_dir / "train").exists()
        assert (output_dir / "test").exists()
        
        # Check files were copied
        train_files = list((output_dir / "train").glob("*.wav"))
        assert len(train_files) == 3  # 3 raags
        
        metadata_files = list((output_dir / "train").glob("*.json"))
        assert len(metadata_files) == 3
    
    def test_prepare_dataset_validation_only(self, cli_runner, sample_dataset):
        """Test dataset preparation in validation-only mode."""
        result = cli_runner.invoke(app, [
            "dataset", "prepare",
            str(sample_dataset),
            "/tmp/nonexistent",  # Output dir won't be created
            "--validate-only"
        ])
        
        assert result.exit_code == 0
        assert "Dataset preparation completed" in result.stdout
    
    def test_prepare_dataset_force_overwrite(self, cli_runner, sample_dataset, tmp_path):
        """Test dataset preparation with force overwrite."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # First run should fail without --force
        result = cli_runner.invoke(app, [
            "dataset", "prepare",
            str(sample_dataset),
            str(output_dir)
        ])
        
        assert result.exit_code == 1
        assert "already exists" in result.stdout
        
        # Second run with --force should succeed
        result = cli_runner.invoke(app, [
            "dataset", "prepare",
            str(sample_dataset),
            str(output_dir),
            "--force"
        ])
        
        assert result.exit_code == 0
    
    def test_prepare_dataset_missing_input(self, cli_runner, tmp_path):
        """Test dataset preparation with missing input directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"
        
        result = cli_runner.invoke(app, [
            "dataset", "prepare",
            str(nonexistent_dir),
            str(output_dir)
        ])
        
        assert result.exit_code == 1


class TestExtractPitchCommand:
    """Test extract-pitch command."""
    
    def test_extract_pitch_basic(self, cli_runner, sample_audio_file):
        """Test basic pitch extraction."""
        audio_file, metadata_file = sample_audio_file
        
        result = cli_runner.invoke(app, [
            "dataset", "extract-pitch",
            str(audio_file)
        ])
        
        assert result.exit_code == 0
        assert "Pitch Extraction" in result.stdout
        assert "f0_hz" in result.stdout
    
    def test_extract_pitch_with_output_file(self, cli_runner, sample_audio_file, tmp_path):
        """Test pitch extraction with output file."""
        audio_file, metadata_file = sample_audio_file
        output_file = tmp_path / "pitch_output.json"
        
        result = cli_runner.invoke(app, [
            "dataset", "extract-pitch",
            str(audio_file),
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Validate output format
        with open(output_file) as f:
            data = json.load(f)
        
        assert "f0_hz" in data
        assert "voicing_prob" in data
        assert "method" in data
        assert isinstance(data["f0_hz"], list)
    
    def test_extract_pitch_with_metadata(self, cli_runner, sample_audio_file):
        """Test pitch extraction with metadata file."""
        audio_file, metadata_file = sample_audio_file
        
        result = cli_runner.invoke(app, [
            "dataset", "extract-pitch",
            str(audio_file),
            "--metadata", str(metadata_file),
            "--quantize"
        ])
        
        assert result.exit_code == 0
        assert "quantized" in result.stdout
    
    def test_extract_pitch_with_tonic(self, cli_runner, sample_audio_file):
        """Test pitch extraction with specified tonic."""
        audio_file, metadata_file = sample_audio_file
        
        result = cli_runner.invoke(app, [
            "dataset", "extract-pitch",
            str(audio_file),
            "--tonic", "261.63",
            "--quantize"
        ])
        
        assert result.exit_code == 0
        assert "quantized" in result.stdout
    
    def test_extract_pitch_different_methods(self, cli_runner, sample_audio_file):
        """Test pitch extraction with different methods."""
        audio_file, metadata_file = sample_audio_file
        
        methods = ["auto", "praat", "librosa"]
        
        for method in methods:
            result = cli_runner.invoke(app, [
                "dataset", "extract-pitch",
                str(audio_file),
                "--method", method
            ])
            
            # Some methods might fail, but should handle gracefully
            assert result.exit_code in [0, 1]
    
    def test_extract_pitch_no_smoothing(self, cli_runner, sample_audio_file):
        """Test pitch extraction without smoothing."""
        audio_file, metadata_file = sample_audio_file
        
        result = cli_runner.invoke(app, [
            "dataset", "extract-pitch",
            str(audio_file),
            "--no-smooth"
        ])
        
        assert result.exit_code == 0
    
    def test_extract_pitch_quantize_without_tonic(self, cli_runner, sample_audio_file):
        """Test pitch extraction with quantization but no tonic."""
        audio_file, metadata_file = sample_audio_file
        
        result = cli_runner.invoke(app, [
            "dataset", "extract-pitch",
            str(audio_file),
            "--quantize"
        ])
        
        assert result.exit_code == 1
        assert "requires tonic frequency" in result.stdout
    
    def test_extract_pitch_nonexistent_file(self, cli_runner, tmp_path):
        """Test pitch extraction with nonexistent file."""
        nonexistent_file = tmp_path / "nonexistent.wav"
        
        result = cli_runner.invoke(app, [
            "dataset", "extract-pitch",
            str(nonexistent_file)
        ])
        
        assert result.exit_code == 1


class TestValidateDatasetCommand:
    """Test validate dataset command."""
    
    def test_validate_dataset_basic(self, cli_runner, sample_dataset):
        """Test basic dataset validation."""
        result = cli_runner.invoke(app, [
            "dataset", "validate",
            str(sample_dataset)
        ])
        
        assert result.exit_code == 0
        assert "Validation completed" in result.stdout
        assert "Raag Distribution" in result.stdout
    
    def test_validate_dataset_specific_split(self, cli_runner, sample_dataset):
        """Test dataset validation for specific split."""
        result = cli_runner.invoke(app, [
            "dataset", "validate",
            str(sample_dataset),
            "--split", "train"
        ])
        
        assert result.exit_code == 0
        assert "Validating train split" in result.stdout
    
    def test_validate_dataset_no_audio_check(self, cli_runner, sample_dataset):
        """Test dataset validation without audio integrity check."""
        result = cli_runner.invoke(app, [
            "dataset", "validate",
            str(sample_dataset),
            "--no-check-audio"
        ])
        
        assert result.exit_code == 0
    
    def test_validate_dataset_with_errors(self, cli_runner, tmp_path):
        """Test dataset validation with errors."""
        # Create dataset with missing metadata
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        
        # Create audio file without metadata
        sample_rate = 22050
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        audio_file = train_dir / "orphan.wav"
        sf.write(str(audio_file), audio_data, sample_rate)
        
        # Create metadata without audio
        metadata = {
            "recording_id": "orphan_metadata",
            "raag": "bihag",
            "tonic_hz": 261.63
        }
        
        metadata_file = train_dir / "orphan_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        result = cli_runner.invoke(app, [
            "dataset", "validate",
            str(tmp_path)
        ])
        
        assert result.exit_code == 1
        assert "Errors found" in result.stdout


class TestCLIErrorHandling:
    """Test CLI error handling and help system."""
    
    def test_main_help(self, cli_runner):
        """Test main help command."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "RaagHMM" in result.stdout
        assert "dataset" in result.stdout
    
    def test_dataset_help(self, cli_runner):
        """Test dataset subcommand help."""
        result = cli_runner.invoke(app, ["dataset", "--help"])
        
        assert result.exit_code == 0
        assert "Dataset preparation" in result.stdout
    
    def test_prepare_help(self, cli_runner):
        """Test prepare command help."""
        result = cli_runner.invoke(app, ["dataset", "prepare", "--help"])
        
        assert result.exit_code == 0
        assert "Prepare dataset" in result.stdout
    
    def test_extract_pitch_help(self, cli_runner):
        """Test extract-pitch command help."""
        result = cli_runner.invoke(app, ["dataset", "extract-pitch", "--help"])
        
        assert result.exit_code == 0
        assert "Extract pitch contour" in result.stdout
    
    def test_validate_help(self, cli_runner):
        """Test validate command help."""
        result = cli_runner.invoke(app, ["dataset", "validate", "--help"])
        
        assert result.exit_code == 0
        assert "Validate dataset" in result.stdout
    
    def test_verbose_logging(self, cli_runner, sample_audio_file):
        """Test verbose logging option."""
        audio_file, metadata_file = sample_audio_file
        
        result = cli_runner.invoke(app, [
            "--verbose",
            "dataset", "extract-pitch",
            str(audio_file)
        ])
        
        # Should not fail with verbose logging
        assert result.exit_code == 0
    
    def test_quiet_logging(self, cli_runner, sample_audio_file):
        """Test quiet logging option."""
        audio_file, metadata_file = sample_audio_file
        
        result = cli_runner.invoke(app, [
            "--quiet",
            "dataset", "extract-pitch",
            str(audio_file)
        ])
        
        # Should not fail with quiet logging
        assert result.exit_code == 0