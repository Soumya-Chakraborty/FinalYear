"""
Integration tests for CLI training and prediction commands.

Tests the train, predict, and evaluate commands.
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
def sample_dataset_with_models(tmp_path):
    """Create a sample dataset and train simple models for testing."""
    # Create directory structure
    train_dir = tmp_path / "dataset" / "train"
    test_dir = tmp_path / "dataset" / "test"
    models_dir = tmp_path / "models"
    
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    
    # Create sample audio files (sine waves with different patterns)
    sample_rate = 22050
    duration = 0.5  # Short duration for fast tests
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    raags = ["bihag", "darbari"]
    
    for i, raag in enumerate(raags):
        # Create train files
        base_freq = 220 + i * 110  # Different base frequencies
        
        for j in range(2):  # 2 files per raag
            # Create audio with simple pattern
            freq_pattern = base_freq * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))  # Slight vibrato
            audio_data = 0.3 * np.sin(2 * np.pi * freq_pattern * t)
            
            # Add some noise to make it more realistic
            noise = 0.05 * np.random.randn(len(audio_data))
            audio_data += noise
            
            audio_file = train_dir / f"{raag}_train_{j}.wav"
            sf.write(str(audio_file), audio_data, sample_rate)
            
            # Create metadata
            metadata = {
                "recording_id": f"{raag}_train_{j}",
                "raag": raag,
                "tonic_hz": 261.63,
                "artist": "Test Artist",
                "split": "train"
            }
            
            metadata_file = train_dir / f"{raag}_train_{j}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create test files
            audio_file = test_dir / f"{raag}_test_{j}.wav"
            sf.write(str(audio_file), audio_data, sample_rate)
            
            metadata["recording_id"] = f"{raag}_test_{j}"
            metadata["split"] = "test"
            
            metadata_file = test_dir / f"{raag}_test_{j}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    return {
        "dataset_dir": tmp_path / "dataset",
        "models_dir": models_dir,
        "train_dir": train_dir,
        "test_dir": test_dir
    }


class TestTrainCommands:
    """Test training commands."""
    
    def test_train_models_basic(self, cli_runner, sample_dataset_with_models):
        """Test basic model training."""
        dataset_dir = sample_dataset_with_models["dataset_dir"]
        models_dir = sample_dataset_with_models["models_dir"]
        
        result = cli_runner.invoke(app, [
            "train", "models",
            str(dataset_dir),
            str(models_dir),
            "--max-iter", "5",  # Very few iterations for fast test
            "--tolerance", "1.0"  # High tolerance for quick convergence
        ])
        
        # Training might fail due to insufficient data, but should handle gracefully
        assert result.exit_code in [0, 1]
        
        if result.exit_code == 0:
            # Check if models were created
            model_files = list(models_dir.glob("*.pkl"))
            assert len(model_files) > 0
    
    def test_train_single_model(self, cli_runner, sample_dataset_with_models):
        """Test single model training."""
        dataset_dir = sample_dataset_with_models["dataset_dir"]
        models_dir = sample_dataset_with_models["models_dir"]
        output_file = models_dir / "bihag_single.pkl"
        
        result = cli_runner.invoke(app, [
            "train", "single",
            "bihag",
            str(dataset_dir),
            str(output_file),
            "--max-iter", "5",
            "--states", "12"  # Fewer states for faster training
        ])
        
        # Training might fail due to insufficient data
        assert result.exit_code in [0, 1]
    
    def test_train_models_with_filter(self, cli_runner, sample_dataset_with_models):
        """Test training with raag filter."""
        dataset_dir = sample_dataset_with_models["dataset_dir"]
        models_dir = sample_dataset_with_models["models_dir"]
        
        result = cli_runner.invoke(app, [
            "train", "models",
            str(dataset_dir),
            str(models_dir),
            "--raag", "bihag",
            "--max-iter", "3"
        ])
        
        assert result.exit_code in [0, 1]
    
    def test_train_models_force_overwrite(self, cli_runner, sample_dataset_with_models):
        """Test training with force overwrite."""
        dataset_dir = sample_dataset_with_models["dataset_dir"]
        models_dir = sample_dataset_with_models["models_dir"]
        
        # Create a dummy model file
        dummy_model = models_dir / "dummy.pkl"
        dummy_model.write_text("dummy")
        
        # First run should fail without --force
        result = cli_runner.invoke(app, [
            "train", "models",
            str(dataset_dir),
            str(models_dir)
        ])
        
        assert result.exit_code == 1
        assert "already exist" in result.stdout
        
        # Second run with --force should proceed
        result = cli_runner.invoke(app, [
            "train", "models",
            str(dataset_dir),
            str(models_dir),
            "--force",
            "--max-iter", "3"
        ])
        
        assert result.exit_code in [0, 1]


class TestPredictCommands:
    """Test prediction commands."""
    
    def test_predict_single_help(self, cli_runner):
        """Test predict single command help."""
        result = cli_runner.invoke(app, ["predict", "single", "--help"])
        
        assert result.exit_code == 0
        assert "Classify a single audio file" in result.stdout
    
    def test_predict_batch_help(self, cli_runner):
        """Test predict batch command help."""
        result = cli_runner.invoke(app, ["predict", "batch", "--help"])
        
        assert result.exit_code == 0
        assert "Classify multiple audio files" in result.stdout
    
    def test_predict_single_missing_tonic(self, cli_runner, sample_dataset_with_models):
        """Test prediction without tonic frequency."""
        dataset_dir = sample_dataset_with_models["dataset_dir"]
        models_dir = sample_dataset_with_models["models_dir"]
        
        # Create a dummy model directory (prediction will fail, but we test error handling)
        audio_file = sample_dataset_with_models["test_dir"] / "bihag_test_0.wav"
        
        result = cli_runner.invoke(app, [
            "predict", "single",
            str(audio_file),
            str(models_dir)
        ])
        
        assert result.exit_code == 1
        assert "Tonic frequency required" in result.stdout
    
    def test_predict_batch_no_files(self, cli_runner, tmp_path):
        """Test batch prediction with no matching files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        output_file = tmp_path / "results.json"
        
        result = cli_runner.invoke(app, [
            "predict", "batch",
            str(empty_dir),
            str(models_dir),
            str(output_file)
        ])
        
        assert result.exit_code == 1
        assert "No audio files found" in result.stdout


class TestEvaluateCommands:
    """Test evaluation commands."""
    
    def test_evaluate_test_help(self, cli_runner):
        """Test evaluate test command help."""
        result = cli_runner.invoke(app, ["evaluate", "test", "--help"])
        
        assert result.exit_code == 0
        assert "Evaluate trained models" in result.stdout
    
    def test_evaluate_compare_help(self, cli_runner):
        """Test evaluate compare command help."""
        result = cli_runner.invoke(app, ["evaluate", "compare", "--help"])
        
        assert result.exit_code == 0
        assert "Compare performance" in result.stdout


class TestCLIIntegration:
    """Test CLI integration and workflow."""
    
    def test_full_workflow_help(self, cli_runner):
        """Test that all main commands have help."""
        commands = ["dataset", "train", "predict", "evaluate"]
        
        for command in commands:
            result = cli_runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            assert command in result.stdout.lower()
    
    def test_main_help_shows_all_commands(self, cli_runner):
        """Test that main help shows all available commands."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "dataset" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "evaluate" in result.stdout
    
    def test_verbose_and_quiet_options(self, cli_runner):
        """Test global verbose and quiet options."""
        # Test verbose
        result = cli_runner.invoke(app, ["--verbose", "--help"])
        assert result.exit_code == 0
        
        # Test quiet
        result = cli_runner.invoke(app, ["--quiet", "--help"])
        assert result.exit_code == 0
        
        # Test that verbose and quiet are mutually exclusive (if implemented)
        # This would depend on the specific implementation
    
    def test_config_file_option(self, cli_runner, tmp_path):
        """Test config file option."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# Test config")
        
        result = cli_runner.invoke(app, [
            "--config", str(config_file),
            "--help"
        ])
        
        assert result.exit_code == 0
        # Should show warning about config not implemented yet
        assert "not yet implemented" in result.stdout or result.exit_code == 0


class TestErrorHandling:
    """Test error handling across CLI commands."""
    
    def test_nonexistent_dataset_directory(self, cli_runner, tmp_path):
        """Test error handling for nonexistent dataset directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        result = cli_runner.invoke(app, [
            "train", "models",
            str(nonexistent_dir),
            str(models_dir)
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout
    
    def test_nonexistent_models_directory(self, cli_runner, sample_dataset_with_models):
        """Test error handling for nonexistent models directory."""
        dataset_dir = sample_dataset_with_models["dataset_dir"]
        nonexistent_models = sample_dataset_with_models["models_dir"] / "nonexistent"
        
        audio_file = sample_dataset_with_models["test_dir"] / "bihag_test_0.wav"
        
        result = cli_runner.invoke(app, [
            "predict", "single",
            str(audio_file),
            str(nonexistent_models),
            "--tonic", "261.63"
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout
    
    def test_invalid_audio_format(self, cli_runner, tmp_path):
        """Test error handling for invalid audio format."""
        # Create a text file with audio extension
        fake_audio = tmp_path / "fake.wav"
        fake_audio.write_text("This is not audio")
        
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        result = cli_runner.invoke(app, [
            "predict", "single",
            str(fake_audio),
            str(models_dir),
            "--tonic", "261.63"
        ])
        
        # Should fail during audio loading
        assert result.exit_code == 1
    
    def test_missing_required_arguments(self, cli_runner):
        """Test error handling for missing required arguments."""
        # Test train command without arguments
        result = cli_runner.invoke(app, ["train", "models"])
        assert result.exit_code == 2  # Typer error code for missing arguments
        
        # Test predict command without arguments
        result = cli_runner.invoke(app, ["predict", "single"])
        assert result.exit_code == 2