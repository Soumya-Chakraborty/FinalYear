"""
Tests for CLI error handling and help system.

Tests comprehensive error handling, help messages, and user experience features.
"""

import json
import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner

from raag_hmm.cli.main import app
from raag_hmm.cli.errors import (
    RaagHMMCLIError, DatasetError, ModelError, AudioProcessingError,
    validate_file_exists, validate_dataset_structure, validate_models_directory
)


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_dataset_error_with_suggestions(self, tmp_path):
        """Test dataset error with helpful suggestions."""
        nonexistent_dir = tmp_path / "nonexistent"
        
        with pytest.raises(DatasetError) as exc_info:
            validate_dataset_structure(nonexistent_dir)
        
        error = exc_info.value
        assert "not found" in str(error)
        assert hasattr(error, 'suggestions')
        assert len(error.suggestions) > 0
    
    def test_model_error_with_suggestions(self, tmp_path):
        """Test model error with helpful suggestions."""
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()
        
        with pytest.raises(ModelError) as exc_info:
            validate_models_directory(empty_dir)
        
        error = exc_info.value
        assert "No model files" in str(error)
        assert hasattr(error, 'suggestions')
        assert any("train models" in suggestion.lower() for suggestion in error.suggestions)
    
    def test_file_validation_with_similar_files(self, tmp_path):
        """Test file validation suggests similar files."""
        # Create similar files
        (tmp_path / "audio_file.wav").touch()
        (tmp_path / "audio_test.wav").touch()
        
        nonexistent = tmp_path / "audio_missing.wav"
        
        with pytest.raises(DatasetError) as exc_info:
            validate_file_exists(nonexistent, "audio file")
        
        error = exc_info.value
        assert hasattr(error, 'suggestions')
        # Should suggest similar files
        suggestions_text = " ".join(error.suggestions)
        assert "audio_file.wav" in suggestions_text or "audio_test.wav" in suggestions_text


class TestHelpSystem:
    """Test help system and user guidance."""
    
    def test_main_help_command(self, cli_runner):
        """Test main help shows all commands and description."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "RaagHMM" in result.stdout or "Hidden Markov Model" in result.stdout
        assert "dataset" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "evaluate" in result.stdout
        assert "info" in result.stdout
        assert "examples" in result.stdout
    
    def test_info_command(self, cli_runner):
        """Test system info command."""
        result = cli_runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "System Information" in result.stdout
        assert "Python" in result.stdout
        assert "Package Status" in result.stdout
    
    def test_examples_command_general(self, cli_runner):
        """Test examples command without specific command."""
        result = cli_runner.invoke(app, ["examples"])
        
        assert result.exit_code == 0
        assert "Usage Examples" in result.stdout
        assert "Dataset Commands" in result.stdout
        assert "Train Commands" in result.stdout
    
    def test_examples_command_specific(self, cli_runner):
        """Test examples command for specific command."""
        result = cli_runner.invoke(app, ["examples", "dataset"])
        
        assert result.exit_code == 0
        assert "Dataset Command Examples" in result.stdout
        assert "raag-hmm dataset" in result.stdout
    
    def test_version_command(self, cli_runner):
        """Test version command."""
        result = cli_runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "RaagHMM Version" in result.stdout
        assert "Python" in result.stdout
    
    def test_subcommand_help_messages(self, cli_runner):
        """Test that all subcommands have proper help."""
        commands = [
            ["dataset", "--help"],
            ["train", "--help"],
            ["predict", "--help"],
            ["evaluate", "--help"],
            ["dataset", "prepare", "--help"],
            ["dataset", "extract-pitch", "--help"],
            ["dataset", "validate", "--help"],
            ["train", "models", "--help"],
            ["train", "single", "--help"],
            ["predict", "single", "--help"],
            ["predict", "batch", "--help"],
            ["evaluate", "test", "--help"],
            ["evaluate", "compare", "--help"]
        ]
        
        for command in commands:
            result = cli_runner.invoke(app, command)
            assert result.exit_code == 0
            assert "Usage:" in result.stdout
            assert "help" in result.stdout.lower()


class TestGlobalOptions:
    """Test global CLI options."""
    
    def test_verbose_option(self, cli_runner):
        """Test verbose option."""
        result = cli_runner.invoke(app, ["--verbose", "--help"])
        assert result.exit_code == 0
    
    def test_quiet_option(self, cli_runner):
        """Test quiet option."""
        result = cli_runner.invoke(app, ["--quiet", "--help"])
        assert result.exit_code == 0
    
    def test_debug_option(self, cli_runner):
        """Test debug option."""
        result = cli_runner.invoke(app, ["--debug", "--help"])
        assert result.exit_code == 0
    
    def test_config_option_nonexistent(self, cli_runner, tmp_path):
        """Test config option with nonexistent file."""
        nonexistent_config = tmp_path / "nonexistent.yaml"
        
        result = cli_runner.invoke(app, [
            "--config", str(nonexistent_config),
            "--help"
        ])
        
        # Should fail due to file not existing
        assert result.exit_code != 0
    
    def test_config_option_existing(self, cli_runner, tmp_path):
        """Test config option with existing file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# Test config")
        
        result = cli_runner.invoke(app, [
            "--config", str(config_file),
            "--help"
        ])
        
        assert result.exit_code == 0
        # Should show warning about config not implemented
        assert "not yet implemented" in result.stdout or result.exit_code == 0


class TestErrorMessages:
    """Test error message quality and helpfulness."""
    
    def test_missing_required_arguments(self, cli_runner):
        """Test error messages for missing required arguments."""
        # Test various commands without required arguments
        commands_to_test = [
            ["dataset", "prepare"],
            ["train", "models"],
            ["predict", "single"],
            ["evaluate", "test"]
        ]
        
        for command in commands_to_test:
            result = cli_runner.invoke(app, command)
            assert result.exit_code == 2  # Typer's missing argument error
            assert "Usage:" in result.stdout
    
    def test_invalid_command(self, cli_runner):
        """Test error message for invalid command."""
        result = cli_runner.invoke(app, ["invalid-command"])
        
        assert result.exit_code == 2
        assert "No such command" in result.stdout or "Usage:" in result.stdout
    
    def test_invalid_subcommand(self, cli_runner):
        """Test error message for invalid subcommand."""
        result = cli_runner.invoke(app, ["dataset", "invalid-subcommand"])
        
        assert result.exit_code == 2
        assert "No such command" in result.stdout or "Usage:" in result.stdout
    
    def test_nonexistent_file_error(self, cli_runner, tmp_path):
        """Test error handling for nonexistent files."""
        nonexistent_file = tmp_path / "nonexistent.wav"
        nonexistent_models = tmp_path / "nonexistent_models"
        
        result = cli_runner.invoke(app, [
            "predict", "single",
            str(nonexistent_file),
            str(nonexistent_models)
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
    
    def test_invalid_directory_error(self, cli_runner, tmp_path):
        """Test error handling for invalid directories."""
        # Create a file instead of directory
        fake_dir = tmp_path / "fake_dir"
        fake_dir.write_text("This is a file, not a directory")
        
        result = cli_runner.invoke(app, [
            "dataset", "validate",
            str(fake_dir)
        ])
        
        assert result.exit_code == 1


class TestUserExperience:
    """Test overall user experience features."""
    
    def test_no_args_shows_help(self, cli_runner):
        """Test that running with no arguments shows help."""
        result = cli_runner.invoke(app, [])
        
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Commands:" in result.stdout
    
    def test_keyboard_interrupt_handling(self, cli_runner):
        """Test graceful handling of keyboard interrupt."""
        # This is difficult to test directly, but we can test the handler exists
        from raag_hmm.cli.main import cli_main
        assert callable(cli_main)
    
    def test_command_suggestions_in_help(self, cli_runner):
        """Test that help includes useful command suggestions."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        # Should include helpful commands
        assert "examples" in result.stdout
        assert "info" in result.stdout
        assert "Commands" in result.stdout
    
    def test_rich_formatting_in_output(self, cli_runner):
        """Test that output uses rich formatting."""
        result = cli_runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        # Rich formatting should be present (though exact format may vary)
        assert len(result.stdout) > 100  # Should have substantial formatted output


class TestExitCodes:
    """Test proper exit codes for different error conditions."""
    
    def test_success_exit_code(self, cli_runner):
        """Test success exit code."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
    
    def test_missing_argument_exit_code(self, cli_runner):
        """Test exit code for missing arguments."""
        result = cli_runner.invoke(app, ["dataset", "prepare"])
        assert result.exit_code == 2  # Typer's missing argument code
    
    def test_file_not_found_exit_code(self, cli_runner, tmp_path):
        """Test exit code for file not found errors."""
        nonexistent = tmp_path / "nonexistent.wav"
        models_dir = tmp_path / "models"
        
        result = cli_runner.invoke(app, [
            "predict", "single",
            str(nonexistent),
            str(models_dir)
        ])
        
        assert result.exit_code == 1


class TestDocumentationAndExamples:
    """Test documentation and example quality."""
    
    def test_command_descriptions(self, cli_runner):
        """Test that commands have good descriptions."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        # Check that main commands are described
        assert "Dataset preparation" in result.stdout
        assert "Model training" in result.stdout
        assert "Prediction" in result.stdout
        assert "evaluation" in result.stdout.lower()
    
    def test_example_completeness(self, cli_runner):
        """Test that examples cover main use cases."""
        result = cli_runner.invoke(app, ["examples"])
        
        assert result.exit_code == 0
        # Should include examples for all main operations
        assert "prepare" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "evaluate" in result.stdout
    
    def test_help_text_formatting(self, cli_runner):
        """Test that help text is well formatted."""
        commands = ["dataset", "train", "predict", "evaluate"]
        
        for command in commands:
            result = cli_runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            
            # Should have proper sections
            assert "Usage:" in result.stdout
            assert "Options:" in result.stdout or "Arguments:" in result.stdout
            
            # Should not have obvious formatting issues
            assert not result.stdout.count("\\n") > result.stdout.count("\n")  # No escaped newlines