"""
CLI utility functions.

Common utilities for CLI commands including error handling and validation.
"""

from pathlib import Path
import typer
from rich.console import Console

from .errors import (
    handle_cli_error, 
    validate_file_exists, 
    validate_dataset_structure,
    validate_models_directory,
    validate_audio_format
)

console = Console()


def handle_error(error: Exception, operation: str, debug: bool = False) -> None:
    """Handle and display errors consistently."""
    handle_cli_error(error, operation, debug)


def validate_audio_file(path: Path) -> Path:
    """Validate that the provided path is a supported audio file."""
    validate_file_exists(path, "audio file")
    validate_audio_format(path)
    return path


def validate_dataset_directory(path: Path) -> Path:
    """Validate that the provided path is a valid dataset directory."""
    return validate_dataset_structure(path)


def validate_model_directory(path: Path) -> Path:
    """Validate that the provided path contains trained models."""
    return validate_models_directory(path)