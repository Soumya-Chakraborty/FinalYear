"""
Comprehensive error handling for CLI commands.

Defines custom exceptions and error handling utilities for better user experience.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
logger = logging.getLogger(__name__)


class RaagHMMCLIError(Exception):
    """Base exception for CLI-specific errors."""
    
    def __init__(self, message: str, exit_code: int = 1, suggestions: Optional[list] = None):
        self.message = message
        self.exit_code = exit_code
        self.suggestions = suggestions or []
        super().__init__(message)


class DatasetError(RaagHMMCLIError):
    """Dataset-related errors."""
    pass


class ModelError(RaagHMMCLIError):
    """Model-related errors."""
    pass


class AudioProcessingError(RaagHMMCLIError):
    """Audio processing errors."""
    pass


class ConfigurationError(RaagHMMCLIError):
    """Configuration errors."""
    pass


def format_error_message(error: Exception, operation: str, debug: bool = False) -> str:
    """Format error message with context and suggestions."""
    error_type = type(error).__name__
    
    # Create main error message
    message_parts = [
        f"[red]Error during {operation}:[/red]",
        f"[red]{error_type}: {error}[/red]"
    ]
    
    # Add suggestions if available
    if hasattr(error, 'suggestions') and error.suggestions:
        message_parts.append("")
        message_parts.append("[yellow]Suggestions:[/yellow]")
        for suggestion in error.suggestions:
            message_parts.append(f"  • {suggestion}")
    
    # Add debug information if requested
    if debug:
        message_parts.append("")
        message_parts.append("[dim]Debug information:[/dim]")
        message_parts.append(f"[dim]{traceback.format_exc()}[/dim]")
    
    return "\n".join(message_parts)


def handle_cli_error(error: Exception, operation: str, debug: bool = False) -> None:
    """Handle CLI errors with rich formatting and helpful messages."""
    
    # Determine if this is a known CLI error
    if isinstance(error, RaagHMMCLIError):
        exit_code = error.exit_code
    else:
        exit_code = 1
    
    # Format and display error
    error_message = format_error_message(error, operation, debug)
    console.print(error_message)
    
    # Add general help information
    console.print(f"\n[dim]For more help, run: raag-hmm {operation.split()[0]} --help[/dim]")
    
    # Log error for debugging
    logger.error(f"CLI error in {operation}: {error}", exc_info=debug)
    
    raise typer.Exit(exit_code)


def validate_file_exists(path: Path, file_type: str = "file") -> Path:
    """Validate that a file exists with helpful error messages."""
    if not path.exists():
        suggestions = []
        
        # Check if parent directory exists
        if not path.parent.exists():
            suggestions.append(f"Create the directory: mkdir -p {path.parent}")
        
        # Check for similar files
        if path.parent.exists():
            similar_files = []
            for file in path.parent.iterdir():
                if file.name.lower().startswith(path.stem.lower()[:3]):
                    similar_files.append(file.name)
            
            if similar_files:
                suggestions.append(f"Did you mean one of: {', '.join(similar_files[:3])}")
        
        raise DatasetError(
            f"{file_type.capitalize()} not found: {path}",
            suggestions=suggestions
        )
    
    return path


def validate_directory_exists(path: Path, dir_type: str = "directory") -> Path:
    """Validate that a directory exists with helpful error messages."""
    if not path.exists():
        suggestions = [
            f"Create the directory: mkdir -p {path}",
            f"Check the path spelling: {path}"
        ]
        
        # Check for similar directories
        if path.parent.exists():
            similar_dirs = []
            for item in path.parent.iterdir():
                if item.is_dir() and item.name.lower().startswith(path.name.lower()[:3]):
                    similar_dirs.append(item.name)
            
            if similar_dirs:
                suggestions.append(f"Did you mean one of: {', '.join(similar_dirs[:3])}")
        
        raise DatasetError(
            f"{dir_type.capitalize()} not found: {path}",
            suggestions=suggestions
        )
    
    if not path.is_dir():
        raise DatasetError(
            f"Path is not a directory: {path}",
            suggestions=[f"Remove the file and create directory: rm {path} && mkdir -p {path}"]
        )
    
    return path


def validate_audio_format(path: Path) -> Path:
    """Validate audio file format with helpful suggestions."""
    supported_formats = {'.wav', '.flac', '.mp3', '.m4a', '.aac'}
    
    if path.suffix.lower() not in supported_formats:
        suggestions = [
            f"Convert to supported format: ffmpeg -i {path} {path.with_suffix('.wav')}",
            f"Supported formats: {', '.join(supported_formats)}"
        ]
        
        raise AudioProcessingError(
            f"Unsupported audio format: {path.suffix}",
            suggestions=suggestions
        )
    
    return path


def validate_models_directory(path: Path) -> Path:
    """Validate models directory with specific checks for model files."""
    validate_directory_exists(path, "models directory")
    
    # Check for model files
    model_files = list(path.glob("*.pkl"))
    if not model_files:
        suggestions = [
            f"Train models first: raag-hmm train models <dataset> {path}",
            "Check if models are in a subdirectory",
            "Verify model file extensions (.pkl expected)"
        ]
        
        raise ModelError(
            f"No model files (*.pkl) found in: {path}",
            suggestions=suggestions
        )
    
    # Check for metadata files
    metadata_files = list(path.glob("*_meta.json"))
    if len(metadata_files) < len(model_files):
        console.print(f"[yellow]Warning: Some model metadata files missing in {path}[/yellow]")
    
    return path


def validate_dataset_structure(path: Path) -> Path:
    """Validate dataset directory structure with detailed feedback."""
    validate_directory_exists(path, "dataset directory")
    
    # Check for expected splits
    expected_splits = ['train', 'test', 'val']
    found_splits = []
    
    for split in expected_splits:
        split_dir = path / split
        if split_dir.exists() and split_dir.is_dir():
            found_splits.append(split)
    
    if not found_splits:
        suggestions = [
            f"Create dataset structure: mkdir -p {path}/train {path}/test",
            "Check if audio files are in the root directory instead of splits",
            "Verify dataset follows expected structure"
        ]
        
        raise DatasetError(
            f"No valid dataset splits found in: {path}",
            suggestions=suggestions
        )
    
    # Check for audio and metadata files in each split
    for split in found_splits:
        split_dir = path / split
        audio_files = []
        metadata_files = []
        
        for ext in ['.wav', '.flac', '.mp3']:
            audio_files.extend(split_dir.glob(f"*{ext}"))
        
        metadata_files = list(split_dir.glob("*.json"))
        
        if not audio_files:
            console.print(f"[yellow]Warning: No audio files found in {split} split[/yellow]")
        
        if not metadata_files:
            console.print(f"[yellow]Warning: No metadata files found in {split} split[/yellow]")
        
        # Check for orphaned files
        audio_stems = {f.stem for f in audio_files}
        metadata_stems = {f.stem for f in metadata_files}
        
        orphaned_audio = audio_stems - metadata_stems
        orphaned_metadata = metadata_stems - audio_stems
        
        if orphaned_audio:
            console.print(f"[yellow]Warning: Audio files without metadata in {split}: "
                         f"{list(orphaned_audio)[:3]}{'...' if len(orphaned_audio) > 3 else ''}[/yellow]")
        
        if orphaned_metadata:
            console.print(f"[yellow]Warning: Metadata files without audio in {split}: "
                         f"{list(orphaned_metadata)[:3]}{'...' if len(orphaned_metadata) > 3 else ''}[/yellow]")
    
    return path


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and return status."""
    requirements = {
        "python_version": {
            "required": "3.8+",
            "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "satisfied": sys.version_info >= (3, 8)
        }
    }
    
    # Check for required packages
    required_packages = [
        "numpy", "scipy", "librosa", "soundfile", 
        "praat-parselmouth", "scikit-learn", "joblib"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            requirements[package] = {"installed": True}
        except ImportError:
            requirements[package] = {
                "installed": False,
                "install_command": f"pip install {package}"
            }
    
    return requirements


def display_system_info() -> None:
    """Display system information and requirements status."""
    requirements = check_system_requirements()
    
    console.print(Panel.fit(
        "[bold]System Information[/bold]",
        border_style="blue"
    ))
    
    # Python version
    python_req = requirements["python_version"]
    status = "[green]✓[/green]" if python_req["satisfied"] else "[red]✗[/red]"
    console.print(f"Python: {status} {python_req['current']} (required: {python_req['required']})")
    
    # Package status
    console.print("\n[bold]Package Status:[/bold]")
    for package, info in requirements.items():
        if package == "python_version":
            continue
        
        if info["installed"]:
            console.print(f"  [green]✓[/green] {package}")
        else:
            console.print(f"  [red]✗[/red] {package} - Install with: {info['install_command']}")


def create_usage_examples() -> Dict[str, list]:
    """Create usage examples for different commands."""
    return {
        "dataset": [
            "# Prepare dataset with resampling",
            "raag-hmm dataset prepare /path/to/raw/dataset /path/to/processed/dataset",
            "",
            "# Extract pitch from single file",
            "raag-hmm dataset extract-pitch audio.wav --metadata metadata.json --quantize",
            "",
            "# Validate dataset structure",
            "raag-hmm dataset validate /path/to/dataset --check-audio"
        ],
        "train": [
            "# Train all raag models",
            "raag-hmm train models /path/to/dataset /path/to/models",
            "",
            "# Train with custom parameters",
            "raag-hmm train models dataset/ models/ --max-iter 300 --tolerance 0.05",
            "",
            "# Train single raag model",
            "raag-hmm train single bihag /path/to/dataset bihag_model.pkl"
        ],
        "predict": [
            "# Classify single audio file",
            "raag-hmm predict single audio.wav models/ --tonic 261.63",
            "",
            "# Batch classification",
            "raag-hmm predict batch audio_dir/ models/ results.json",
            "",
            "# Save detailed results",
            "raag-hmm predict single audio.wav models/ -m metadata.json -o results.json"
        ],
        "evaluate": [
            "# Evaluate on test set",
            "raag-hmm evaluate test dataset/ models/ results/",
            "",
            "# Compare multiple model sets",
            "raag-hmm evaluate compare dataset/ models_v1/ models_v2/ comparison.json",
            "",
            "# Custom evaluation parameters",
            "raag-hmm evaluate test dataset/ models/ results/ --top-k 1 --top-k 5 --format json"
        ]
    }


def display_usage_examples(command: Optional[str] = None) -> None:
    """Display usage examples for commands."""
    examples = create_usage_examples()
    
    if command and command in examples:
        console.print(Panel.fit(
            f"[bold]{command.title()} Command Examples[/bold]\n\n" + 
            "\n".join(examples[command]),
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold]RaagHMM Usage Examples[/bold]",
            border_style="green"
        ))
        
        for cmd, cmd_examples in examples.items():
            console.print(f"\n[bold cyan]{cmd.title()} Commands:[/bold cyan]")
            for example in cmd_examples[:3]:  # Show first 3 examples
                if example.startswith("#"):
                    console.print(f"[dim]{example}[/dim]")
                else:
                    console.print(f"  {example}")


def suggest_next_steps(current_command: str, success: bool = True) -> None:
    """Suggest logical next steps based on current command."""
    suggestions = {
        "dataset": {
            True: [
                "Train models: raag-hmm train models <dataset> <output_dir>",
                "Validate dataset: raag-hmm dataset validate <dataset>"
            ],
            False: [
                "Check dataset structure and file formats",
                "Verify audio files and metadata are properly paired"
            ]
        },
        "train": {
            True: [
                "Evaluate models: raag-hmm evaluate test <dataset> <models> <results>",
                "Make predictions: raag-hmm predict single <audio> <models>"
            ],
            False: [
                "Check dataset quality and size",
                "Try different hyperparameters (--max-iter, --tolerance)",
                "Validate training data: raag-hmm dataset validate <dataset>"
            ]
        },
        "predict": {
            True: [
                "Evaluate on test set: raag-hmm evaluate test <dataset> <models> <results>",
                "Process more files: raag-hmm predict batch <audio_dir> <models> <results>"
            ],
            False: [
                "Check model files exist and are valid",
                "Verify audio format and tonic frequency",
                "Try different audio preprocessing parameters"
            ]
        },
        "evaluate": {
            True: [
                "Analyze results in the output directory",
                "Compare with different model configurations",
                "Process new audio files: raag-hmm predict single <audio> <models>"
            ],
            False: [
                "Check test dataset quality",
                "Verify model compatibility",
                "Review training data and model parameters"
            ]
        }
    }
    
    if current_command in suggestions:
        status_suggestions = suggestions[current_command][success]
        status_text = "Next Steps" if success else "Troubleshooting"
        
        console.print(f"\n[bold yellow]{status_text}:[/bold yellow]")
        for suggestion in status_suggestions:
            console.print(f"  • {suggestion}")


# Exit codes for different error types
EXIT_CODES = {
    "success": 0,
    "general_error": 1,
    "invalid_usage": 2,
    "dataset_error": 10,
    "model_error": 11,
    "audio_error": 12,
    "config_error": 13,
    "system_error": 20
}