"""
Main CLI application for RaagHMM system.

Provides command-line interface for dataset preparation, training, inference, and evaluation.
"""

import sys
from pathlib import Path
from typing import Optional, List
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ..config import get_config
from ..logger import get_logger, set_log_level
from .errors import (
    handle_cli_error, 
    display_system_info, 
    display_usage_examples,
    suggest_next_steps,
    EXIT_CODES
)

# Initialize Rich console for pretty output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="raag-hmm",
    help="Hidden Markov Model-based Raag Detection System for Indian Classical Music",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Import and add subcommands
from .dataset import dataset_app
from .train import train_app
from .predict import predict_app
from .evaluate import evaluate_app

app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")
app.add_typer(evaluate_app, name="evaluate")


@app.command("info")
def system_info():
    """Display system information and requirements status."""
    display_system_info()


@app.command("examples")
def show_examples(
    command: Optional[str] = typer.Argument(
        None,
        help="Show examples for specific command (dataset/train/predict/evaluate)"
    )
):
    """Show usage examples for RaagHMM commands."""
    display_usage_examples(command)


@app.command("version")
def show_version():
    """Show RaagHMM version information."""
    try:
        from .. import __version__
        version = __version__
    except ImportError:
        version = "0.1.0"  # Fallback version
    
    console.print(Panel.fit(
        f"[bold]RaagHMM Version {version}[/bold]\n"
        f"Hidden Markov Model-based Raag Detection System\n"
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        border_style="blue"
    ))


# Global options
@app.callback()
def main(
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        "-v", 
        help="Enable verbose logging and debug information"
    ),
    quiet: bool = typer.Option(
        False, 
        "--quiet", 
        "-q", 
        help="Suppress all output except errors"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with detailed error traces"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )
):
    """
    RaagHMM: Hidden Markov Model-based Raag Detection System
    
    A comprehensive system for automatic raag (raga) detection in Indian classical music
    using Hidden Markov Models with discrete emissions.
    
    \b
    Quick Start:
    1. Prepare dataset:    raag-hmm dataset prepare <input> <output>
    2. Train models:       raag-hmm train models <dataset> <models>
    3. Make predictions:   raag-hmm predict single <audio> <models>
    4. Evaluate results:   raag-hmm evaluate test <dataset> <models> <results>
    
    \b
    For detailed examples: raag-hmm examples
    For system info:       raag-hmm info
    """
    # Store global options in context for error handling
    try:
        import click
        ctx = click.get_current_context()
        if ctx:
            ctx.meta["verbose"] = verbose
            ctx.meta["quiet"] = quiet
            ctx.meta["debug"] = debug
    except:
        # Context not available, continue without storing options
        pass
    
    # Set up logging based on verbosity
    if quiet:
        log_level = logging.ERROR
    elif verbose or debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    if log_level == logging.ERROR:
        set_log_level('ERROR')
    elif log_level == logging.DEBUG:
        set_log_level('DEBUG')
    else:
        set_log_level('INFO')
    
    # Load configuration if provided
    if config_file:
        # TODO: Implement config file loading
        if not quiet:
            console.print(f"[yellow]Config file loading not yet implemented: {config_file}[/yellow]")


def cli_main():
    """Main entry point for CLI with error handling."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(EXIT_CODES["general_error"])
    except Exception as e:
        # Get debug flag from context if available
        debug = False
        try:
            import click
            ctx = click.get_current_context()
            debug = ctx.meta.get("debug", False) if ctx else False
        except:
            pass
        
        handle_cli_error(e, "CLI operation", debug)


if __name__ == "__main__":
    cli_main()