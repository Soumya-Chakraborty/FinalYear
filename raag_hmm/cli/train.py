"""
Training CLI commands.

Commands for training HMM models on raag datasets.
"""

import json
from pathlib import Path
from typing import Optional, List
import logging
import time

import typer
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ..train import RaagTrainer
from ..io import iter_dataset
from .utils import handle_error, validate_dataset_directory

console = Console()
logger = logging.getLogger(__name__)

# Create train subcommand group
train_app = typer.Typer(
    name="train",
    help="Model training commands"
)


@train_app.command("models")
def train_models(
    dataset_dir: Path = typer.Argument(
        ...,
        help="Dataset directory containing training data",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for trained models"
    ),
    max_iterations: int = typer.Option(
        200,
        "--max-iter",
        "-i",
        help="Maximum training iterations per model"
    ),
    tolerance: float = typer.Option(
        0.1,
        "--tolerance",
        "-t",
        help="Convergence tolerance for training"
    ),
    regularization: float = typer.Option(
        0.01,
        "--regularization",
        "-r",
        help="Dirichlet regularization parameter"
    ),
    n_states: int = typer.Option(
        36,
        "--states",
        "-s",
        help="Number of HMM hidden states"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing models"
    ),
    raag_filter: Optional[List[str]] = typer.Option(
        None,
        "--raag",
        help="Train only specific raag classes (can be used multiple times)"
    )
):
    """
    Train HMM models for raag classification.
    
    This command trains separate Hidden Markov Models for each raag class
    found in the training dataset. Models are trained using the Baum-Welch
    algorithm with the specified hyperparameters.
    
    Examples:
    ```
    # Train all raag models
    raag-hmm train models /path/to/dataset /path/to/models
    
    # Train with custom parameters
    raag-hmm train models dataset/ models/ --max-iter 300 --tolerance 0.05
    
    # Train only specific raags
    raag-hmm train models dataset/ models/ --raag bihag --raag darbari
    ```
    """
    try:
        dataset_dir = validate_dataset_directory(dataset_dir)
        
        # Check output directory
        if output_dir.exists() and not force:
            existing_models = list(output_dir.glob("*.pkl"))
            if existing_models:
                console.print(f"[red]Models already exist in: {output_dir}[/red]")
                console.print("[yellow]Use --force to overwrite[/yellow]")
                raise typer.Exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(Panel.fit(
            f"[bold]Model Training[/bold]\n"
            f"Dataset: {dataset_dir}\n"
            f"Output: {output_dir}\n"
            f"Max Iterations: {max_iterations}\n"
            f"Tolerance: {tolerance}\n"
            f"Regularization: {regularization}\n"
            f"States: {n_states}\n"
            f"Raag Filter: {raag_filter or 'All'}",
            border_style="blue"
        ))
        
        # Initialize trainer
        trainer = RaagTrainer(
            n_states=n_states,
            n_observations=36,  # Fixed for chromatic quantization
            max_iterations=max_iterations,
            convergence_tolerance=tolerance,
            regularization_alpha=regularization
        )
        
        # Load training data
        console.print("[bold]Loading training data...[/bold]")
        
        training_sequences = {}
        total_sequences = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            load_task = progress.add_task("Loading sequences...", total=None)
            
            for audio_path, metadata in iter_dataset(str(dataset_dir), 'train'):
                raag = metadata['raag']
                
                # Apply raag filter if specified
                if raag_filter and raag not in raag_filter:
                    continue
                
                if raag not in training_sequences:
                    training_sequences[raag] = []
                
                # Load and process audio to get quantized sequence
                try:
                    from ..io import load_audio
                    from ..pitch import extract_pitch_with_fallback, smooth_pitch
                    from ..quantize import quantize_sequence
                    
                    # Load audio
                    audio_data = load_audio(audio_path)
                    
                    # Extract pitch
                    f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)
                    
                    # Smooth pitch
                    f0_hz = smooth_pitch(f0_hz, voicing_prob)
                    
                    # Quantize
                    tonic_hz = metadata['tonic_hz']
                    quantized = quantize_sequence(f0_hz, tonic_hz)
                    
                    # Remove NaN values
                    valid_indices = ~np.isnan(quantized)
                    if np.sum(valid_indices) > 0:
                        clean_sequence = quantized[valid_indices].astype(int)
                        training_sequences[raag].append(clean_sequence)
                        total_sequences += 1
                    
                    progress.update(load_task, description=f"Loaded {total_sequences} sequences...")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to process {audio_path}: {e}[/yellow]")
        
        if not training_sequences:
            console.print("[red]No training sequences found![/red]")
            raise typer.Exit(1)
        
        # Display training data summary
        console.print(f"\n[bold]Training Data Summary:[/bold]")
        summary_table = Table()
        summary_table.add_column("Raag", style="cyan")
        summary_table.add_column("Sequences", style="magenta")
        summary_table.add_column("Total Frames", style="green")
        
        for raag, sequences in training_sequences.items():
            total_frames = sum(len(seq) for seq in sequences)
            summary_table.add_row(raag, str(len(sequences)), str(total_frames))
        
        console.print(summary_table)
        
        # Train models
        console.print(f"\n[bold]Training {len(training_sequences)} models...[/bold]")
        
        training_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            overall_task = progress.add_task("Training models", total=len(training_sequences))
            
            for raag, sequences in training_sequences.items():
                start_time = time.time()
                
                progress.update(overall_task, description=f"Training {raag}...")
                
                try:
                    # Train model
                    model, training_info = trainer.train_raag_model(raag, sequences)
                    
                    # Save model
                    model_path = output_dir / f"{raag}.pkl"
                    trainer.save_model(model, str(model_path))
                    
                    # Save metadata
                    training_time = time.time() - start_time
                    metadata = {
                        "raag_name": raag,
                        "n_sequences": len(sequences),
                        "total_frames": sum(len(seq) for seq in sequences),
                        "n_states": n_states,
                        "n_observations": 36,
                        "max_iterations": max_iterations,
                        "tolerance": tolerance,
                        "regularization_alpha": regularization,
                        "convergence_iterations": training_info.get('iterations', 0),
                        "final_log_likelihood": training_info.get('log_likelihood', 0.0),
                        "converged": training_info.get('converged', False),
                        "training_time": training_time
                    }
                    
                    metadata_path = output_dir / f"{raag}_meta.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    training_results[raag] = metadata
                    
                    console.print(f"[green]✓ {raag}: {training_info.get('iterations', 0)} iterations, "
                                f"LL={training_info.get('log_likelihood', 0.0):.2f}[/green]")
                    
                except Exception as e:
                    console.print(f"[red]✗ {raag}: Training failed - {e}[/red]")
                    training_results[raag] = {"error": str(e)}
                
                progress.update(overall_task, advance=1)
        
        # Display final results
        console.print(f"\n[bold green]Training completed![/bold green]")
        
        results_table = Table(title="Training Results")
        results_table.add_column("Raag", style="cyan")
        results_table.add_column("Status", style="magenta")
        results_table.add_column("Iterations", style="green")
        results_table.add_column("Log-Likelihood", style="yellow")
        results_table.add_column("Time (s)", style="blue")
        
        successful_models = 0
        for raag, result in training_results.items():
            if "error" in result:
                results_table.add_row(raag, "[red]Failed[/red]", "-", "-", "-")
            else:
                results_table.add_row(
                    raag,
                    "[green]Success[/green]",
                    str(result["convergence_iterations"]),
                    f"{result['final_log_likelihood']:.2f}",
                    f"{result['training_time']:.1f}"
                )
                successful_models += 1
        
        console.print(results_table)
        console.print(f"Successfully trained {successful_models}/{len(training_sequences)} models")
        
        if successful_models == 0:
            raise typer.Exit(1)
        
    except Exception as e:
        handle_error(e, "model training")


@train_app.command("single")
def train_single_model(
    raag_name: str = typer.Argument(..., help="Name of the raag to train"),
    dataset_dir: Path = typer.Argument(
        ...,
        help="Dataset directory containing training data",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    output_file: Path = typer.Argument(..., help="Output file for the trained model"),
    max_iterations: int = typer.Option(
        200,
        "--max-iter",
        "-i",
        help="Maximum training iterations"
    ),
    tolerance: float = typer.Option(
        0.1,
        "--tolerance",
        "-t",
        help="Convergence tolerance"
    ),
    n_states: int = typer.Option(
        36,
        "--states",
        "-s",
        help="Number of HMM hidden states"
    )
):
    """
    Train a single HMM model for a specific raag.
    
    This command trains a single Hidden Markov Model for the specified raag
    class using all available training sequences for that raag.
    
    Examples:
    ```
    # Train a single raag model
    raag-hmm train single bihag /path/to/dataset bihag_model.pkl
    
    # Train with custom parameters
    raag-hmm train single darbari dataset/ darbari.pkl --max-iter 300 --states 48
    ```
    """
    try:
        dataset_dir = validate_dataset_directory(dataset_dir)
        
        console.print(Panel.fit(
            f"[bold]Single Model Training[/bold]\n"
            f"Raag: {raag_name}\n"
            f"Dataset: {dataset_dir}\n"
            f"Output: {output_file}\n"
            f"Max Iterations: {max_iterations}\n"
            f"States: {n_states}",
            border_style="blue"
        ))
        
        # Initialize trainer
        trainer = RaagTrainer(
            n_states=n_states,
            n_observations=36,
            max_iterations=max_iterations,
            convergence_tolerance=tolerance
        )
        
        # Load sequences for the specific raag
        console.print(f"[bold]Loading {raag_name} sequences...[/bold]")
        
        sequences = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            load_task = progress.add_task("Loading sequences...", total=None)
            
            for audio_path, metadata in iter_dataset(str(dataset_dir), 'train'):
                if metadata['raag'] != raag_name:
                    continue
                
                try:
                    from ..io import load_audio
                    from ..pitch import extract_pitch_with_fallback, smooth_pitch
                    from ..quantize import quantize_sequence
                    
                    # Process audio
                    audio_data = load_audio(audio_path)
                    f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)
                    f0_hz = smooth_pitch(f0_hz, voicing_prob)
                    
                    tonic_hz = metadata['tonic_hz']
                    quantized = quantize_sequence(f0_hz, tonic_hz)
                    
                    # Clean sequence
                    valid_indices = ~np.isnan(quantized)
                    if np.sum(valid_indices) > 0:
                        clean_sequence = quantized[valid_indices].astype(int)
                        sequences.append(clean_sequence)
                    
                    progress.update(load_task, description=f"Loaded {len(sequences)} sequences...")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to process {audio_path}: {e}[/yellow]")
        
        if not sequences:
            console.print(f"[red]No training sequences found for raag: {raag_name}[/red]")
            raise typer.Exit(1)
        
        console.print(f"Found {len(sequences)} sequences for {raag_name}")
        
        # Train model
        console.print(f"[bold]Training {raag_name} model...[/bold]")
        
        start_time = time.time()
        model, training_info = trainer.train_raag_model(raag_name, sequences)
        training_time = time.time() - start_time
        
        # Save model
        trainer.save_model(model, str(output_file))
        
        # Save metadata
        metadata_file = output_file.with_suffix('.json')
        metadata = {
            "raag_name": raag_name,
            "n_sequences": len(sequences),
            "total_frames": sum(len(seq) for seq in sequences),
            "n_states": n_states,
            "n_observations": 36,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "convergence_iterations": training_info.get('iterations', 0),
            "final_log_likelihood": training_info.get('log_likelihood', 0.0),
            "converged": training_info.get('converged', False),
            "training_time": training_time
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Display results
        console.print(f"\n[bold green]Training completed![/bold green]")
        console.print(f"Model saved to: {output_file}")
        console.print(f"Metadata saved to: {metadata_file}")
        console.print(f"Iterations: {training_info.get('iterations', 0)}")
        console.print(f"Final log-likelihood: {training_info.get('log_likelihood', 0.0):.2f}")
        console.print(f"Converged: {training_info.get('converged', False)}")
        console.print(f"Training time: {training_time:.1f} seconds")
        
    except Exception as e:
        handle_error(e, "single model training")