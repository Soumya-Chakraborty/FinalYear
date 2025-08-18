"""
Prediction and inference CLI commands.

Commands for classifying audio files using trained HMM models.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import typer
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ..infer import RaagClassifier, ModelLoader
from ..io import load_audio, load_metadata
from ..pitch import extract_pitch_with_fallback, smooth_pitch
from ..quantize import quantize_sequence
from .utils import handle_error, validate_audio_file, validate_model_directory

console = Console()
logger = logging.getLogger(__name__)

# Create predict subcommand group
predict_app = typer.Typer(
    name="predict",
    help="Prediction and inference commands"
)


@predict_app.command("single")
def predict_single(
    audio_file: Path = typer.Argument(
        ...,
        help="Audio file to classify",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    models_dir: Path = typer.Argument(
        ...,
        help="Directory containing trained models",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    metadata_file: Optional[Path] = typer.Option(
        None,
        "--metadata",
        "-m",
        help="Metadata JSON file containing tonic_hz",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    tonic_hz: Optional[float] = typer.Option(
        None,
        "--tonic",
        "-t",
        help="Tonic frequency in Hz (overrides metadata)"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for prediction results (JSON format)"
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        help="Number of top predictions to show"
    ),
    confidence_threshold: float = typer.Option(
        0.0,
        "--threshold",
        help="Minimum confidence threshold for predictions"
    )
):
    """
    Classify a single audio file using trained HMM models.
    
    This command processes an audio file through the complete pipeline:
    pitch extraction, quantization, and HMM-based classification.
    
    Examples:
    ```
    # Basic prediction
    raag-hmm predict single audio.wav models/
    
    # With metadata for tonic
    raag-hmm predict single audio.wav models/ -m metadata.json
    
    # Specify tonic directly
    raag-hmm predict single audio.wav models/ --tonic 261.63
    
    # Save results to file
    raag-hmm predict single audio.wav models/ -o results.json --top-k 5
    ```
    """
    try:
        audio_file = validate_audio_file(audio_file)
        models_dir = validate_model_directory(models_dir)
        
        console.print(Panel.fit(
            f"[bold]Single Audio Prediction[/bold]\n"
            f"Audio: {audio_file}\n"
            f"Models: {models_dir}\n"
            f"Top-K: {top_k}\n"
            f"Threshold: {confidence_threshold}",
            border_style="blue"
        ))
        
        # Load tonic frequency
        tonic_frequency = None
        if metadata_file:
            metadata = load_metadata(str(metadata_file))
            tonic_frequency = metadata.get('tonic_hz')
            console.print(f"[dim]Loaded tonic from metadata: {tonic_frequency} Hz[/dim]")
        
        if tonic_hz is not None:
            tonic_frequency = tonic_hz
            console.print(f"[dim]Using specified tonic: {tonic_frequency} Hz[/dim]")
        
        if tonic_frequency is None:
            console.print("[red]Tonic frequency required for classification (--tonic or --metadata)[/red]")
            raise typer.Exit(1)
        
        # Load models
        with console.status("[bold blue]Loading models..."):
            classifier = RaagClassifier(str(models_dir))
        
        console.print(f"[green]Loaded {len(classifier.models)} models: {list(classifier.models.keys())}[/green]")
        
        # Process audio
        with console.status("[bold blue]Loading audio..."):
            audio_data = load_audio(str(audio_file))
        
        with console.status("[bold blue]Extracting pitch..."):
            f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)
        
        with console.status("[bold blue]Smoothing pitch..."):
            f0_hz = smooth_pitch(f0_hz, voicing_prob)
        
        with console.status("[bold blue]Quantizing sequence..."):
            quantized = quantize_sequence(f0_hz, tonic_frequency)
        
        # Clean sequence
        valid_indices = ~np.isnan(quantized)
        if np.sum(valid_indices) == 0:
            console.print("[red]No valid pitch detected in audio file[/red]")
            raise typer.Exit(1)
        
        clean_sequence = quantized[valid_indices].astype(int)
        
        console.print(f"[dim]Processed sequence: {len(clean_sequence)} frames, "
                     f"range [{clean_sequence.min()}, {clean_sequence.max()}][/dim]")
        
        # Make prediction
        with console.status("[bold blue]Classifying..."):
            predicted_raag, confidence_scores = classifier.predict(clean_sequence)
        
        # Sort scores for top-k
        sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = {
            "audio_file": str(audio_file),
            "tonic_hz": tonic_frequency,
            "sequence_length": len(clean_sequence),
            "predicted_raag": predicted_raag,
            "confidence_scores": confidence_scores,
            "top_k_predictions": sorted_scores[:top_k]
        }
        
        # Display results
        console.print(f"\n[bold green]Prediction Results[/bold green]")
        console.print(f"[bold]Predicted Raag: [cyan]{predicted_raag}[/cyan][/bold]")
        console.print(f"Confidence: {confidence_scores[predicted_raag]:.4f}")
        
        # Top-K table
        if top_k > 1:
            console.print(f"\n[bold]Top-{top_k} Predictions:[/bold]")
            
            predictions_table = Table()
            predictions_table.add_column("Rank", style="cyan")
            predictions_table.add_column("Raag", style="magenta")
            predictions_table.add_column("Confidence", style="green")
            predictions_table.add_column("Status", style="yellow")
            
            for i, (raag, score) in enumerate(sorted_scores[:top_k], 1):
                status = "✓" if score >= confidence_threshold else "✗"
                predictions_table.add_row(
                    str(i),
                    raag,
                    f"{score:.4f}",
                    status
                )
            
            console.print(predictions_table)
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Results saved to: {output_file}[/green]")
        
        # Check confidence threshold
        if confidence_scores[predicted_raag] < confidence_threshold:
            console.print(f"\n[yellow]Warning: Prediction confidence ({confidence_scores[predicted_raag]:.4f}) "
                         f"below threshold ({confidence_threshold})[/yellow]")
        
    except Exception as e:
        handle_error(e, "single audio prediction")


@predict_app.command("batch")
def predict_batch(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing audio files to classify",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    models_dir: Path = typer.Argument(
        ...,
        help="Directory containing trained models",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Output file for batch prediction results (JSON format)"
    ),
    pattern: str = typer.Option(
        "*.wav",
        "--pattern",
        "-p",
        help="File pattern to match (e.g., '*.wav', '*.flac')"
    ),
    require_metadata: bool = typer.Option(
        True,
        "--require-metadata/--no-require-metadata",
        help="Require corresponding metadata files for tonic information"
    ),
    default_tonic: Optional[float] = typer.Option(
        None,
        "--default-tonic",
        help="Default tonic frequency when metadata is missing"
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        help="Number of top predictions to save for each file"
    )
):
    """
    Classify multiple audio files in batch mode.
    
    This command processes all audio files in a directory and generates
    predictions for each one. Results are saved in a structured JSON format.
    
    Examples:
    ```
    # Batch prediction with metadata
    raag-hmm predict batch audio_dir/ models/ results.json
    
    # Process only FLAC files
    raag-hmm predict batch audio_dir/ models/ results.json --pattern "*.flac"
    
    # Use default tonic when metadata missing
    raag-hmm predict batch audio_dir/ models/ results.json --no-require-metadata --default-tonic 261.63
    ```
    """
    try:
        models_dir = validate_model_directory(models_dir)
        
        console.print(Panel.fit(
            f"[bold]Batch Audio Prediction[/bold]\n"
            f"Input: {input_dir}\n"
            f"Models: {models_dir}\n"
            f"Output: {output_file}\n"
            f"Pattern: {pattern}\n"
            f"Require Metadata: {require_metadata}",
            border_style="blue"
        ))
        
        # Find audio files
        audio_files = list(input_dir.glob(pattern))
        if not audio_files:
            console.print(f"[red]No audio files found matching pattern: {pattern}[/red]")
            raise typer.Exit(1)
        
        console.print(f"Found {len(audio_files)} audio files")
        
        # Load models
        with console.status("[bold blue]Loading models..."):
            classifier = RaagClassifier(str(models_dir))
        
        console.print(f"[green]Loaded {len(classifier.models)} models[/green]")
        
        # Process files
        results = {
            "input_directory": str(input_dir),
            "models_directory": str(models_dir),
            "pattern": pattern,
            "total_files": len(audio_files),
            "predictions": []
        }
        
        successful_predictions = 0
        errors = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing files", total=len(audio_files))
            
            for audio_file in audio_files:
                try:
                    progress.update(task, description=f"Processing {audio_file.name}")
                    
                    # Load tonic frequency
                    tonic_frequency = None
                    
                    if require_metadata:
                        metadata_file = audio_file.with_suffix('.json')
                        if metadata_file.exists():
                            metadata = load_metadata(str(metadata_file))
                            tonic_frequency = metadata.get('tonic_hz')
                        else:
                            errors.append(f"{audio_file.name}: Missing metadata file")
                            continue
                    else:
                        # Try to load metadata, fall back to default
                        metadata_file = audio_file.with_suffix('.json')
                        if metadata_file.exists():
                            try:
                                metadata = load_metadata(str(metadata_file))
                                tonic_frequency = metadata.get('tonic_hz')
                            except:
                                pass
                        
                        if tonic_frequency is None:
                            tonic_frequency = default_tonic
                    
                    if tonic_frequency is None:
                        errors.append(f"{audio_file.name}: No tonic frequency available")
                        continue
                    
                    # Process audio
                    audio_data = load_audio(str(audio_file))
                    f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)
                    f0_hz = smooth_pitch(f0_hz, voicing_prob)
                    quantized = quantize_sequence(f0_hz, tonic_frequency)
                    
                    # Clean sequence
                    valid_indices = ~np.isnan(quantized)
                    if np.sum(valid_indices) == 0:
                        errors.append(f"{audio_file.name}: No valid pitch detected")
                        continue
                    
                    clean_sequence = quantized[valid_indices].astype(int)
                    
                    # Make prediction
                    predicted_raag, confidence_scores = classifier.predict(clean_sequence)
                    
                    # Sort for top-k
                    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Store result
                    file_result = {
                        "filename": audio_file.name,
                        "filepath": str(audio_file),
                        "tonic_hz": tonic_frequency,
                        "sequence_length": len(clean_sequence),
                        "predicted_raag": predicted_raag,
                        "confidence": confidence_scores[predicted_raag],
                        "top_k_predictions": [
                            {"raag": raag, "confidence": score}
                            for raag, score in sorted_scores[:top_k]
                        ]
                    }
                    
                    results["predictions"].append(file_result)
                    successful_predictions += 1
                    
                except Exception as e:
                    errors.append(f"{audio_file.name}: {str(e)}")
                
                progress.update(task, advance=1)
        
        # Save results
        results["successful_predictions"] = successful_predictions
        results["errors"] = errors
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        console.print(f"\n[bold green]Batch prediction completed![/bold green]")
        console.print(f"Results saved to: {output_file}")
        console.print(f"Successful predictions: {successful_predictions}/{len(audio_files)}")
        
        if errors:
            console.print(f"[yellow]Errors: {len(errors)}[/yellow]")
            for error in errors[:5]:  # Show first 5 errors
                console.print(f"  [red]• {error}[/red]")
            if len(errors) > 5:
                console.print(f"  [dim]... and {len(errors) - 5} more errors[/dim]")
        
        # Show prediction summary
        if successful_predictions > 0:
            console.print(f"\n[bold]Prediction Summary:[/bold]")
            
            raag_counts = {}
            for pred in results["predictions"]:
                raag = pred["predicted_raag"]
                raag_counts[raag] = raag_counts.get(raag, 0) + 1
            
            summary_table = Table()
            summary_table.add_column("Predicted Raag", style="cyan")
            summary_table.add_column("Count", style="magenta")
            summary_table.add_column("Percentage", style="green")
            
            for raag, count in sorted(raag_counts.items()):
                percentage = count / successful_predictions * 100
                summary_table.add_row(raag, str(count), f"{percentage:.1f}%")
            
            console.print(summary_table)
        
    except Exception as e:
        handle_error(e, "batch prediction")