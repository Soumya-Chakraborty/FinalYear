"""
Dataset preparation CLI commands.

Commands for preparing datasets, extracting pitch, and preprocessing audio files.
"""

import json
from pathlib import Path
from typing import Optional, List
import logging

import typer
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ..io import load_audio, load_metadata, iter_dataset
from ..pitch import extract_pitch_with_fallback, smooth_pitch
from ..quantize import quantize_sequence
from .utils import handle_error, validate_audio_file, validate_dataset_directory

console = Console()
logger = logging.getLogger(__name__)

# Create dataset subcommand group
dataset_app = typer.Typer(
    name="dataset",
    help="Dataset preparation and preprocessing commands"
)


@dataset_app.command("prepare")
def prepare_dataset(
    input_dir: Path = typer.Argument(
        ...,
        help="Input dataset directory containing audio files and metadata",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for processed dataset"
    ),
    sample_rate: int = typer.Option(
        22050,
        "--sample-rate",
        "-sr",
        help="Target sample rate for audio resampling"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing output directory"
    ),
    validate_only: bool = typer.Option(
        False,
        "--validate-only",
        help="Only validate dataset structure without processing"
    )
):
    """
    Prepare dataset by resampling audio files and validating metadata.
    
    This command processes a dataset directory containing audio files and their
    corresponding metadata JSON files. It resamples all audio to the specified
    sample rate and validates metadata structure.
    
    Expected directory structure:
    ```
    input_dir/
    ├── train/
    │   ├── audio1.wav
    │   ├── audio1.json
    │   └── ...
    └── test/
        ├── audio2.wav
        ├── audio2.json
        └── ...
    ```
    """
    try:
        input_dir = validate_dataset_directory(input_dir)
        
        # Check output directory
        if output_dir.exists() and not force:
            console.print(f"[red]Output directory already exists: {output_dir}[/red]")
            console.print("[yellow]Use --force to overwrite[/yellow]")
            raise typer.Exit(1)
        
        # Create output directory
        if not validate_only:
            output_dir.mkdir(parents=True, exist_ok=force)
        
        console.print(Panel.fit(
            f"[bold]Dataset Preparation[/bold]\n"
            f"Input: {input_dir}\n"
            f"Output: {output_dir}\n"
            f"Sample Rate: {sample_rate} Hz\n"
            f"Mode: {'Validation Only' if validate_only else 'Full Processing'}",
            border_style="blue"
        ))
        
        # Process each split
        splits = ['train', 'test', 'val']
        total_files = 0
        processed_files = 0
        errors = []
        
        for split in splits:
            split_dir = input_dir / split
            if not split_dir.exists():
                console.print(f"[yellow]Skipping missing split: {split}[/yellow]")
                continue
            
            output_split_dir = output_dir / split
            if not validate_only:
                output_split_dir.mkdir(exist_ok=True)
            
            console.print(f"\n[bold]Processing {split} split...[/bold]")
            
            # Count files for progress
            audio_files = list(split_dir.glob("*.wav")) + list(split_dir.glob("*.flac")) + list(split_dir.glob("*.mp3"))
            total_files += len(audio_files)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Processing {split}", total=len(audio_files))
                
                for audio_file in audio_files:
                    try:
                        # Check for corresponding metadata
                        metadata_file = audio_file.with_suffix('.json')
                        if not metadata_file.exists():
                            errors.append(f"Missing metadata for {audio_file.name}")
                            continue
                        
                        # Validate metadata
                        metadata = load_metadata(str(metadata_file))
                        
                        if not validate_only:
                            # Load and resample audio
                            audio_data = load_audio(str(audio_file), sr=sample_rate)
                            
                            # Save processed audio
                            output_audio = output_split_dir / audio_file.name
                            import soundfile as sf
                            sf.write(str(output_audio), audio_data, sample_rate)
                            
                            # Copy metadata
                            output_metadata = output_split_dir / metadata_file.name
                            with open(output_metadata, 'w') as f:
                                json.dump(metadata, f, indent=2)
                        
                        processed_files += 1
                        progress.update(task, advance=1)
                        
                    except Exception as e:
                        errors.append(f"Error processing {audio_file.name}: {e}")
                        progress.update(task, advance=1)
        
        # Display results
        console.print(f"\n[bold green]Dataset preparation completed![/bold green]")
        console.print(f"Total files found: {total_files}")
        console.print(f"Successfully processed: {processed_files}")
        
        if errors:
            console.print(f"[yellow]Errors encountered: {len(errors)}[/yellow]")
            for error in errors[:5]:  # Show first 5 errors
                console.print(f"  [red]• {error}[/red]")
            if len(errors) > 5:
                console.print(f"  [dim]... and {len(errors) - 5} more errors[/dim]")
        
    except Exception as e:
        handle_error(e, "dataset preparation")


@dataset_app.command("extract-pitch")
def extract_pitch_command(
    audio_file: Path = typer.Argument(
        ...,
        help="Input audio file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for pitch data (JSON format). If not specified, prints to stdout"
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
    method: str = typer.Option(
        "auto",
        "--method",
        help="Pitch extraction method: 'praat', 'librosa', or 'auto' for fallback"
    ),
    smooth: bool = typer.Option(
        True,
        "--smooth/--no-smooth",
        help="Apply pitch smoothing"
    ),
    quantize: bool = typer.Option(
        False,
        "--quantize",
        help="Quantize pitch to chromatic bins"
    )
):
    """
    Extract pitch contour from a single audio file.
    
    This command extracts the fundamental frequency (F0) contour from an audio file
    using the specified method. Optionally applies smoothing and quantization.
    
    Examples:
    ```
    # Basic pitch extraction
    raag-hmm dataset extract-pitch audio.wav
    
    # Extract and save to file
    raag-hmm dataset extract-pitch audio.wav -o pitch.json
    
    # Extract with metadata for tonic normalization
    raag-hmm dataset extract-pitch audio.wav -m metadata.json --quantize
    
    # Specify tonic directly
    raag-hmm dataset extract-pitch audio.wav --tonic 261.63 --quantize
    ```
    """
    try:
        audio_file = validate_audio_file(audio_file)
        
        console.print(Panel.fit(
            f"[bold]Pitch Extraction[/bold]\n"
            f"Audio: {audio_file}\n"
            f"Method: {method}\n"
            f"Smoothing: {'Yes' if smooth else 'No'}\n"
            f"Quantization: {'Yes' if quantize else 'No'}",
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
        
        # Load audio
        with console.status("[bold blue]Loading audio..."):
            audio_data = load_audio(str(audio_file))
        
        # Extract pitch
        with console.status("[bold blue]Extracting pitch..."):
            if method == "auto":
                f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)
            elif method == "praat":
                from ..pitch import extract_pitch_praat
                f0_hz, voicing_prob = extract_pitch_praat(audio_data, sr=22050)
            elif method == "librosa":
                from ..pitch import extract_pitch_librosa
                f0_hz, voicing_prob = extract_pitch_librosa(audio_data, sr=22050)
            else:
                console.print(f"[red]Unknown method: {method}[/red]")
                raise typer.Exit(1)
        
        # Apply smoothing
        if smooth:
            with console.status("[bold blue]Smoothing pitch..."):
                f0_hz = smooth_pitch(f0_hz, voicing_prob)
        
        # Prepare output data
        result = {
            "audio_file": str(audio_file),
            "method": method,
            "sample_rate": 22050,
            "frame_count": len(f0_hz),
            "f0_hz": f0_hz.tolist(),
            "voicing_prob": voicing_prob.tolist() if voicing_prob is not None else None,
            "smoothed": smooth
        }
        
        # Add quantization if requested
        if quantize:
            if tonic_frequency is None:
                console.print("[red]Quantization requires tonic frequency (--tonic or --metadata)[/red]")
                raise typer.Exit(1)
            
            with console.status("[bold blue]Quantizing pitch..."):
                quantized = quantize_sequence(f0_hz, tonic_frequency)
                result["quantized"] = quantized.tolist()
                result["tonic_hz"] = tonic_frequency
        
        # Output results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            console.print(f"[green]Pitch data saved to: {output_file}[/green]")
        else:
            console.print_json(data=result)
        
        # Display summary
        valid_frames = np.sum(~np.isnan(f0_hz))
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"Total frames: {len(f0_hz)}")
        console.print(f"Valid pitch frames: {valid_frames} ({valid_frames/len(f0_hz)*100:.1f}%)")
        
        if valid_frames > 0:
            valid_f0 = f0_hz[~np.isnan(f0_hz)]
            console.print(f"Pitch range: {valid_f0.min():.1f} - {valid_f0.max():.1f} Hz")
            console.print(f"Mean pitch: {valid_f0.mean():.1f} Hz")
        
    except Exception as e:
        handle_error(e, "pitch extraction")


@dataset_app.command("validate")
def validate_dataset_command(
    dataset_dir: Path = typer.Argument(
        ...,
        help="Dataset directory to validate",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    split: Optional[str] = typer.Option(
        None,
        "--split",
        help="Validate specific split only (train/test/val)"
    ),
    check_audio: bool = typer.Option(
        True,
        "--check-audio/--no-check-audio",
        help="Validate audio file integrity"
    )
):
    """
    Validate dataset structure and integrity.
    
    This command checks that the dataset follows the expected structure,
    validates metadata files, and optionally checks audio file integrity.
    """
    try:
        dataset_dir = validate_dataset_directory(dataset_dir)
        
        console.print(Panel.fit(
            f"[bold]Dataset Validation[/bold]\n"
            f"Directory: {dataset_dir}\n"
            f"Split: {split or 'All'}\n"
            f"Audio Check: {'Yes' if check_audio else 'No'}",
            border_style="blue"
        ))
        
        splits_to_check = [split] if split else ['train', 'test', 'val']
        
        total_files = 0
        valid_files = 0
        errors = []
        raag_counts = {}
        
        for split_name in splits_to_check:
            split_dir = dataset_dir / split_name
            if not split_dir.exists():
                console.print(f"[yellow]Split directory not found: {split_name}[/yellow]")
                continue
            
            console.print(f"\n[bold]Validating {split_name} split...[/bold]")
            
            # Find all metadata files
            metadata_files = list(split_dir.glob("*.json"))
            total_files += len(metadata_files)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Validating {split_name}", total=len(metadata_files))
                
                for metadata_file in metadata_files:
                    try:
                        # Validate metadata
                        metadata = load_metadata(str(metadata_file))
                        
                        # Check required fields
                        required_fields = ['recording_id', 'raag', 'tonic_hz']
                        for field in required_fields:
                            if field not in metadata:
                                errors.append(f"{metadata_file.name}: Missing field '{field}'")
                                continue
                        
                        # Count raag classes
                        raag = metadata.get('raag')
                        if raag:
                            raag_counts[raag] = raag_counts.get(raag, 0) + 1
                        
                        # Check for corresponding audio file
                        audio_file = metadata_file.with_suffix('.wav')
                        if not audio_file.exists():
                            # Try other formats
                            for ext in ['.flac', '.mp3']:
                                audio_file = metadata_file.with_suffix(ext)
                                if audio_file.exists():
                                    break
                            else:
                                errors.append(f"{metadata_file.name}: No corresponding audio file found")
                                continue
                        
                        # Validate audio if requested
                        if check_audio:
                            try:
                                audio_data = load_audio(str(audio_file))
                                if len(audio_data) == 0:
                                    errors.append(f"{audio_file.name}: Empty audio file")
                            except Exception as e:
                                errors.append(f"{audio_file.name}: Audio loading error - {e}")
                        
                        valid_files += 1
                        progress.update(task, advance=1)
                        
                    except Exception as e:
                        errors.append(f"{metadata_file.name}: Validation error - {e}")
                        progress.update(task, advance=1)
        
        # Display results
        console.print(f"\n[bold green]Validation completed![/bold green]")
        
        # Summary table
        table = Table(title="Validation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total files", str(total_files))
        table.add_row("Valid files", str(valid_files))
        table.add_row("Errors", str(len(errors)))
        table.add_row("Success rate", f"{valid_files/total_files*100:.1f}%" if total_files > 0 else "N/A")
        
        console.print(table)
        
        # Raag distribution
        if raag_counts:
            console.print(f"\n[bold]Raag Distribution:[/bold]")
            raag_table = Table()
            raag_table.add_column("Raag", style="cyan")
            raag_table.add_column("Count", style="magenta")
            
            for raag, count in sorted(raag_counts.items()):
                raag_table.add_row(raag, str(count))
            
            console.print(raag_table)
        
        # Show errors
        if errors:
            console.print(f"\n[red]Errors found ({len(errors)}):[/red]")
            for error in errors[:10]:  # Show first 10 errors
                console.print(f"  [red]• {error}[/red]")
            if len(errors) > 10:
                console.print(f"  [dim]... and {len(errors) - 10} more errors[/dim]")
        
        if len(errors) > 0:
            raise typer.Exit(1)
        
    except Exception as e:
        handle_error(e, "dataset validation")