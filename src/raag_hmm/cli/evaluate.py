"""
Evaluation CLI commands.

Commands for evaluating model performance on test datasets.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import typer
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ..infer import RaagClassifier
from ..io import iter_dataset, load_audio
from ..pitch import extract_pitch_with_fallback, smooth_pitch
from ..quantize import quantize_sequence
from ..evaluate import (
    compute_comprehensive_metrics,
    compute_confusion_matrix,
    export_confusion_matrix_json,
    export_confusion_matrix_csv,
    export_classification_report
)
from .utils import handle_error, validate_dataset_directory, validate_model_directory

console = Console()
logger = logging.getLogger(__name__)

# Create evaluate subcommand group
evaluate_app = typer.Typer(
    name="evaluate",
    help="Model evaluation commands"
)


@evaluate_app.command("test")
def evaluate_test_set(
    dataset_dir: Path = typer.Argument(
        ...,
        help="Dataset directory containing test data",
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
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for evaluation results"
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Dataset split to evaluate (test/val)"
    ),
    top_k_values: Optional[List[int]] = typer.Option(
        [1, 3, 5],
        "--top-k",
        help="Top-K accuracy values to compute (can be used multiple times)"
    ),
    export_format: List[str] = typer.Option(
        ["json", "csv"],
        "--format",
        help="Export formats for results (json/csv)"
    ),
    detailed_errors: bool = typer.Option(
        True,
        "--detailed-errors/--no-detailed-errors",
        help="Include detailed error analysis"
    )
):
    """
    Evaluate trained models on test dataset.
    
    This command runs comprehensive evaluation of trained HMM models on a test
    dataset, computing accuracy metrics, confusion matrices, and error analysis.
    
    Examples:
    ```
    # Basic evaluation
    raag-hmm evaluate test dataset/ models/ results/
    
    # Evaluate validation set with custom top-k
    raag-hmm evaluate test dataset/ models/ results/ --split val --top-k 1 --top-k 5
    
    # Export only JSON format
    raag-hmm evaluate test dataset/ models/ results/ --format json
    ```
    """
    try:
        dataset_dir = validate_dataset_directory(dataset_dir)
        models_dir = validate_model_directory(models_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(Panel.fit(
            f"[bold]Model Evaluation[/bold]\n"
            f"Dataset: {dataset_dir}\n"
            f"Models: {models_dir}\n"
            f"Output: {output_dir}\n"
            f"Split: {split}\n"
            f"Top-K: {top_k_values}\n"
            f"Formats: {export_format}",
            border_style="blue"
        ))
        
        # Load models
        with console.status("[bold blue]Loading models..."):
            classifier = RaagClassifier(str(models_dir))
        
        console.print(f"[green]Loaded {len(classifier.models)} models: {list(classifier.models.keys())}[/green]")
        
        # Load test data
        console.print(f"[bold]Loading {split} data...[/bold]")
        
        test_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            load_task = progress.add_task("Loading test sequences...", total=None)
            
            for audio_path, metadata in iter_dataset(str(dataset_dir), split):
                try:
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
                        
                        test_data.append({
                            'audio_path': audio_path,
                            'true_raag': metadata['raag'],
                            'sequence': clean_sequence,
                            'tonic_hz': tonic_hz,
                            'recording_id': metadata.get('recording_id', Path(audio_path).stem)
                        })
                    
                    progress.update(load_task, description=f"Loaded {len(test_data)} sequences...")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to process {audio_path}: {e}[/yellow]")
        
        if not test_data:
            console.print(f"[red]No test data found in {split} split![/red]")
            raise typer.Exit(1)
        
        console.print(f"Loaded {len(test_data)} test sequences")
        
        # Make predictions
        console.print(f"[bold]Making predictions...[/bold]")
        
        predictions = []
        true_labels = []
        confidence_scores_list = []
        detailed_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            pred_task = progress.add_task("Making predictions", total=len(test_data))
            
            for item in test_data:
                try:
                    predicted_raag, confidence_scores = classifier.predict(item['sequence'])
                    
                    predictions.append(predicted_raag)
                    true_labels.append(item['true_raag'])
                    confidence_scores_list.append(confidence_scores)
                    
                    # Store detailed result
                    detailed_result = {
                        'recording_id': item['recording_id'],
                        'audio_path': item['audio_path'],
                        'true_raag': item['true_raag'],
                        'predicted_raag': predicted_raag,
                        'correct': predicted_raag == item['true_raag'],
                        'confidence': confidence_scores[predicted_raag],
                        'all_scores': confidence_scores,
                        'sequence_length': len(item['sequence']),
                        'tonic_hz': item['tonic_hz']
                    }
                    
                    detailed_results.append(detailed_result)
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Prediction failed for {item['recording_id']}: {e}[/yellow]")
                
                progress.update(pred_task, advance=1)
        
        if not predictions:
            console.print("[red]No predictions made![/red]")
            raise typer.Exit(1)
        
        # Compute metrics
        console.print("[bold]Computing evaluation metrics...[/bold]")
        
        # Get unique class labels
        all_labels = sorted(set(true_labels + predictions))
        
        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            true_labels, 
            predictions, 
            confidence_scores_list,
            class_labels=all_labels,
            top_k_values=top_k_values
        )
        
        # Compute confusion matrix
        confusion_matrix = compute_confusion_matrix(true_labels, predictions, all_labels)
        
        # Display results
        console.print(f"\n[bold green]Evaluation Results[/bold green]")
        
        # Overall metrics table
        metrics_table = Table(title="Overall Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")
        
        metrics_table.add_row("Overall Accuracy", f"{metrics['overall_accuracy']:.4f}")
        metrics_table.add_row("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")
        
        for k in top_k_values:
            if f'top_{k}_accuracy' in metrics:
                metrics_table.add_row(f"Top-{k} Accuracy", f"{metrics[f'top_{k}_accuracy']:.4f}")
        
        metrics_table.add_row("Mean Confidence", f"{metrics['confidence_stats']['mean']:.4f}")
        metrics_table.add_row("Std Confidence", f"{metrics['confidence_stats']['std']:.4f}")
        
        console.print(metrics_table)
        
        # Per-class accuracy table
        if 'per_class_accuracy' in metrics:
            console.print(f"\n[bold]Per-Class Accuracy:[/bold]")
            
            class_table = Table()
            class_table.add_column("Raag", style="cyan")
            class_table.add_column("Accuracy", style="magenta")
            class_table.add_column("Support", style="green")
            
            for raag in all_labels:
                accuracy = metrics['per_class_accuracy'].get(raag, 0.0)
                support = metrics['class_support'].get(raag, 0)
                class_table.add_row(raag, f"{accuracy:.4f}", str(support))
            
            console.print(class_table)
        
        # Confusion matrix (simplified view)
        console.print(f"\n[bold]Confusion Matrix:[/bold]")
        
        cm_table = Table()
        cm_table.add_column("True \\ Pred", style="cyan")
        for label in all_labels:
            cm_table.add_column(label[:8], style="magenta")  # Truncate long names
        
        for i, true_label in enumerate(all_labels):
            row = [true_label[:8]]
            for j, pred_label in enumerate(all_labels):
                count = confusion_matrix[i, j]
                if i == j:  # Diagonal (correct predictions)
                    row.append(f"[green]{count}[/green]")
                else:
                    row.append(str(count) if count > 0 else "[dim]0[/dim]")
            cm_table.add_row(*row)
        
        console.print(cm_table)
        
        # Save results
        console.print(f"\n[bold]Saving results...[/bold]")
        
        # Prepare complete results
        evaluation_results = {
            "dataset_info": {
                "dataset_dir": str(dataset_dir),
                "split": split,
                "total_samples": len(test_data),
                "successful_predictions": len(predictions)
            },
            "model_info": {
                "models_dir": str(models_dir),
                "available_models": list(classifier.models.keys()),
                "num_models": len(classifier.models)
            },
            "metrics": metrics,
            "confusion_matrix": confusion_matrix.tolist(),
            "class_labels": all_labels
        }
        
        if detailed_errors:
            evaluation_results["detailed_results"] = detailed_results
        
        # Export in requested formats
        for fmt in export_format:
            if fmt == "json":
                json_file = output_dir / "evaluation_results.json"
                with open(json_file, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
                console.print(f"[green]JSON results saved to: {json_file}[/green]")
                
                # Export confusion matrix separately
                cm_json_file = output_dir / "confusion_matrix.json"
                export_confusion_matrix_json(confusion_matrix, all_labels, str(cm_json_file))
                
            elif fmt == "csv":
                # Export confusion matrix as CSV
                cm_csv_file = output_dir / "confusion_matrix.csv"
                export_confusion_matrix_csv(confusion_matrix, all_labels, str(cm_csv_file))
                console.print(f"[green]CSV confusion matrix saved to: {cm_csv_file}[/green]")
                
                # Export classification report as CSV
                report_csv_file = output_dir / "classification_report.csv"
                export_classification_report(
                    true_labels, predictions, all_labels, str(report_csv_file)
                )
                console.print(f"[green]Classification report saved to: {report_csv_file}[/green]")
        
        # Summary
        console.print(f"\n[bold green]Evaluation completed![/bold green]")
        console.print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
        console.print(f"Results saved to: {output_dir}")
        
        # Check if accuracy is below expected threshold
        if metrics['overall_accuracy'] < 0.5:
            console.print("[yellow]Warning: Low accuracy detected. Consider retraining models.[/yellow]")
        
    except Exception as e:
        handle_error(e, "model evaluation")


@evaluate_app.command("compare")
def compare_models(
    dataset_dir: Path = typer.Argument(
        ...,
        help="Dataset directory containing test data",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    models_dirs: List[Path] = typer.Argument(
        ...,
        help="Multiple model directories to compare"
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Output file for comparison results (JSON format)"
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Dataset split to evaluate"
    ),
    model_names: Optional[List[str]] = typer.Option(
        None,
        "--name",
        help="Names for model sets (must match number of directories)"
    )
):
    """
    Compare performance of multiple model sets.
    
    This command evaluates multiple sets of trained models on the same test
    dataset and provides a comparative analysis of their performance.
    
    Examples:
    ```
    # Compare two model sets
    raag-hmm evaluate compare dataset/ models_v1/ models_v2/ comparison.json
    
    # Compare with custom names
    raag-hmm evaluate compare dataset/ models_v1/ models_v2/ comparison.json \\
        --name "Baseline" --name "Improved"
    ```
    """
    try:
        dataset_dir = validate_dataset_directory(dataset_dir)
        
        # Validate model directories
        for models_dir in models_dirs:
            validate_model_directory(models_dir)
        
        # Generate model names if not provided
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(models_dirs))]
        elif len(model_names) != len(models_dirs):
            console.print("[red]Number of model names must match number of model directories[/red]")
            raise typer.Exit(1)
        
        console.print(Panel.fit(
            f"[bold]Model Comparison[/bold]\n"
            f"Dataset: {dataset_dir}\n"
            f"Split: {split}\n"
            f"Models: {len(models_dirs)}\n"
            f"Names: {model_names}",
            border_style="blue"
        ))
        
        # Load test data once
        console.print(f"[bold]Loading {split} data...[/bold]")
        
        test_data = []
        for audio_path, metadata in iter_dataset(str(dataset_dir), split):
            try:
                audio_data = load_audio(audio_path)
                f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)
                f0_hz = smooth_pitch(f0_hz, voicing_prob)
                
                tonic_hz = metadata['tonic_hz']
                quantized = quantize_sequence(f0_hz, tonic_hz)
                
                valid_indices = ~np.isnan(quantized)
                if np.sum(valid_indices) > 0:
                    clean_sequence = quantized[valid_indices].astype(int)
                    test_data.append({
                        'true_raag': metadata['raag'],
                        'sequence': clean_sequence,
                        'recording_id': metadata.get('recording_id', Path(audio_path).stem)
                    })
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to process {audio_path}: {e}[/yellow]")
        
        console.print(f"Loaded {len(test_data)} test sequences")
        
        # Evaluate each model set
        comparison_results = {
            "dataset_info": {
                "dataset_dir": str(dataset_dir),
                "split": split,
                "total_samples": len(test_data)
            },
            "model_comparisons": {}
        }
        
        all_results = {}
        
        for model_name, models_dir in zip(model_names, models_dirs):
            console.print(f"\n[bold]Evaluating {model_name}...[/bold]")
            
            # Load models
            classifier = RaagClassifier(str(models_dir))
            
            # Make predictions
            predictions = []
            true_labels = []
            confidence_scores_list = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[progress.description]{{task.description}} ({model_name})"),
                BarColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Predicting", total=len(test_data))
                
                for item in test_data:
                    try:
                        predicted_raag, confidence_scores = classifier.predict(item['sequence'])
                        predictions.append(predicted_raag)
                        true_labels.append(item['true_raag'])
                        confidence_scores_list.append(confidence_scores)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Prediction failed: {e}[/yellow]")
                    
                    progress.update(task, advance=1)
            
            # Compute metrics
            all_labels = sorted(set(true_labels + predictions))
            metrics = compute_comprehensive_metrics(
                true_labels, predictions, confidence_scores_list, class_labels=all_labels
            )
            
            all_results[model_name] = {
                "models_dir": str(models_dir),
                "metrics": metrics,
                "predictions": predictions,
                "true_labels": true_labels
            }
            
            comparison_results["model_comparisons"][model_name] = {
                "models_dir": str(models_dir),
                "metrics": metrics
            }
        
        # Create comparison table
        console.print(f"\n[bold green]Comparison Results[/bold green]")
        
        comp_table = Table(title="Model Performance Comparison")
        comp_table.add_column("Model", style="cyan")
        comp_table.add_column("Overall Accuracy", style="magenta")
        comp_table.add_column("Balanced Accuracy", style="green")
        comp_table.add_column("Top-3 Accuracy", style="yellow")
        comp_table.add_column("Mean Confidence", style="blue")
        
        best_accuracy = 0
        best_model = None
        
        for model_name, results in all_results.items():
            metrics = results["metrics"]
            accuracy = metrics["overall_accuracy"]
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
            
            comp_table.add_row(
                model_name,
                f"{accuracy:.4f}",
                f"{metrics['balanced_accuracy']:.4f}",
                f"{metrics.get('top_3_accuracy', 0.0):.4f}",
                f"{metrics['confidence_stats']['mean']:.4f}"
            )
        
        console.print(comp_table)
        
        if best_model:
            console.print(f"\n[bold green]Best performing model: {best_model} "
                         f"(accuracy: {best_accuracy:.4f})[/bold green]")
        
        # Save comparison results
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        console.print(f"\n[green]Comparison results saved to: {output_file}[/green]")
        
    except Exception as e:
        handle_error(e, "model comparison")