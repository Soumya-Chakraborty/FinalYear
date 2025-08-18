"""
Accuracy metrics computation for raag classification evaluation.

This module implements comprehensive metrics including overall accuracy,
per-class accuracy, top-k accuracy, and statistical analysis functions.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter, defaultdict
import warnings
import json
import csv
from pathlib import Path

from ..logger import get_logger

logger = get_logger(__name__)


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    Compute overall accuracy across test set.
    
    Args:
        y_true: List of true raag labels
        y_pred: List of predicted raag labels
        
    Returns:
        Overall accuracy as float between 0 and 1
        
    Raises:
        ValueError: If input lists have different lengths or are empty
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Empty input lists provided")
    
    # Convert to numpy arrays for efficient computation
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Compute accuracy
    correct_predictions = np.sum(y_true_arr == y_pred_arr)
    total_predictions = len(y_true_arr)
    
    accuracy = correct_predictions / total_predictions
    
    logger.debug(f"Overall accuracy: {correct_predictions}/{total_predictions} = {accuracy:.4f}")
    
    return accuracy


def compute_per_class_accuracy(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Compute per-class accuracy for individual raag analysis.
    
    Args:
        y_true: List of true raag labels
        y_pred: List of predicted raag labels
        
    Returns:
        Dictionary mapping raag names to their individual accuracy scores
        
    Raises:
        ValueError: If input lists have different lengths or are empty
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Empty input lists provided")
    
    # Get unique classes
    unique_classes = sorted(set(y_true))
    
    per_class_accuracy = {}
    
    for class_name in unique_classes:
        # Find indices where true label is this class
        class_indices = [i for i, label in enumerate(y_true) if label == class_name]
        
        if not class_indices:
            # This shouldn't happen given we got classes from y_true, but be safe
            per_class_accuracy[class_name] = 0.0
            continue
        
        # Count correct predictions for this class
        correct_for_class = sum(1 for i in class_indices if y_pred[i] == class_name)
        total_for_class = len(class_indices)
        
        class_accuracy = correct_for_class / total_for_class
        per_class_accuracy[class_name] = class_accuracy
        
        logger.debug(f"Class {class_name}: {correct_for_class}/{total_for_class} = {class_accuracy:.4f}")
    
    return per_class_accuracy


def compute_top_k_accuracy(y_true: List[str], y_scores: List[Dict[str, float]], k: int = 3) -> float:
    """
    Compute top-k accuracy metrics (especially top-3 as specified).
    
    Args:
        y_true: List of true raag labels
        y_scores: List of dictionaries containing scores for all classes
        k: Number of top predictions to consider (default: 3)
        
    Returns:
        Top-k accuracy as float between 0 and 1
        
    Raises:
        ValueError: If input lists have different lengths, are empty, or k is invalid
    """
    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_scores={len(y_scores)}")
    
    if len(y_true) == 0:
        raise ValueError("Empty input lists provided")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    correct_top_k = 0
    total_predictions = len(y_true)
    
    for i, (true_label, scores_dict) in enumerate(zip(y_true, y_scores)):
        if not scores_dict:
            logger.warning(f"Empty scores dictionary at index {i}")
            continue
        
        # Sort classes by score (highest first) and take top k
        sorted_classes = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)
        top_k_predictions = sorted_classes[:k]
        
        # Check if true label is in top k predictions
        if true_label in top_k_predictions:
            correct_top_k += 1
    
    top_k_accuracy = correct_top_k / total_predictions
    
    logger.debug(f"Top-{k} accuracy: {correct_top_k}/{total_predictions} = {top_k_accuracy:.4f}")
    
    return top_k_accuracy


def compute_multiple_top_k_accuracies(y_true: List[str], y_scores: List[Dict[str, float]], 
                                    k_values: List[int] = [1, 3, 5]) -> Dict[int, float]:
    """
    Compute multiple top-k accuracy metrics efficiently.
    
    Args:
        y_true: List of true raag labels
        y_scores: List of dictionaries containing scores for all classes
        k_values: List of k values to compute (default: [1, 3, 5])
        
    Returns:
        Dictionary mapping k values to their corresponding accuracies
        
    Raises:
        ValueError: If input validation fails
    """
    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_scores={len(y_scores)}")
    
    if len(y_true) == 0:
        raise ValueError("Empty input lists provided")
    
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("All k values must be positive")
    
    max_k = max(k_values)
    total_predictions = len(y_true)
    
    # Count correct predictions for each k
    correct_counts = {k: 0 for k in k_values}
    
    for i, (true_label, scores_dict) in enumerate(zip(y_true, y_scores)):
        if not scores_dict:
            logger.warning(f"Empty scores dictionary at index {i}")
            continue
        
        # Sort classes by score (highest first)
        sorted_classes = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)
        
        # Find position of true label (1-indexed)
        try:
            true_label_position = sorted_classes.index(true_label) + 1
        except ValueError:
            # True label not in scores (shouldn't happen in normal cases)
            logger.warning(f"True label '{true_label}' not found in scores at index {i}")
            true_label_position = float('inf')
        
        # Update counts for all k values where true label is in top k
        for k in k_values:
            if true_label_position <= k:
                correct_counts[k] += 1
    
    # Compute accuracies
    accuracies = {k: correct_counts[k] / total_predictions for k in k_values}
    
    logger.debug(f"Multiple top-k accuracies: {accuracies}")
    
    return accuracies


def compute_balanced_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    Compute balanced accuracy (average of per-class accuracies).
    
    This metric is useful when classes are imbalanced, as it gives equal
    weight to each class regardless of support.
    
    Args:
        y_true: List of true raag labels
        y_pred: List of predicted raag labels
        
    Returns:
        Balanced accuracy as float between 0 and 1
        
    Raises:
        ValueError: If input validation fails
    """
    per_class_acc = compute_per_class_accuracy(y_true, y_pred)
    
    if not per_class_acc:
        raise ValueError("No classes found for balanced accuracy computation")
    
    balanced_acc = np.mean(list(per_class_acc.values()))
    
    logger.debug(f"Balanced accuracy: {balanced_acc:.4f} (avg of {len(per_class_acc)} classes)")
    
    return balanced_acc


def compute_class_support(y_true: List[str]) -> Dict[str, int]:
    """
    Compute support (number of samples) for each class.
    
    Args:
        y_true: List of true raag labels
        
    Returns:
        Dictionary mapping class names to their support counts
    """
    if not y_true:
        return {}
    
    support = Counter(y_true)
    
    logger.debug(f"Class support: {dict(support)}")
    
    return dict(support)


def compute_confidence_statistics(y_scores: List[Dict[str, float]], 
                                y_true: List[str]) -> Dict[str, Any]:
    """
    Compute statistics about prediction confidence scores.
    
    Args:
        y_scores: List of dictionaries containing scores for all classes
        y_true: List of true raag labels (for correct/incorrect analysis)
        
    Returns:
        Dictionary containing confidence statistics
        
    Raises:
        ValueError: If input validation fails
    """
    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_scores={len(y_scores)}")
    
    if len(y_true) == 0:
        return {}
    
    max_scores = []  # Confidence of top prediction
    true_label_scores = []  # Score of true label
    score_margins = []  # Difference between top 2 scores
    correct_predictions = []  # Boolean array for correct/incorrect
    
    for true_label, scores_dict in zip(y_true, y_scores):
        if not scores_dict:
            continue
        
        # Sort scores in descending order
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Max score (confidence of top prediction)
        max_score = sorted_items[0][1]
        max_scores.append(max_score)
        
        # Score of true label
        true_score = scores_dict.get(true_label, float('-inf'))
        true_label_scores.append(true_score)
        
        # Score margin (difference between top 2)
        if len(sorted_items) >= 2:
            margin = sorted_items[0][1] - sorted_items[1][1]
        else:
            margin = 0.0
        score_margins.append(margin)
        
        # Correctness
        predicted_label = sorted_items[0][0]
        correct_predictions.append(predicted_label == true_label)
    
    if not max_scores:
        return {}
    
    # Convert to numpy arrays for statistics
    max_scores = np.array(max_scores)
    true_label_scores = np.array(true_label_scores)
    score_margins = np.array(score_margins)
    correct_predictions = np.array(correct_predictions)
    
    # Compute statistics
    stats = {
        'max_score_stats': {
            'mean': float(np.mean(max_scores)),
            'std': float(np.std(max_scores)),
            'min': float(np.min(max_scores)),
            'max': float(np.max(max_scores)),
            'median': float(np.median(max_scores))
        },
        'true_label_score_stats': {
            'mean': float(np.mean(true_label_scores)),
            'std': float(np.std(true_label_scores)),
            'min': float(np.min(true_label_scores)),
            'max': float(np.max(true_label_scores)),
            'median': float(np.median(true_label_scores))
        },
        'score_margin_stats': {
            'mean': float(np.mean(score_margins)),
            'std': float(np.std(score_margins)),
            'min': float(np.min(score_margins)),
            'max': float(np.max(score_margins)),
            'median': float(np.median(score_margins))
        }
    }
    
    # Separate statistics for correct vs incorrect predictions
    if np.any(correct_predictions):
        correct_max_scores = max_scores[correct_predictions]
        stats['correct_predictions'] = {
            'count': int(np.sum(correct_predictions)),
            'max_score_mean': float(np.mean(correct_max_scores)),
            'max_score_std': float(np.std(correct_max_scores))
        }
    
    if np.any(~correct_predictions):
        incorrect_max_scores = max_scores[~correct_predictions]
        stats['incorrect_predictions'] = {
            'count': int(np.sum(~correct_predictions)),
            'max_score_mean': float(np.mean(incorrect_max_scores)),
            'max_score_std': float(np.std(incorrect_max_scores))
        }
    
    logger.debug(f"Confidence statistics computed for {len(max_scores)} predictions")
    
    return stats


def compute_comprehensive_metrics(y_true: List[str], y_pred: List[str], 
                                y_scores: Optional[List[Dict[str, float]]] = None,
                                k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
    """
    Compute comprehensive accuracy metrics in a single function call.
    
    Args:
        y_true: List of true raag labels
        y_pred: List of predicted raag labels
        y_scores: Optional list of score dictionaries for top-k metrics
        k_values: List of k values for top-k accuracy (default: [1, 3, 5])
        
    Returns:
        Dictionary containing all computed metrics
        
    Raises:
        ValueError: If input validation fails
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Empty input lists provided")
    
    # Basic accuracy metrics
    metrics = {
        'overall_accuracy': compute_accuracy(y_true, y_pred),
        'per_class_accuracy': compute_per_class_accuracy(y_true, y_pred),
        'balanced_accuracy': compute_balanced_accuracy(y_true, y_pred),
        'class_support': compute_class_support(y_true),
        'total_samples': len(y_true),
        'num_classes': len(set(y_true))
    }
    
    # Top-k accuracy metrics (if scores provided)
    if y_scores is not None:
        if len(y_scores) != len(y_true):
            logger.warning(f"Scores length mismatch: {len(y_scores)} vs {len(y_true)}")
        else:
            try:
                metrics['top_k_accuracies'] = compute_multiple_top_k_accuracies(
                    y_true, y_scores, k_values
                )
                metrics['confidence_statistics'] = compute_confidence_statistics(y_scores, y_true)
            except Exception as e:
                logger.warning(f"Failed to compute top-k metrics: {e}")
                metrics['top_k_accuracies'] = {}
                metrics['confidence_statistics'] = {}
    else:
        metrics['top_k_accuracies'] = {}
        metrics['confidence_statistics'] = {}
    
    logger.info(f"Comprehensive metrics computed: overall_accuracy={metrics['overall_accuracy']:.4f}")
    
    return metrics


def validate_predictions(y_true: List[str], y_pred: List[str], 
                        y_scores: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Validate prediction inputs and return validation report.
    
    Args:
        y_true: List of true raag labels
        y_pred: List of predicted raag labels
        y_scores: Optional list of score dictionaries
        
    Returns:
        Dictionary containing validation results and warnings
    """
    validation_report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check basic requirements
    if len(y_true) != len(y_pred):
        validation_report['is_valid'] = False
        validation_report['errors'].append(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    if len(y_true) == 0:
        validation_report['is_valid'] = False
        validation_report['errors'].append("Empty input lists")
        return validation_report
    
    # Check for missing values
    if any(label is None or label == '' for label in y_true):
        validation_report['warnings'].append("Found None or empty strings in y_true")
    
    if any(label is None or label == '' for label in y_pred):
        validation_report['warnings'].append("Found None or empty strings in y_pred")
    
    # Check class consistency
    true_classes = set(y_true)
    pred_classes = set(y_pred)
    
    if pred_classes - true_classes:
        unknown_classes = pred_classes - true_classes
        validation_report['warnings'].append(
            f"Predictions contain unknown classes: {unknown_classes}"
        )
    
    # Validate scores if provided
    if y_scores is not None:
        if len(y_scores) != len(y_true):
            validation_report['warnings'].append(
                f"Scores length mismatch: {len(y_scores)} vs {len(y_true)}"
            )
        
        # Check score dictionaries
        empty_scores = sum(1 for scores in y_scores if not scores)
        if empty_scores > 0:
            validation_report['warnings'].append(
                f"Found {empty_scores} empty score dictionaries"
            )
    
    # Compute basic statistics
    validation_report['statistics'] = {
        'n_samples': len(y_true),
        'n_true_classes': len(true_classes),
        'n_pred_classes': len(pred_classes),
        'true_classes': sorted(true_classes),
        'pred_classes': sorted(pred_classes)
    }
    
    return validation_report

def compute_confusion_matrix(y_true: List[str], y_pred: List[str], 
                           class_order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Create confusion matrix generation with proper class ordering.
    
    Args:
        y_true: List of true raag labels
        y_pred: List of predicted raag labels
        class_order: Optional list specifying class order (default: sorted unique classes)
        
    Returns:
        Tuple of (confusion_matrix, class_names)
        - confusion_matrix: 2D numpy array where entry (i,j) is count of true class i predicted as class j
        - class_names: List of class names in matrix order
        
    Raises:
        ValueError: If input validation fails
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Empty input lists provided")
    
    # Determine class order
    if class_order is None:
        class_names = sorted(set(y_true) | set(y_pred))
    else:
        class_names = class_order.copy()
        # Add any missing classes from predictions
        missing_classes = (set(y_true) | set(y_pred)) - set(class_names)
        if missing_classes:
            logger.warning(f"Adding missing classes to confusion matrix: {missing_classes}")
            class_names.extend(sorted(missing_classes))
    
    n_classes = len(class_names)
    
    # Create class to index mapping
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        confusion_matrix[true_idx, pred_idx] += 1
    
    logger.debug(f"Confusion matrix computed: {n_classes}x{n_classes} classes")
    
    return confusion_matrix, class_names


def analyze_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """
    Add statistical analysis of classification patterns and errors.
    
    Args:
        confusion_matrix: 2D numpy array confusion matrix
        class_names: List of class names corresponding to matrix indices
        
    Returns:
        Dictionary containing detailed analysis of confusion matrix
        
    Raises:
        ValueError: If matrix and class names don't match
    """
    if confusion_matrix.shape[0] != len(class_names) or confusion_matrix.shape[1] != len(class_names):
        raise ValueError(
            f"Matrix shape {confusion_matrix.shape} doesn't match class names length {len(class_names)}"
        )
    
    n_classes = len(class_names)
    total_samples = np.sum(confusion_matrix)
    
    if total_samples == 0:
        return {'error': 'Empty confusion matrix'}
    
    analysis = {
        'matrix_shape': confusion_matrix.shape,
        'total_samples': int(total_samples),
        'n_classes': n_classes,
        'class_names': class_names
    }
    
    # Per-class statistics
    per_class_stats = {}
    
    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp  # Column sum minus diagonal
        fn = np.sum(confusion_matrix[i, :]) - tp  # Row sum minus diagonal
        tn = total_samples - tp - fp - fn
        
        # Support (actual occurrences)
        support = np.sum(confusion_matrix[i, :])
        
        # Precision, recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_stats[class_name] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'support': int(support),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    analysis['per_class_stats'] = per_class_stats
    
    # Overall statistics
    macro_precision = np.mean([stats['precision'] for stats in per_class_stats.values()])
    macro_recall = np.mean([stats['recall'] for stats in per_class_stats.values()])
    macro_f1 = np.mean([stats['f1_score'] for stats in per_class_stats.values()])
    
    # Weighted averages (by support)
    total_support = sum(stats['support'] for stats in per_class_stats.values())
    if total_support > 0:
        weighted_precision = sum(
            stats['precision'] * stats['support'] for stats in per_class_stats.values()
        ) / total_support
        weighted_recall = sum(
            stats['recall'] * stats['support'] for stats in per_class_stats.values()
        ) / total_support
        weighted_f1 = sum(
            stats['f1_score'] * stats['support'] for stats in per_class_stats.values()
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    analysis['overall_stats'] = {
        'accuracy': float(np.trace(confusion_matrix) / total_samples),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    # Error analysis
    analysis['error_analysis'] = analyze_classification_errors(confusion_matrix, class_names)
    
    logger.debug(f"Confusion matrix analysis completed for {n_classes} classes")
    
    return analysis


def analyze_classification_errors(confusion_matrix: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """
    Analyze classification patterns and common errors.
    
    Args:
        confusion_matrix: 2D numpy array confusion matrix
        class_names: List of class names corresponding to matrix indices
        
    Returns:
        Dictionary containing error pattern analysis
    """
    n_classes = len(class_names)
    total_samples = np.sum(confusion_matrix)
    
    error_analysis = {
        'most_confused_pairs': [],
        'most_accurate_classes': [],
        'least_accurate_classes': [],
        'class_confusion_rates': {}
    }
    
    # Find most confused class pairs (excluding diagonal)
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and confusion_matrix[i, j] > 0:
                confusion_rate = confusion_matrix[i, j] / np.sum(confusion_matrix[i, :])
                confusion_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(confusion_matrix[i, j]),
                    'confusion_rate': confusion_rate
                })
    
    # Sort by confusion rate and take top confusions
    confusion_pairs.sort(key=lambda x: x['confusion_rate'], reverse=True)
    error_analysis['most_confused_pairs'] = confusion_pairs[:10]  # Top 10
    
    # Class accuracy ranking
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        support = np.sum(confusion_matrix[i, :])
        if support > 0:
            accuracy = confusion_matrix[i, i] / support
            class_accuracies.append({
                'class': class_name,
                'accuracy': accuracy,
                'support': int(support)
            })
    
    class_accuracies.sort(key=lambda x: x['accuracy'], reverse=True)
    error_analysis['most_accurate_classes'] = class_accuracies[:3]  # Top 3
    error_analysis['least_accurate_classes'] = class_accuracies[-3:]  # Bottom 3
    
    # Per-class confusion rates
    for i, class_name in enumerate(class_names):
        support = np.sum(confusion_matrix[i, :])
        if support > 0:
            # Rate at which this class is confused with others
            confusion_rate = (support - confusion_matrix[i, i]) / support
            error_analysis['class_confusion_rates'][class_name] = confusion_rate
    
    return error_analysis


def export_confusion_matrix_json(confusion_matrix: np.ndarray, class_names: List[str], 
                                output_path: str, include_analysis: bool = True) -> None:
    """
    Export confusion matrix and analysis to JSON format.
    
    Args:
        confusion_matrix: 2D numpy array confusion matrix
        class_names: List of class names corresponding to matrix indices
        output_path: Path to save JSON file
        include_analysis: Whether to include detailed analysis (default: True)
        
    Raises:
        ValueError: If export fails
    """
    try:
        # Convert matrix to list for JSON serialization
        matrix_list = confusion_matrix.tolist()
        
        export_data = {
            'confusion_matrix': matrix_list,
            'class_names': class_names,
            'matrix_shape': confusion_matrix.shape,
            'total_samples': int(np.sum(confusion_matrix))
        }
        
        if include_analysis:
            analysis = analyze_confusion_matrix(confusion_matrix, class_names)
            export_data['analysis'] = analysis
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Confusion matrix exported to JSON: {output_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to export confusion matrix to JSON: {str(e)}")


def export_confusion_matrix_csv(confusion_matrix: np.ndarray, class_names: List[str], 
                               output_path: str) -> None:
    """
    Export confusion matrix to CSV format.
    
    Args:
        confusion_matrix: 2D numpy array confusion matrix
        class_names: List of class names corresponding to matrix indices
        output_path: Path to save CSV file
        
    Raises:
        ValueError: If export fails
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header row
            header = ['True\\Predicted'] + class_names
            writer.writerow(header)
            
            # Write matrix rows
            for i, class_name in enumerate(class_names):
                row = [class_name] + confusion_matrix[i, :].tolist()
                writer.writerow(row)
        
        logger.info(f"Confusion matrix exported to CSV: {output_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to export confusion matrix to CSV: {str(e)}")


def export_classification_report(y_true: List[str], y_pred: List[str], 
                               output_path: str, format: str = 'json',
                               class_order: Optional[List[str]] = None) -> None:
    """
    Export comprehensive classification report in structured format.
    
    Args:
        y_true: List of true raag labels
        y_pred: List of predicted raag labels
        output_path: Path to save report file
        format: Export format ('json' or 'csv', default: 'json')
        class_order: Optional list specifying class order
        
    Raises:
        ValueError: If export fails or format is unsupported
    """
    if format not in ['json', 'csv']:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")
    
    try:
        # Compute confusion matrix and analysis
        confusion_matrix, class_names = compute_confusion_matrix(y_true, y_pred, class_order)
        analysis = analyze_confusion_matrix(confusion_matrix, class_names)
        
        # Compute comprehensive metrics
        comprehensive_metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        # Create comprehensive report
        report = {
            'summary': {
                'total_samples': len(y_true),
                'n_classes': len(class_names),
                'overall_accuracy': comprehensive_metrics['overall_accuracy'],
                'balanced_accuracy': comprehensive_metrics['balanced_accuracy']
            },
            'confusion_matrix': {
                'matrix': confusion_matrix.tolist(),
                'class_names': class_names
            },
            'detailed_analysis': analysis,
            'comprehensive_metrics': comprehensive_metrics
        }
        
        # Export based on format
        output_path = Path(output_path)
        
        if format == 'json':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.json')
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.csv')
            
            # For CSV, export flattened metrics
            _export_flattened_report_csv(report, output_path)
        
        logger.info(f"Classification report exported ({format}): {output_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to export classification report: {str(e)}")


def _export_flattened_report_csv(report: Dict[str, Any], output_path: Path) -> None:
    """
    Export flattened classification report to CSV.
    
    Args:
        report: Classification report dictionary
        output_path: Path to save CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write summary section
        writer.writerow(['Section', 'Metric', 'Value'])
        writer.writerow(['Summary', 'Total Samples', report['summary']['total_samples']])
        writer.writerow(['Summary', 'Number of Classes', report['summary']['n_classes']])
        writer.writerow(['Summary', 'Overall Accuracy', report['summary']['overall_accuracy']])
        writer.writerow(['Summary', 'Balanced Accuracy', report['summary']['balanced_accuracy']])
        writer.writerow([])  # Empty row
        
        # Write per-class metrics
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        per_class_stats = report['detailed_analysis']['per_class_stats']
        for class_name, stats in per_class_stats.items():
            writer.writerow([
                class_name,
                stats['precision'],
                stats['recall'],
                stats['f1_score'],
                stats['support']
            ])
        
        writer.writerow([])  # Empty row
        
        # Write overall metrics
        overall_stats = report['detailed_analysis']['overall_stats']
        writer.writerow(['Overall Metric', 'Value'])
        for metric, value in overall_stats.items():
            writer.writerow([metric.replace('_', ' ').title(), value])


def compute_normalized_confusion_matrix(confusion_matrix: np.ndarray, 
                                      normalization: str = 'true') -> np.ndarray:
    """
    Compute normalized confusion matrix.
    
    Args:
        confusion_matrix: 2D numpy array confusion matrix
        normalization: Type of normalization ('true', 'pred', 'all')
            - 'true': Normalize by true class (rows sum to 1)
            - 'pred': Normalize by predicted class (columns sum to 1)  
            - 'all': Normalize by total samples (all entries sum to 1)
            
    Returns:
        Normalized confusion matrix
        
    Raises:
        ValueError: If normalization type is invalid
    """
    if normalization not in ['true', 'pred', 'all']:
        raise ValueError(f"Invalid normalization: {normalization}. Use 'true', 'pred', or 'all'")
    
    if confusion_matrix.size == 0:
        return confusion_matrix.copy()
    
    normalized_matrix = confusion_matrix.astype(float)
    
    if normalization == 'true':
        # Normalize by rows (true classes)
        row_sums = np.sum(normalized_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        normalized_matrix = normalized_matrix / row_sums
        
    elif normalization == 'pred':
        # Normalize by columns (predicted classes)
        col_sums = np.sum(normalized_matrix, axis=0, keepdims=True)
        # Avoid division by zero
        col_sums[col_sums == 0] = 1
        normalized_matrix = normalized_matrix / col_sums
        
    elif normalization == 'all':
        # Normalize by total
        total = np.sum(normalized_matrix)
        if total > 0:
            normalized_matrix = normalized_matrix / total
    
    return normalized_matrix


def compute_class_imbalance_metrics(y_true: List[str]) -> Dict[str, Any]:
    """
    Compute metrics related to class imbalance.
    
    Args:
        y_true: List of true raag labels
        
    Returns:
        Dictionary containing imbalance metrics
    """
    if not y_true:
        return {}
    
    class_counts = Counter(y_true)
    total_samples = len(y_true)
    n_classes = len(class_counts)
    
    # Class frequencies
    class_frequencies = {cls: count / total_samples for cls, count in class_counts.items()}
    
    # Imbalance ratio (max frequency / min frequency)
    max_freq = max(class_frequencies.values())
    min_freq = min(class_frequencies.values())
    imbalance_ratio = max_freq / min_freq if min_freq > 0 else float('inf')
    
    # Entropy (measure of balance)
    entropy = -sum(freq * np.log2(freq) for freq in class_frequencies.values() if freq > 0)
    max_entropy = np.log2(n_classes)  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Gini impurity
    gini = 1 - sum(freq ** 2 for freq in class_frequencies.values())
    
    return {
        'n_classes': n_classes,
        'total_samples': total_samples,
        'class_counts': dict(class_counts),
        'class_frequencies': class_frequencies,
        'imbalance_ratio': imbalance_ratio,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'gini_impurity': gini,
        'is_balanced': imbalance_ratio <= 2.0  # Heuristic threshold
    }