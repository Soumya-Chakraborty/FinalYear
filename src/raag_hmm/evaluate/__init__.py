"""
Evaluation module.

Comprehensive performance analysis and metrics computation.
"""

from .metrics import (
    compute_accuracy,
    compute_per_class_accuracy,
    compute_top_k_accuracy,
    compute_multiple_top_k_accuracies,
    compute_balanced_accuracy,
    compute_class_support,
    compute_confidence_statistics,
    compute_comprehensive_metrics,
    validate_predictions,
    compute_confusion_matrix,
    analyze_confusion_matrix,
    analyze_classification_errors,
    export_confusion_matrix_json,
    export_confusion_matrix_csv,
    export_classification_report,
    compute_normalized_confusion_matrix,
    compute_class_imbalance_metrics
)

__all__ = [
    "compute_accuracy",
    "compute_per_class_accuracy", 
    "compute_top_k_accuracy",
    "compute_multiple_top_k_accuracies",
    "compute_balanced_accuracy",
    "compute_class_support",
    "compute_confidence_statistics",
    "compute_comprehensive_metrics",
    "validate_predictions",
    "compute_confusion_matrix",
    "analyze_confusion_matrix",
    "analyze_classification_errors",
    "export_confusion_matrix_json",
    "export_confusion_matrix_csv",
    "export_classification_report",
    "compute_normalized_confusion_matrix",
    "compute_class_imbalance_metrics"
]