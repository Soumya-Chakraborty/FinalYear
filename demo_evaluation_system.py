#!/usr/bin/env python3
"""
Demonstration of the comprehensive evaluation system for raag classification.

This script shows how to use the evaluation metrics, confusion matrix analysis,
and export functionality implemented in task 8.
"""

import numpy as np
from pathlib import Path

from raag_hmm.evaluate import (
    compute_comprehensive_metrics,
    compute_confusion_matrix,
    analyze_confusion_matrix,
    export_classification_report,
    compute_normalized_confusion_matrix,
    compute_class_imbalance_metrics
)


def main():
    """Demonstrate the evaluation system functionality."""
    print("=== RaagHMM Evaluation System Demo ===\n")
    
    # Create sample data representing realistic raag classification results
    raag_classes = ['bihag', 'darbari', 'desh', 'gaud_malhar', 'yaman']
    
    # Simulate test results with different accuracy levels per class
    np.random.seed(42)  # For reproducible results
    
    y_true = []
    y_pred = []
    y_scores = []
    
    # Generate realistic test data
    class_samples = [25, 20, 18, 15, 12]  # Slightly imbalanced
    class_accuracies = [0.88, 0.82, 0.75, 0.67, 0.58]  # Varying difficulty
    
    for i, (class_name, n_samples, accuracy) in enumerate(zip(raag_classes, class_samples, class_accuracies)):
        n_correct = int(n_samples * accuracy)
        n_incorrect = n_samples - n_correct
        
        # Add correct predictions
        y_true.extend([class_name] * n_correct)
        y_pred.extend([class_name] * n_correct)
        
        # Add incorrect predictions
        if n_incorrect > 0:
            other_classes = [c for c in raag_classes if c != class_name]
            incorrect_preds = np.random.choice(other_classes, n_incorrect, replace=True)
            y_true.extend([class_name] * n_incorrect)
            y_pred.extend(incorrect_preds.tolist())
        
        # Generate score distributions
        for j in range(n_samples):
            scores = {}
            if j < n_correct:
                # Correct prediction - high confidence
                scores[class_name] = np.random.uniform(0.6, 0.9)
                remaining = 1.0 - scores[class_name]
                for other_class in other_classes:
                    scores[other_class] = np.random.uniform(0, remaining / len(other_classes))
            else:
                # Incorrect prediction - lower confidence
                for cls in raag_classes:
                    scores[cls] = np.random.uniform(0.1, 0.3)
                pred_class = y_pred[len(y_scores)]
                scores[pred_class] = np.random.uniform(0.4, 0.6)
            
            # Normalize scores
            total = sum(scores.values())
            scores = {k: v/total for k, v in scores.items()}
            y_scores.append(scores)
    
    print(f"Dataset: {len(y_true)} samples, {len(raag_classes)} classes")
    print(f"Classes: {raag_classes}\n")
    
    # 1. Comprehensive Metrics
    print("1. COMPREHENSIVE METRICS")
    print("-" * 40)
    
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_scores)
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Number of Classes: {metrics['num_classes']}")
    
    print("\nPer-Class Accuracy:")
    for class_name, accuracy in metrics['per_class_accuracy'].items():
        support = metrics['class_support'][class_name]
        print(f"  {class_name:12}: {accuracy:.3f} (n={support})")
    
    print("\nTop-K Accuracies:")
    for k, accuracy in metrics['top_k_accuracies'].items():
        print(f"  Top-{k}: {accuracy:.3f}")
    
    print()
    
    # 2. Confusion Matrix Analysis
    print("2. CONFUSION MATRIX ANALYSIS")
    print("-" * 40)
    
    confusion_matrix, class_names = compute_confusion_matrix(y_true, y_pred)
    analysis = analyze_confusion_matrix(confusion_matrix, class_names)
    
    print("Confusion Matrix:")
    print("True\\Pred", end="")
    for class_name in class_names:
        print(f"{class_name:>8}", end="")
    print()
    
    for i, true_class in enumerate(class_names):
        print(f"{true_class:8}", end="")
        for j in range(len(class_names)):
            print(f"{confusion_matrix[i, j]:>8}", end="")
        print()
    
    print("\nPer-Class Statistics:")
    print(f"{'Class':12} {'Precision':>9} {'Recall':>9} {'F1-Score':>9} {'Support':>9}")
    print("-" * 60)
    
    for class_name in class_names:
        stats = analysis['per_class_stats'][class_name]
        print(f"{class_name:12} {stats['precision']:>9.3f} {stats['recall']:>9.3f} "
              f"{stats['f1_score']:>9.3f} {stats['support']:>9}")
    
    print("\nOverall Statistics:")
    overall = analysis['overall_stats']
    print(f"  Macro Precision: {overall['macro_precision']:.3f}")
    print(f"  Macro Recall:    {overall['macro_recall']:.3f}")
    print(f"  Macro F1-Score:  {overall['macro_f1']:.3f}")
    print(f"  Weighted F1:     {overall['weighted_f1']:.3f}")
    
    # 3. Error Analysis
    print("\n3. ERROR ANALYSIS")
    print("-" * 40)
    
    error_analysis = analysis['error_analysis']
    
    print("Most Confused Class Pairs:")
    for i, pair in enumerate(error_analysis['most_confused_pairs'][:5]):
        print(f"  {i+1}. {pair['true_class']} → {pair['predicted_class']}: "
              f"{pair['count']} errors ({pair['confusion_rate']:.1%})")
    
    print("\nClass Accuracy Ranking:")
    print("Most Accurate:")
    for i, cls_info in enumerate(error_analysis['most_accurate_classes'][:3]):
        print(f"  {i+1}. {cls_info['class']}: {cls_info['accuracy']:.1%} "
              f"(n={cls_info['support']})")
    
    print("Least Accurate:")
    for i, cls_info in enumerate(error_analysis['least_accurate_classes'][-3:]):
        print(f"  {i+1}. {cls_info['class']}: {cls_info['accuracy']:.1%} "
              f"(n={cls_info['support']})")
    
    # 4. Class Imbalance Analysis
    print("\n4. CLASS IMBALANCE ANALYSIS")
    print("-" * 40)
    
    imbalance_metrics = compute_class_imbalance_metrics(y_true)
    
    print(f"Imbalance Ratio: {imbalance_metrics['imbalance_ratio']:.2f}")
    print(f"Normalized Entropy: {imbalance_metrics['normalized_entropy']:.3f}")
    print(f"Gini Impurity: {imbalance_metrics['gini_impurity']:.3f}")
    print(f"Is Balanced: {imbalance_metrics['is_balanced']}")
    
    print("\nClass Frequencies:")
    for class_name, freq in imbalance_metrics['class_frequencies'].items():
        count = imbalance_metrics['class_counts'][class_name]
        print(f"  {class_name:12}: {freq:.1%} ({count} samples)")
    
    # 5. Normalized Confusion Matrix
    print("\n5. NORMALIZED CONFUSION MATRIX (by true class)")
    print("-" * 40)
    
    normalized_matrix = compute_normalized_confusion_matrix(confusion_matrix, 'true')
    
    print("True\\Pred", end="")
    for class_name in class_names:
        print(f"{class_name:>8}", end="")
    print()
    
    for i, true_class in enumerate(class_names):
        print(f"{true_class:8}", end="")
        for j in range(len(class_names)):
            print(f"{normalized_matrix[i, j]:>8.2f}", end="")
        print()
    
    # 6. Export Results
    print("\n6. EXPORTING RESULTS")
    print("-" * 40)
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Export comprehensive report
    json_path = output_dir / "classification_report.json"
    csv_path = output_dir / "classification_report.csv"
    
    export_classification_report(y_true, y_pred, json_path, format='json')
    export_classification_report(y_true, y_pred, csv_path, format='csv')
    
    print(f"✓ JSON report exported: {json_path}")
    print(f"✓ CSV report exported: {csv_path}")
    
    print(f"\nEvaluation complete! Check the '{output_dir}' directory for detailed reports.")


if __name__ == "__main__":
    main()