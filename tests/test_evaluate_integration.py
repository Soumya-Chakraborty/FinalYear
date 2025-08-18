"""
Integration tests for evaluation pipeline completeness.

Tests the complete evaluation system including metrics computation,
confusion matrix analysis, and integration with the classifier.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import json

from raag_hmm.evaluate.metrics import (
    compute_comprehensive_metrics,
    compute_confusion_matrix,
    analyze_confusion_matrix,
    export_classification_report
)


class TestEvaluationPipelineIntegration:
    """Test complete evaluation pipeline integration."""
    
    def test_end_to_end_evaluation_pipeline(self):
        """Test complete evaluation pipeline from predictions to analysis."""
        # Create realistic test scenario with 5 raag classes
        raag_classes = ['bihag', 'darbari', 'desh', 'gaud_malhar', 'yaman']
        
        # Generate test data with realistic patterns
        y_true = []
        y_pred = []
        y_scores = []
        
        # Simulate different accuracy levels per class
        class_accuracies = {
            'bihag': 0.9,      # High accuracy
            'darbari': 0.8,    # Good accuracy
            'desh': 0.7,       # Moderate accuracy
            'gaud_malhar': 0.6, # Lower accuracy
            'yaman': 0.5       # Challenging class
        }
        
        samples_per_class = 20
        
        for true_class in raag_classes:
            accuracy = class_accuracies[true_class]
            n_correct = int(samples_per_class * accuracy)
            n_incorrect = samples_per_class - n_correct
            
            # Add correct predictions
            y_true.extend([true_class] * n_correct)
            y_pred.extend([true_class] * n_correct)
            
            # Add incorrect predictions (distributed among other classes)
            if n_incorrect > 0:
                other_classes = [c for c in raag_classes if c != true_class]
                incorrect_preds = np.random.choice(other_classes, n_incorrect, replace=True)
                
                y_true.extend([true_class] * n_incorrect)
                y_pred.extend(incorrect_preds.tolist())
            
            # Generate realistic score distributions
            for i in range(samples_per_class):
                scores = {}
                if i < n_correct:
                    # Correct prediction - high score for true class
                    scores[true_class] = np.random.uniform(0.6, 0.9)
                    remaining_prob = 1.0 - scores[true_class]
                    
                    for other_class in [c for c in raag_classes if c != true_class]:
                        scores[other_class] = np.random.uniform(0, remaining_prob / 4)
                else:
                    # Incorrect prediction - distribute scores more evenly
                    for class_name in raag_classes:
                        scores[class_name] = np.random.uniform(0.1, 0.3)
                    
                    # Make one random class have higher score
                    pred_class = y_pred[len(y_scores)]
                    scores[pred_class] = np.random.uniform(0.4, 0.7)
                
                # Normalize scores to sum to 1
                total = sum(scores.values())
                scores = {k: v/total for k, v in scores.items()}
                
                y_scores.append(scores)
        
        # Test comprehensive metrics computation
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_scores)
        
        # Verify structure
        assert 'overall_accuracy' in metrics
        assert 'per_class_accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert 'top_k_accuracies' in metrics
        assert 'confidence_statistics' in metrics
        assert 'class_support' in metrics
        
        # Verify values are reasonable
        assert 0 <= metrics['overall_accuracy'] <= 1
        assert 0 <= metrics['balanced_accuracy'] <= 1
        assert metrics['total_samples'] == len(y_true)
        assert metrics['num_classes'] == len(raag_classes)
        
        # Test per-class accuracy
        for class_name in raag_classes:
            assert class_name in metrics['per_class_accuracy']
            assert 0 <= metrics['per_class_accuracy'][class_name] <= 1
        
        # Test top-k accuracies
        assert 1 in metrics['top_k_accuracies']
        assert 3 in metrics['top_k_accuracies']
        assert 5 in metrics['top_k_accuracies']
        
        # Top-k accuracy should be non-decreasing
        assert metrics['top_k_accuracies'][1] <= metrics['top_k_accuracies'][3]
        assert metrics['top_k_accuracies'][3] <= metrics['top_k_accuracies'][5]
        
        # Test confusion matrix
        confusion_matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        
        assert confusion_matrix.shape == (5, 5)
        assert len(class_names) == 5
        assert set(class_names) == set(raag_classes)
        assert np.sum(confusion_matrix) == len(y_true)
        
        # Test confusion matrix analysis
        analysis = analyze_confusion_matrix(confusion_matrix, class_names)
        
        assert 'per_class_stats' in analysis
        assert 'overall_stats' in analysis
        assert 'error_analysis' in analysis
        
        # Verify per-class statistics
        for class_name in class_names:
            stats = analysis['per_class_stats'][class_name]
            assert 'precision' in stats
            assert 'recall' in stats
            assert 'f1_score' in stats
            assert 'support' in stats
            
            # All metrics should be between 0 and 1
            assert 0 <= stats['precision'] <= 1
            assert 0 <= stats['recall'] <= 1
            assert 0 <= stats['f1_score'] <= 1
            assert stats['support'] >= 0
        
        # Test export functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / 'evaluation_report.json'
            
            export_classification_report(y_true, y_pred, report_path, format='json')
            
            assert report_path.exists()
            
            # Verify exported report
            with open(report_path) as f:
                report = json.load(f)
            
            assert 'summary' in report
            assert 'confusion_matrix' in report
            assert 'detailed_analysis' in report
            assert 'comprehensive_metrics' in report
            
            # Verify summary matches computed metrics
            assert report['summary']['total_samples'] == metrics['total_samples']
            assert report['summary']['n_classes'] == metrics['num_classes']
            assert abs(report['summary']['overall_accuracy'] - metrics['overall_accuracy']) < 1e-10
    
    def test_evaluation_with_perfect_classification(self):
        """Test evaluation pipeline with perfect classification results."""
        raag_classes = ['bihag', 'darbari', 'desh']
        samples_per_class = 10
        
        # Perfect classification
        y_true = []
        y_pred = []
        y_scores = []
        
        for class_name in raag_classes:
            y_true.extend([class_name] * samples_per_class)
            y_pred.extend([class_name] * samples_per_class)
            
            # Perfect scores
            for _ in range(samples_per_class):
                scores = {c: 0.01 for c in raag_classes}
                scores[class_name] = 0.98
                y_scores.append(scores)
        
        # Test metrics
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_scores)
        
        # Should have perfect accuracy
        assert metrics['overall_accuracy'] == 1.0
        assert metrics['balanced_accuracy'] == 1.0
        
        # All per-class accuracies should be 1.0
        for accuracy in metrics['per_class_accuracy'].values():
            assert accuracy == 1.0
        
        # All top-k accuracies should be 1.0
        for k_acc in metrics['top_k_accuracies'].values():
            assert k_acc == 1.0
        
        # Test confusion matrix
        confusion_matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        analysis = analyze_confusion_matrix(confusion_matrix, class_names)
        
        # Should be perfect diagonal matrix
        assert np.array_equal(confusion_matrix, np.diag([samples_per_class] * 3))
        
        # All precision, recall, F1 should be 1.0
        for stats in analysis['per_class_stats'].values():
            assert stats['precision'] == 1.0
            assert stats['recall'] == 1.0
            assert stats['f1_score'] == 1.0
        
        # No confusion pairs should exist
        assert len(analysis['error_analysis']['most_confused_pairs']) == 0
    
    def test_evaluation_with_worst_case_classification(self):
        """Test evaluation pipeline with worst-case classification results."""
        raag_classes = ['bihag', 'darbari', 'desh']
        samples_per_class = 5
        
        # Worst case: always predict wrong class
        y_true = []
        y_pred = []
        
        for i, true_class in enumerate(raag_classes):
            wrong_class = raag_classes[(i + 1) % len(raag_classes)]  # Next class in cycle
            
            y_true.extend([true_class] * samples_per_class)
            y_pred.extend([wrong_class] * samples_per_class)
        
        # Test metrics
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        # Should have zero accuracy
        assert metrics['overall_accuracy'] == 0.0
        
        # All per-class accuracies should be 0.0
        for accuracy in metrics['per_class_accuracy'].values():
            assert accuracy == 0.0
        
        # Test confusion matrix
        confusion_matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        analysis = analyze_confusion_matrix(confusion_matrix, class_names)
        
        # Diagonal should be all zeros
        assert np.sum(np.diag(confusion_matrix)) == 0
        
        # All precision and recall should be 0.0
        for stats in analysis['per_class_stats'].values():
            assert stats['precision'] == 0.0
            assert stats['recall'] == 0.0
            assert stats['f1_score'] == 0.0
    
    def test_evaluation_with_imbalanced_dataset(self):
        """Test evaluation pipeline with highly imbalanced dataset."""
        # Set random seed for reproducible test
        np.random.seed(42)
        
        # Highly imbalanced: 80% bihag, 15% darbari, 5% desh
        y_true = ['bihag'] * 80 + ['darbari'] * 15 + ['desh'] * 5
        
        # Create deterministic predictions for testing
        y_pred = []
        for i, true_label in enumerate(y_true):
            if true_label == 'bihag':
                # Majority class: mostly correct
                pred = 'bihag' if i % 10 != 0 else 'darbari'  # 90% correct
            elif true_label == 'darbari':
                # Minority class: some errors
                pred = 'darbari' if i % 5 != 0 else 'bihag'  # 80% correct
            else:  # desh
                # Smallest class: more errors
                pred = 'desh' if i % 3 == 0 else 'bihag'  # ~33% correct
            
            y_pred.append(pred)
        
        # Test metrics
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        # Overall accuracy should be high due to majority class
        assert metrics['overall_accuracy'] > 0.7
        
        # Balanced accuracy should be lower than overall accuracy
        assert metrics['balanced_accuracy'] < metrics['overall_accuracy']
        
        # Class support should reflect imbalance
        support = metrics['class_support']
        assert support['bihag'] == 80
        assert support['darbari'] == 15
        assert support['desh'] == 5
        
        # Test confusion matrix analysis
        confusion_matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        analysis = analyze_confusion_matrix(confusion_matrix, class_names)
        
        # Majority class should have reasonable precision
        bihag_stats = analysis['per_class_stats']['bihag']
        assert bihag_stats['precision'] > 0.6  # Relaxed threshold
        
        # Minority classes should have lower recall than majority
        desh_stats = analysis['per_class_stats']['desh']
        assert desh_stats['recall'] <= bihag_stats['recall']
    
    def test_evaluation_error_handling(self):
        """Test evaluation pipeline error handling."""
        # Test with empty inputs
        with pytest.raises(ValueError):
            compute_comprehensive_metrics([], [])
        
        # Test with mismatched lengths
        with pytest.raises(ValueError):
            compute_comprehensive_metrics(['bihag'], ['bihag', 'darbari'])
        
        # Test confusion matrix with invalid inputs
        with pytest.raises(ValueError):
            compute_confusion_matrix(['bihag'], ['bihag', 'darbari'])
        
        # Test analysis with mismatched matrix and class names
        matrix = np.array([[1, 0], [0, 1]])
        class_names = ['bihag', 'darbari', 'desh']  # 3 names for 2x2 matrix
        
        with pytest.raises(ValueError):
            analyze_confusion_matrix(matrix, class_names)
    
    def test_evaluation_with_missing_classes(self):
        """Test evaluation when some classes are missing from predictions."""
        # True labels include all classes, but predictions miss some
        y_true = ['bihag', 'darbari', 'desh', 'gaud_malhar', 'yaman']
        y_pred = ['bihag', 'bihag', 'bihag', 'bihag', 'bihag']  # Only predicts bihag
        
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        # Should handle missing classes gracefully
        assert metrics['num_classes'] == 5
        assert len(metrics['per_class_accuracy']) == 5
        
        # Only bihag should have non-zero accuracy
        assert metrics['per_class_accuracy']['bihag'] > 0
        for class_name in ['darbari', 'desh', 'gaud_malhar', 'yaman']:
            assert metrics['per_class_accuracy'][class_name] == 0.0
        
        # Test confusion matrix
        confusion_matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        
        # Matrix should be 5x5 even though only one class is predicted
        assert confusion_matrix.shape == (5, 5)
        
        # Only first column (bihag predictions) should have non-zero values
        assert np.sum(confusion_matrix[:, 0]) == 5  # All predictions are bihag
        assert np.sum(confusion_matrix[:, 1:]) == 0  # No other predictions


if __name__ == '__main__':
    pytest.main([__file__])