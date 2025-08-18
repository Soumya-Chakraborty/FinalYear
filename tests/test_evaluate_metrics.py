"""
Unit tests for raag_hmm.evaluate.metrics module.

Tests accuracy metrics computation including overall accuracy,
per-class accuracy, top-k accuracy, and validation functions.
"""

import pytest
import numpy as np
from typing import List, Dict

from raag_hmm.evaluate.metrics import (
    compute_accuracy,
    compute_per_class_accuracy,
    compute_top_k_accuracy,
    compute_multiple_top_k_accuracies,
    compute_balanced_accuracy,
    compute_class_support,
    compute_confidence_statistics,
    compute_comprehensive_metrics,
    validate_predictions
)


class TestComputeAccuracy:
    """Test overall accuracy computation."""
    
    def test_perfect_accuracy(self):
        """Test case with 100% accuracy."""
        y_true = ['bihag', 'darbari', 'desh', 'gaud_malhar', 'yaman']
        y_pred = ['bihag', 'darbari', 'desh', 'gaud_malhar', 'yaman']
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_zero_accuracy(self):
        """Test case with 0% accuracy."""
        y_true = ['bihag', 'bihag', 'bihag']
        y_pred = ['darbari', 'desh', 'yaman']
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_partial_accuracy(self):
        """Test case with partial accuracy."""
        y_true = ['bihag', 'darbari', 'desh', 'gaud_malhar']
        y_pred = ['bihag', 'bihag', 'desh', 'yaman']
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 0.5  # 2 out of 4 correct
    
    def test_single_sample(self):
        """Test with single sample."""
        y_true = ['bihag']
        y_pred = ['bihag']
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_empty_lists(self):
        """Test with empty lists should raise ValueError."""
        with pytest.raises(ValueError, match="Empty input lists"):
            compute_accuracy([], [])
    
    def test_length_mismatch(self):
        """Test with mismatched list lengths should raise ValueError."""
        y_true = ['bihag', 'darbari']
        y_pred = ['bihag']
        
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_accuracy(y_true, y_pred)


class TestComputePerClassAccuracy:
    """Test per-class accuracy computation."""
    
    def test_perfect_per_class_accuracy(self):
        """Test case with perfect accuracy for all classes."""
        y_true = ['bihag', 'darbari', 'desh', 'bihag', 'darbari']
        y_pred = ['bihag', 'darbari', 'desh', 'bihag', 'darbari']
        
        per_class_acc = compute_per_class_accuracy(y_true, y_pred)
        
        expected = {'bihag': 1.0, 'darbari': 1.0, 'desh': 1.0}
        assert per_class_acc == expected
    
    def test_mixed_per_class_accuracy(self):
        """Test case with mixed accuracy across classes."""
        y_true = ['bihag', 'bihag', 'darbari', 'darbari', 'desh']
        y_pred = ['bihag', 'darbari', 'darbari', 'desh', 'desh']
        
        per_class_acc = compute_per_class_accuracy(y_true, y_pred)
        
        expected = {
            'bihag': 0.5,    # 1 out of 2 correct
            'darbari': 0.5,  # 1 out of 2 correct
            'desh': 1.0      # 1 out of 1 correct
        }
        assert per_class_acc == expected
    
    def test_zero_accuracy_for_class(self):
        """Test case where one class has zero accuracy."""
        y_true = ['bihag', 'bihag', 'darbari']
        y_pred = ['darbari', 'desh', 'darbari']
        
        per_class_acc = compute_per_class_accuracy(y_true, y_pred)
        
        expected = {
            'bihag': 0.0,    # 0 out of 2 correct
            'darbari': 1.0   # 1 out of 1 correct
        }
        assert per_class_acc == expected
    
    def test_single_class(self):
        """Test with only one class."""
        y_true = ['bihag', 'bihag', 'bihag']
        y_pred = ['bihag', 'darbari', 'bihag']
        
        per_class_acc = compute_per_class_accuracy(y_true, y_pred)
        
        expected = {'bihag': 2/3}  # 2 out of 3 correct
        assert abs(per_class_acc['bihag'] - 2/3) < 1e-10


class TestComputeTopKAccuracy:
    """Test top-k accuracy computation."""
    
    def test_top_1_accuracy(self):
        """Test top-1 accuracy (should equal regular accuracy)."""
        y_true = ['bihag', 'darbari', 'desh']
        y_scores = [
            {'bihag': 0.8, 'darbari': 0.1, 'desh': 0.1},
            {'bihag': 0.2, 'darbari': 0.7, 'desh': 0.1},
            {'bihag': 0.1, 'darbari': 0.2, 'desh': 0.7}
        ]
        
        top_1_acc = compute_top_k_accuracy(y_true, y_scores, k=1)
        assert top_1_acc == 1.0
    
    def test_top_3_accuracy(self):
        """Test top-3 accuracy with all classes in top 3."""
        y_true = ['bihag', 'darbari']
        y_scores = [
            {'bihag': 0.2, 'darbari': 0.7, 'desh': 0.1},  # bihag not in top 1 but in top 3
            {'bihag': 0.1, 'darbari': 0.8, 'desh': 0.1}   # darbari in top 1
        ]
        
        top_3_acc = compute_top_k_accuracy(y_true, y_scores, k=3)
        assert top_3_acc == 1.0  # Both true labels are in top 3
    
    def test_top_k_partial_accuracy(self):
        """Test top-k with partial accuracy."""
        y_true = ['bihag', 'darbari', 'desh']
        y_scores = [
            {'bihag': 0.8, 'darbari': 0.1, 'desh': 0.1},  # bihag correct (top 1)
            {'bihag': 0.7, 'darbari': 0.2, 'desh': 0.1},  # darbari is 2nd, so in top 2
            {'bihag': 0.1, 'darbari': 0.2, 'desh': 0.7}   # desh correct (top 1)
        ]
        
        top_2_acc = compute_top_k_accuracy(y_true, y_scores, k=2)
        assert top_2_acc == 1.0  # All 3 are in top 2
    
    def test_empty_scores_dict(self):
        """Test handling of empty scores dictionary."""
        y_true = ['bihag', 'darbari']
        y_scores = [
            {'bihag': 0.8, 'darbari': 0.2},
            {}  # Empty scores dict
        ]
        
        top_1_acc = compute_top_k_accuracy(y_true, y_scores, k=1)
        assert top_1_acc == 0.5  # Only first prediction counted
    
    def test_invalid_k(self):
        """Test with invalid k value."""
        y_true = ['bihag']
        y_scores = [{'bihag': 0.8}]
        
        with pytest.raises(ValueError, match="k must be positive"):
            compute_top_k_accuracy(y_true, y_scores, k=0)


class TestComputeMultipleTopKAccuracies:
    """Test multiple top-k accuracy computation."""
    
    def test_multiple_k_values(self):
        """Test computing multiple k values efficiently."""
        y_true = ['bihag', 'darbari', 'desh', 'gaud_malhar']
        y_scores = [
            {'bihag': 0.8, 'darbari': 0.1, 'desh': 0.05, 'gaud_malhar': 0.05},  # bihag top 1
            {'bihag': 0.3, 'darbari': 0.6, 'desh': 0.05, 'gaud_malhar': 0.05},  # darbari top 1
            {'bihag': 0.4, 'darbari': 0.3, 'desh': 0.2, 'gaud_malhar': 0.1},    # desh top 3
            {'bihag': 0.1, 'darbari': 0.1, 'desh': 0.1, 'gaud_malhar': 0.7}     # gaud_malhar top 1
        ]
        
        k_values = [1, 2, 3]
        accuracies = compute_multiple_top_k_accuracies(y_true, y_scores, k_values)
        
        expected = {
            1: 0.75,  # 3 out of 4 correct at top 1
            2: 0.75,  # Same as top 1 in this case
            3: 1.0    # All correct in top 3
        }
        
        for k in k_values:
            assert abs(accuracies[k] - expected[k]) < 1e-10
    
    def test_single_k_value(self):
        """Test with single k value."""
        y_true = ['bihag', 'darbari']
        y_scores = [
            {'bihag': 0.8, 'darbari': 0.2},
            {'bihag': 0.3, 'darbari': 0.7}
        ]
        
        accuracies = compute_multiple_top_k_accuracies(y_true, y_scores, [1])
        assert accuracies == {1: 1.0}


class TestComputeBalancedAccuracy:
    """Test balanced accuracy computation."""
    
    def test_balanced_accuracy_equal_classes(self):
        """Test balanced accuracy with equal class performance."""
        y_true = ['bihag', 'bihag', 'darbari', 'darbari']
        y_pred = ['bihag', 'darbari', 'darbari', 'bihag']
        
        balanced_acc = compute_balanced_accuracy(y_true, y_pred)
        assert balanced_acc == 0.5  # Both classes have 50% accuracy
    
    def test_balanced_accuracy_imbalanced_classes(self):
        """Test balanced accuracy with imbalanced classes."""
        y_true = ['bihag', 'bihag', 'bihag', 'darbari']  # 3:1 ratio
        y_pred = ['bihag', 'bihag', 'bihag', 'darbari']  # All correct
        
        balanced_acc = compute_balanced_accuracy(y_true, y_pred)
        assert balanced_acc == 1.0  # Both classes have 100% accuracy
    
    def test_balanced_vs_regular_accuracy(self):
        """Test difference between balanced and regular accuracy."""
        # Imbalanced dataset where majority class performs well
        y_true = ['bihag'] * 9 + ['darbari']  # 9:1 ratio
        y_pred = ['bihag'] * 9 + ['bihag']    # Majority class all correct, minority wrong
        
        regular_acc = compute_accuracy(y_true, y_pred)
        balanced_acc = compute_balanced_accuracy(y_true, y_pred)
        
        assert regular_acc == 0.9  # 9 out of 10 correct
        assert balanced_acc == 0.5  # (100% + 0%) / 2 = 50%


class TestComputeClassSupport:
    """Test class support computation."""
    
    def test_class_support_equal(self):
        """Test with equal class support."""
        y_true = ['bihag', 'darbari', 'bihag', 'darbari']
        
        support = compute_class_support(y_true)
        expected = {'bihag': 2, 'darbari': 2}
        assert support == expected
    
    def test_class_support_imbalanced(self):
        """Test with imbalanced class support."""
        y_true = ['bihag'] * 5 + ['darbari'] * 2 + ['desh']
        
        support = compute_class_support(y_true)
        expected = {'bihag': 5, 'darbari': 2, 'desh': 1}
        assert support == expected
    
    def test_empty_list(self):
        """Test with empty list."""
        support = compute_class_support([])
        assert support == {}


class TestComputeConfidenceStatistics:
    """Test confidence statistics computation."""
    
    def test_confidence_statistics_basic(self):
        """Test basic confidence statistics computation."""
        y_true = ['bihag', 'darbari']
        y_scores = [
            {'bihag': 0.8, 'darbari': 0.2},  # Correct prediction, high confidence
            {'bihag': 0.3, 'darbari': 0.7}   # Correct prediction, high confidence
        ]
        
        stats = compute_confidence_statistics(y_scores, y_true)
        
        # Check structure
        assert 'max_score_stats' in stats
        assert 'true_label_score_stats' in stats
        assert 'score_margin_stats' in stats
        assert 'correct_predictions' in stats
        
        # Check values
        assert stats['max_score_stats']['mean'] == 0.75  # (0.8 + 0.7) / 2
        assert stats['correct_predictions']['count'] == 2
    
    def test_confidence_statistics_with_errors(self):
        """Test confidence statistics with incorrect predictions."""
        y_true = ['bihag', 'darbari']
        y_scores = [
            {'bihag': 0.3, 'darbari': 0.7},  # Incorrect: predicted darbari
            {'bihag': 0.2, 'darbari': 0.8}   # Correct: predicted darbari
        ]
        
        stats = compute_confidence_statistics(y_scores, y_true)
        
        assert stats['correct_predictions']['count'] == 1
        assert stats['incorrect_predictions']['count'] == 1
    
    def test_empty_scores(self):
        """Test with empty scores."""
        stats = compute_confidence_statistics([], [])
        assert stats == {}


class TestComputeComprehensiveMetrics:
    """Test comprehensive metrics computation."""
    
    def test_comprehensive_metrics_without_scores(self):
        """Test comprehensive metrics without score information."""
        y_true = ['bihag', 'darbari', 'desh', 'bihag']
        y_pred = ['bihag', 'bihag', 'desh', 'bihag']
        
        metrics = compute_comprehensive_metrics(y_true, y_pred)
        
        # Check required fields
        required_fields = [
            'overall_accuracy', 'per_class_accuracy', 'balanced_accuracy',
            'class_support', 'total_samples', 'num_classes'
        ]
        for field in required_fields:
            assert field in metrics
        
        assert metrics['overall_accuracy'] == 0.75  # 3 out of 4 correct
        assert metrics['total_samples'] == 4
        assert metrics['num_classes'] == 3
    
    def test_comprehensive_metrics_with_scores(self):
        """Test comprehensive metrics with score information."""
        y_true = ['bihag', 'darbari']
        y_pred = ['bihag', 'darbari']
        y_scores = [
            {'bihag': 0.8, 'darbari': 0.2},
            {'bihag': 0.3, 'darbari': 0.7}
        ]
        
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_scores)
        
        assert 'top_k_accuracies' in metrics
        assert 'confidence_statistics' in metrics
        assert metrics['top_k_accuracies'][1] == 1.0  # Perfect top-1 accuracy


class TestValidatePredictions:
    """Test prediction validation."""
    
    def test_valid_predictions(self):
        """Test validation of valid predictions."""
        y_true = ['bihag', 'darbari', 'desh']
        y_pred = ['bihag', 'bihag', 'desh']
        
        report = validate_predictions(y_true, y_pred)
        
        assert report['is_valid'] is True
        assert len(report['errors']) == 0
        assert report['statistics']['n_samples'] == 3
    
    def test_length_mismatch_validation(self):
        """Test validation with length mismatch."""
        y_true = ['bihag', 'darbari']
        y_pred = ['bihag']
        
        report = validate_predictions(y_true, y_pred)
        
        assert report['is_valid'] is False
        assert any('Length mismatch' in error for error in report['errors'])
    
    def test_unknown_classes_warning(self):
        """Test warning for unknown classes in predictions."""
        y_true = ['bihag', 'darbari']
        y_pred = ['bihag', 'unknown_raag']
        
        report = validate_predictions(y_true, y_pred)
        
        assert report['is_valid'] is True  # Warning, not error
        assert any('unknown classes' in warning for warning in report['warnings'])
    
    def test_empty_scores_warning(self):
        """Test warning for empty score dictionaries."""
        y_true = ['bihag', 'darbari']
        y_pred = ['bihag', 'darbari']
        y_scores = [{'bihag': 0.8}, {}]  # Second is empty
        
        report = validate_predictions(y_true, y_pred, y_scores)
        
        assert any('empty score dictionaries' in warning for warning in report['warnings'])


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_nan_scores(self):
        """Test handling of NaN scores."""
        y_true = ['bihag']
        y_scores = [{'bihag': float('nan'), 'darbari': 0.5}]
        
        # Should not crash, but may produce unexpected results
        top_k_acc = compute_top_k_accuracy(y_true, y_scores, k=1)
        assert isinstance(top_k_acc, float)
    
    def test_infinite_scores(self):
        """Test handling of infinite scores."""
        y_true = ['bihag']
        y_scores = [{'bihag': float('inf'), 'darbari': 0.5}]
        
        top_k_acc = compute_top_k_accuracy(y_true, y_scores, k=1)
        assert top_k_acc == 1.0  # inf should be highest score
    
    def test_very_large_dataset(self):
        """Test with large dataset for performance."""
        n_samples = 10000
        y_true = ['bihag'] * (n_samples // 2) + ['darbari'] * (n_samples // 2)
        y_pred = y_true.copy()  # Perfect predictions
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 1.0
        
        per_class_acc = compute_per_class_accuracy(y_true, y_pred)
        assert all(acc == 1.0 for acc in per_class_acc.values())


if __name__ == '__main__':
    pytest.main([__file__])