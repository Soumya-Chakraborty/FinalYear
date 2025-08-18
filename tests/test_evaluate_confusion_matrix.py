"""
Unit tests for confusion matrix and detailed analysis functions.

Tests confusion matrix generation, statistical analysis, error pattern detection,
and export functionality in structured formats.
"""

import pytest
import numpy as np
import json
import csv
import tempfile
from pathlib import Path
from typing import List, Dict

from raag_hmm.evaluate.metrics import (
    compute_confusion_matrix,
    analyze_confusion_matrix,
    analyze_classification_errors,
    export_confusion_matrix_json,
    export_confusion_matrix_csv,
    export_classification_report,
    compute_normalized_confusion_matrix,
    compute_class_imbalance_metrics
)


class TestComputeConfusionMatrix:
    """Test confusion matrix computation."""
    
    def test_perfect_classification(self):
        """Test confusion matrix with perfect classification."""
        y_true = ['bihag', 'darbari', 'desh', 'bihag', 'darbari']
        y_pred = ['bihag', 'darbari', 'desh', 'bihag', 'darbari']
        
        matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        
        expected_classes = ['bihag', 'darbari', 'desh']
        assert class_names == expected_classes
        
        # Should be diagonal matrix
        expected_matrix = np.array([
            [2, 0, 0],  # bihag
            [0, 2, 0],  # darbari
            [0, 0, 1]   # desh
        ])
        np.testing.assert_array_equal(matrix, expected_matrix)
    
    def test_with_errors(self):
        """Test confusion matrix with classification errors."""
        y_true = ['bihag', 'darbari', 'desh', 'bihag']
        y_pred = ['bihag', 'bihag', 'desh', 'darbari']  # One error: darbari->bihag, bihag->darbari
        
        matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        
        expected_classes = ['bihag', 'darbari', 'desh']
        assert class_names == expected_classes
        
        expected_matrix = np.array([
            [1, 1, 0],  # bihag: 1 correct, 1 misclassified as darbari
            [1, 0, 0],  # darbari: 1 misclassified as bihag, 0 correct
            [0, 0, 1]   # desh: 1 correct
        ])
        np.testing.assert_array_equal(matrix, expected_matrix)
    
    def test_custom_class_order(self):
        """Test confusion matrix with custom class ordering."""
        y_true = ['bihag', 'darbari', 'desh']
        y_pred = ['bihag', 'darbari', 'desh']
        
        custom_order = ['desh', 'bihag', 'darbari']
        matrix, class_names = compute_confusion_matrix(y_true, y_pred, custom_order)
        
        assert class_names == custom_order
        
        expected_matrix = np.array([
            [1, 0, 0],  # desh
            [0, 1, 0],  # bihag
            [0, 0, 1]   # darbari
        ])
        np.testing.assert_array_equal(matrix, expected_matrix)
    
    def test_unknown_classes_in_predictions(self):
        """Test handling of unknown classes in predictions."""
        y_true = ['bihag', 'darbari']
        y_pred = ['bihag', 'unknown_raag']
        
        matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        
        # Should include the unknown class
        assert 'unknown_raag' in class_names
        assert len(class_names) == 3  # bihag, darbari, unknown_raag
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        with pytest.raises(ValueError, match="Empty input lists"):
            compute_confusion_matrix([], [])
    
    def test_length_mismatch(self):
        """Test with mismatched input lengths."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_confusion_matrix(['bihag'], ['bihag', 'darbari'])


class TestAnalyzeConfusionMatrix:
    """Test confusion matrix analysis."""
    
    def test_perfect_classification_analysis(self):
        """Test analysis of perfect classification."""
        # Perfect diagonal matrix
        matrix = np.array([
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 1]
        ])
        class_names = ['bihag', 'darbari', 'desh']
        
        analysis = analyze_confusion_matrix(matrix, class_names)
        
        # Check structure
        assert 'per_class_stats' in analysis
        assert 'overall_stats' in analysis
        assert 'error_analysis' in analysis
        
        # Check overall accuracy
        assert analysis['overall_stats']['accuracy'] == 1.0
        
        # Check per-class metrics (should all be perfect)
        for class_name in class_names:
            stats = analysis['per_class_stats'][class_name]
            assert stats['precision'] == 1.0
            assert stats['recall'] == 1.0
            assert stats['f1_score'] == 1.0
    
    def test_analysis_with_errors(self):
        """Test analysis with classification errors."""
        # Matrix with some errors
        matrix = np.array([
            [2, 1, 0],  # bihag: 2 correct, 1 misclassified as darbari
            [0, 2, 1],  # darbari: 2 correct, 1 misclassified as desh
            [0, 0, 1]   # desh: 1 correct
        ])
        class_names = ['bihag', 'darbari', 'desh']
        
        analysis = analyze_confusion_matrix(matrix, class_names)
        
        # Check bihag stats
        bihag_stats = analysis['per_class_stats']['bihag']
        assert bihag_stats['true_positives'] == 2
        assert bihag_stats['false_positives'] == 0  # No other class predicted as bihag
        assert bihag_stats['false_negatives'] == 1  # 1 bihag predicted as darbari
        assert bihag_stats['support'] == 3
        assert bihag_stats['precision'] == 1.0  # 2/(2+0)
        assert abs(bihag_stats['recall'] - 2/3) < 1e-10  # 2/(2+1)
        
        # Check overall accuracy
        expected_accuracy = (2 + 2 + 1) / 7  # 5 correct out of 7 total
        assert abs(analysis['overall_stats']['accuracy'] - expected_accuracy) < 1e-10
    
    def test_empty_matrix(self):
        """Test analysis of empty matrix."""
        matrix = np.zeros((3, 3), dtype=int)
        class_names = ['bihag', 'darbari', 'desh']
        
        analysis = analyze_confusion_matrix(matrix, class_names)
        
        # Empty matrix should return error message
        assert 'error' in analysis
        assert analysis['error'] == 'Empty confusion matrix'
    
    def test_matrix_class_mismatch(self):
        """Test with mismatched matrix and class names."""
        matrix = np.array([[1, 0], [0, 1]])
        class_names = ['bihag', 'darbari', 'desh']  # 3 names for 2x2 matrix
        
        with pytest.raises(ValueError, match="Matrix shape.*doesn't match"):
            analyze_confusion_matrix(matrix, class_names)


class TestAnalyzeClassificationErrors:
    """Test classification error pattern analysis."""
    
    def test_error_pattern_analysis(self):
        """Test identification of error patterns."""
        # Matrix where bihag is often confused with darbari
        matrix = np.array([
            [1, 2, 0],  # bihag: 1 correct, 2 confused with darbari
            [0, 3, 0],  # darbari: 3 correct
            [1, 0, 2]   # desh: 2 correct, 1 confused with bihag
        ])
        class_names = ['bihag', 'darbari', 'desh']
        
        error_analysis = analyze_classification_errors(matrix, class_names)
        
        # Check most confused pairs
        confused_pairs = error_analysis['most_confused_pairs']
        assert len(confused_pairs) > 0
        
        # The highest confusion should be bihag->darbari (2/3 = 0.67)
        top_confusion = confused_pairs[0]
        assert top_confusion['true_class'] == 'bihag'
        assert top_confusion['predicted_class'] == 'darbari'
        assert top_confusion['count'] == 2
        assert abs(top_confusion['confusion_rate'] - 2/3) < 1e-10
        
        # Check class accuracy ranking
        most_accurate = error_analysis['most_accurate_classes']
        least_accurate = error_analysis['least_accurate_classes']
        
        assert len(most_accurate) > 0
        assert len(least_accurate) > 0
        
        # darbari should be most accurate (100%)
        assert most_accurate[0]['class'] == 'darbari'
        assert most_accurate[0]['accuracy'] == 1.0
    
    def test_no_errors(self):
        """Test error analysis with perfect classification."""
        matrix = np.array([
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 1]
        ])
        class_names = ['bihag', 'darbari', 'desh']
        
        error_analysis = analyze_classification_errors(matrix, class_names)
        
        # Should have no confused pairs
        assert len(error_analysis['most_confused_pairs']) == 0
        
        # All classes should have 0 confusion rate
        for class_name in class_names:
            assert error_analysis['class_confusion_rates'][class_name] == 0.0


class TestExportFunctions:
    """Test export functionality."""
    
    def test_export_confusion_matrix_json(self):
        """Test JSON export of confusion matrix."""
        matrix = np.array([[2, 1], [0, 3]])
        class_names = ['bihag', 'darbari']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            export_confusion_matrix_json(matrix, class_names, temp_path)
            
            # Verify file was created and contains expected data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'confusion_matrix' in data
            assert 'class_names' in data
            assert 'analysis' in data
            assert data['class_names'] == class_names
            assert data['confusion_matrix'] == matrix.tolist()
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_export_confusion_matrix_csv(self):
        """Test CSV export of confusion matrix."""
        matrix = np.array([[2, 1], [0, 3]])
        class_names = ['bihag', 'darbari']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            export_confusion_matrix_csv(matrix, class_names, temp_path)
            
            # Verify file was created and contains expected data
            with open(temp_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Check header
            assert rows[0] == ['True\\Predicted', 'bihag', 'darbari']
            
            # Check data rows
            assert rows[1] == ['bihag', '2', '1']
            assert rows[2] == ['darbari', '0', '3']
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_export_classification_report_json(self):
        """Test JSON export of classification report."""
        y_true = ['bihag', 'darbari', 'bihag', 'darbari']
        y_pred = ['bihag', 'bihag', 'bihag', 'darbari']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            export_classification_report(y_true, y_pred, temp_path, format='json')
            
            # Verify file was created and contains expected structure
            with open(temp_path, 'r') as f:
                report = json.load(f)
            
            assert 'summary' in report
            assert 'confusion_matrix' in report
            assert 'detailed_analysis' in report
            assert 'comprehensive_metrics' in report
            
            assert report['summary']['total_samples'] == 4
            assert report['summary']['n_classes'] == 2
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_export_classification_report_csv(self):
        """Test CSV export of classification report."""
        y_true = ['bihag', 'darbari']
        y_pred = ['bihag', 'darbari']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            export_classification_report(y_true, y_pred, temp_path, format='csv')
            
            # Verify file was created
            assert Path(temp_path).exists()
            
            # Check it contains CSV data
            with open(temp_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            assert len(rows) > 0
            assert 'Section' in rows[0] or 'Class' in rows[0]  # Should have headers
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_export_invalid_format(self):
        """Test export with invalid format."""
        y_true = ['bihag']
        y_pred = ['bihag']
        
        with pytest.raises(ValueError, match="Unsupported format"):
            export_classification_report(y_true, y_pred, 'test.txt', format='xml')


class TestComputeNormalizedConfusionMatrix:
    """Test normalized confusion matrix computation."""
    
    def test_normalize_by_true_class(self):
        """Test normalization by true class (rows)."""
        matrix = np.array([
            [2, 1, 0],
            [0, 3, 1],
            [1, 0, 2]
        ])
        
        normalized = compute_normalized_confusion_matrix(matrix, 'true')
        
        # Each row should sum to 1
        row_sums = np.sum(normalized, axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0])
        
        # Check specific values
        expected_first_row = [2/3, 1/3, 0.0]
        np.testing.assert_allclose(normalized[0, :], expected_first_row)
    
    def test_normalize_by_predicted_class(self):
        """Test normalization by predicted class (columns)."""
        matrix = np.array([
            [2, 1, 0],
            [0, 3, 1],
            [1, 0, 2]
        ])
        
        normalized = compute_normalized_confusion_matrix(matrix, 'pred')
        
        # Each column should sum to 1
        col_sums = np.sum(normalized, axis=0)
        np.testing.assert_allclose(col_sums, [1.0, 1.0, 1.0])
        
        # Check specific values
        expected_first_col = [2/3, 0.0, 1/3]
        np.testing.assert_allclose(normalized[:, 0], expected_first_col)
    
    def test_normalize_by_all(self):
        """Test normalization by total samples."""
        matrix = np.array([
            [2, 1],
            [0, 3]
        ])
        
        normalized = compute_normalized_confusion_matrix(matrix, 'all')
        
        # All entries should sum to 1
        total_sum = np.sum(normalized)
        assert abs(total_sum - 1.0) < 1e-10
        
        # Check specific values
        total = 2 + 1 + 0 + 3  # = 6
        expected = np.array([
            [2/6, 1/6],
            [0/6, 3/6]
        ])
        np.testing.assert_allclose(normalized, expected)
    
    def test_normalize_with_zeros(self):
        """Test normalization with zero rows/columns."""
        matrix = np.array([
            [0, 0, 0],
            [1, 2, 0],
            [0, 0, 0]
        ])
        
        # Should handle zero rows gracefully
        normalized = compute_normalized_confusion_matrix(matrix, 'true')
        
        # Non-zero row should be normalized
        np.testing.assert_allclose(normalized[1, :], [1/3, 2/3, 0.0])
        
        # Zero rows should remain zero
        np.testing.assert_allclose(normalized[0, :], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(normalized[2, :], [0.0, 0.0, 0.0])
    
    def test_invalid_normalization(self):
        """Test with invalid normalization type."""
        matrix = np.array([[1, 0], [0, 1]])
        
        with pytest.raises(ValueError, match="Invalid normalization"):
            compute_normalized_confusion_matrix(matrix, 'invalid')


class TestComputeClassImbalanceMetrics:
    """Test class imbalance metrics computation."""
    
    def test_balanced_classes(self):
        """Test with perfectly balanced classes."""
        y_true = ['bihag', 'darbari', 'desh'] * 3  # 3 of each
        
        metrics = compute_class_imbalance_metrics(y_true)
        
        assert metrics['n_classes'] == 3
        assert metrics['total_samples'] == 9
        assert metrics['imbalance_ratio'] == 1.0  # All classes equal
        assert metrics['is_balanced'] is True
        
        # All classes should have frequency 1/3
        for freq in metrics['class_frequencies'].values():
            assert abs(freq - 1/3) < 1e-10
    
    def test_imbalanced_classes(self):
        """Test with imbalanced classes."""
        y_true = ['bihag'] * 6 + ['darbari'] * 2 + ['desh'] * 1  # 6:2:1 ratio
        
        metrics = compute_class_imbalance_metrics(y_true)
        
        assert metrics['n_classes'] == 3
        assert metrics['total_samples'] == 9
        assert metrics['imbalance_ratio'] == 6.0  # 6/9 / 1/9 = 6
        assert metrics['is_balanced'] is False
        
        # Check frequencies
        expected_freqs = {'bihag': 6/9, 'darbari': 2/9, 'desh': 1/9}
        for class_name, expected_freq in expected_freqs.items():
            assert abs(metrics['class_frequencies'][class_name] - expected_freq) < 1e-10
    
    def test_single_class(self):
        """Test with single class."""
        y_true = ['bihag'] * 5
        
        metrics = compute_class_imbalance_metrics(y_true)
        
        assert metrics['n_classes'] == 1
        assert metrics['imbalance_ratio'] == 1.0  # Only one class
        assert metrics['class_frequencies']['bihag'] == 1.0
    
    def test_empty_input(self):
        """Test with empty input."""
        metrics = compute_class_imbalance_metrics([])
        assert metrics == {}
    
    def test_entropy_calculation(self):
        """Test entropy calculation for balance measure."""
        # Perfectly balanced 2 classes should have maximum entropy
        y_true = ['bihag'] * 5 + ['darbari'] * 5
        
        metrics = compute_class_imbalance_metrics(y_true)
        
        # Maximum entropy for 2 classes is log2(2) = 1
        expected_max_entropy = np.log2(2)
        assert abs(metrics['entropy'] - expected_max_entropy) < 1e-10
        assert abs(metrics['normalized_entropy'] - 1.0) < 1e-10


class TestIntegrationTests:
    """Integration tests for complete evaluation pipeline."""
    
    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation from predictions to export."""
        # Create realistic test data
        y_true = ['bihag'] * 10 + ['darbari'] * 8 + ['desh'] * 6 + ['gaud_malhar'] * 4 + ['yaman'] * 2
        y_pred = (
            ['bihag'] * 8 + ['darbari'] * 2 +  # 8/10 bihag correct
            ['darbari'] * 6 + ['bihag'] * 2 +  # 6/8 darbari correct
            ['desh'] * 5 + ['yaman'] * 1 +     # 5/6 desh correct
            ['gaud_malhar'] * 3 + ['desh'] * 1 +  # 3/4 gaud_malhar correct
            ['yaman'] * 1 + ['bihag'] * 1      # 1/2 yaman correct
        )
        
        # Compute confusion matrix
        matrix, class_names = compute_confusion_matrix(y_true, y_pred)
        
        # Analyze matrix
        analysis = analyze_confusion_matrix(matrix, class_names)
        
        # Check that analysis contains all expected components
        assert 'per_class_stats' in analysis
        assert 'overall_stats' in analysis
        assert 'error_analysis' in analysis
        
        # Verify some expected results
        assert len(class_names) == 5
        assert analysis['total_samples'] == 30
        
        # Test export functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / 'report.json'
            csv_path = Path(temp_dir) / 'report.csv'
            
            export_classification_report(y_true, y_pred, json_path, format='json')
            export_classification_report(y_true, y_pred, csv_path, format='csv')
            
            assert json_path.exists()
            assert csv_path.exists()
            
            # Verify JSON content
            with open(json_path) as f:
                report = json.load(f)
            
            assert report['summary']['total_samples'] == 30
            assert report['summary']['n_classes'] == 5


if __name__ == '__main__':
    pytest.main([__file__])