"""
Integration tests for forward algorithm scoring and classification.

Tests verify that the forward algorithm implementation in DiscreteHMM
works correctly with the RaagClassifier for end-to-end classification.
"""

import pytest
import numpy as np
import tempfile
import shutil
import joblib
import json
from pathlib import Path

from raag_hmm.hmm.model import DiscreteHMM
from raag_hmm.infer.classifier import RaagClassifier
from raag_hmm.exceptions import ClassificationError


class TestForwardAlgorithmIntegration:
    """Integration tests for forward algorithm scoring."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for test models."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def trained_models(self):
        """Create trained models with different characteristics."""
        models = {}
        
        # Create models with different random seeds to get different parameters
        raag_names = ["bihag", "darbari", "desh"]
        
        for i, raag_name in enumerate(raag_names):
            model = DiscreteHMM(n_states=36, n_observations=36, random_state=i * 42)
            
            # Train with some synthetic sequences to make models more distinct
            synthetic_sequences = []
            for seq_idx in range(5):
                # Create sequences with different patterns for each raag
                if raag_name == "bihag":
                    # Bihag-like pattern: emphasize certain notes
                    base_pattern = [0, 2, 4, 7, 9, 11, 12, 14, 16, 19, 21, 23, 24]
                elif raag_name == "darbari":
                    # Darbari-like pattern: different note emphasis
                    base_pattern = [0, 1, 4, 6, 7, 10, 12, 13, 16, 18, 19, 22, 24]
                else:  # desh
                    # Desh-like pattern: another pattern
                    base_pattern = [0, 2, 5, 7, 9, 12, 14, 17, 19, 21, 24, 26, 28]
                
                # Add some variation and repetition
                sequence = []
                for _ in range(20):  # 20 notes per sequence
                    note = np.random.choice(base_pattern)
                    sequence.append(note)
                
                synthetic_sequences.append(sequence)
            
            # Train the model
            training_stats = model.train(
                synthetic_sequences, 
                max_iterations=50, 
                convergence_tolerance=0.5,
                verbose=False
            )
            
            metadata = {
                'model_class': 'DiscreteHMM',
                'model_parameters': {
                    'n_states': model.n_states,
                    'n_observations': model.n_observations
                },
                'saved_at': '2024-01-01T12:00:00',
                'n_sequences': len(synthetic_sequences),
                'total_frames': sum(len(seq) for seq in synthetic_sequences),
                'converged': training_stats['converged'],
                'final_log_likelihood': training_stats['final_log_likelihood'],
                'training_time': 10.0 + i
            }
            
            models[raag_name] = (model, metadata)
        
        return models
    
    def create_model_files(self, models_dir, trained_models):
        """Create model files from trained models."""
        models_path = Path(models_dir)
        
        for raag_name, (model, metadata) in trained_models.items():
            # Save model
            model_path = models_path / f"{raag_name}.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata_path = models_path / f"{raag_name}_meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
    
    def test_forward_algorithm_scoring_consistency(self, temp_models_dir, trained_models):
        """Test that forward algorithm scoring is consistent."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Create test sequence
        test_sequence = np.array([0, 2, 4, 7, 9, 11, 12, 14, 16, 19, 21, 23, 24, 0])
        
        # Score the sequence multiple times - should be consistent
        scores1 = {}
        scores2 = {}
        
        for raag_name in ["bihag", "darbari", "desh"]:
            scores1[raag_name] = classifier.score_sequence(test_sequence, raag_name)
            scores2[raag_name] = classifier.score_sequence(test_sequence, raag_name)
        
        # Scores should be identical (deterministic)
        for raag_name in scores1:
            assert abs(scores1[raag_name] - scores2[raag_name]) < 1e-10
    
    def test_forward_algorithm_different_sequences_different_scores(self, temp_models_dir, trained_models):
        """Test that different sequences produce different scores."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Create two different sequences
        sequence1 = np.array([0, 2, 4, 7, 9, 11, 12])
        sequence2 = np.array([1, 3, 6, 8, 10, 13, 15])
        
        # Score both sequences
        scores1 = classifier.score_sequence(sequence1, "bihag")
        scores2 = classifier.score_sequence(sequence2, "bihag")
        
        # Scores should be different (very unlikely to be identical)
        assert abs(scores1 - scores2) > 1e-6
    
    def test_argmax_prediction_correctness(self, temp_models_dir, trained_models):
        """Test that argmax prediction returns the highest scoring model."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Create test sequence
        test_sequence = np.array([0, 2, 4, 7, 9, 11, 12, 14, 16, 19])
        
        # Get prediction with all scores
        predicted_raag, all_scores = classifier.predict_raag(test_sequence, return_all_scores=True)
        
        # Verify that predicted raag has the highest score
        max_score_raag = max(all_scores, key=all_scores.get)
        assert predicted_raag == max_score_raag
        
        # Verify that the prediction is one of the available raags
        assert predicted_raag in ["bihag", "darbari", "desh"]
    
    def test_confidence_scores_properties(self, temp_models_dir, trained_models):
        """Test properties of confidence scores."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        test_sequence = np.array([0, 2, 4, 7, 9, 11, 12, 14, 16, 19, 21, 23])
        
        # Get prediction with confidence
        result = classifier.predict_with_confidence(test_sequence, normalize_scores=True)
        
        # Check result structure
        assert 'predicted_raag' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert 'probabilities' in result
        assert 'ranking' in result
        
        # Check probabilities sum to 1
        prob_sum = sum(result['probabilities'].values())
        assert abs(prob_sum - 1.0) < 1e-10
        
        # Check that all probabilities are non-negative
        assert all(prob >= 0 for prob in result['probabilities'].values())
        
        # Check that confidence is the probability of the predicted class
        predicted_prob = result['probabilities'][result['predicted_raag']]
        assert abs(result['confidence'] - predicted_prob) < 1e-10
        
        # Check ranking order
        ranking = result['ranking']
        assert len(ranking) == 3
        
        # Verify ranking is sorted by score (highest first)
        for i in range(len(ranking) - 1):
            current_score = result['scores'][ranking[i]]
            next_score = result['scores'][ranking[i + 1]]
            assert current_score >= next_score
    
    def test_forward_algorithm_numerical_stability(self, temp_models_dir, trained_models):
        """Test numerical stability of forward algorithm with various sequence lengths."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Test with sequences of different lengths
        sequence_lengths = [1, 5, 10, 50, 100, 500]
        
        for length in sequence_lengths:
            # Create random sequence of given length
            test_sequence = np.random.randint(0, 36, size=length)
            
            # Score with all models
            for raag_name in ["bihag", "darbari", "desh"]:
                score = classifier.score_sequence(test_sequence, raag_name)
                
                # Check that score is finite and not NaN
                assert np.isfinite(score), f"Non-finite score for length {length}, raag {raag_name}"
                assert not np.isnan(score), f"NaN score for length {length}, raag {raag_name}"
                
                # Score should be negative (log-likelihood)
                assert score <= 0, f"Positive log-likelihood for length {length}, raag {raag_name}"
    
    def test_sequence_length_effect_on_scores(self, temp_models_dir, trained_models):
        """Test that longer sequences generally have lower (more negative) log-likelihoods."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Create sequences of increasing length with same pattern
        base_pattern = [0, 2, 4, 7, 9, 11, 12]
        
        short_sequence = np.array(base_pattern[:3])
        medium_sequence = np.array(base_pattern[:5])
        long_sequence = np.array(base_pattern)
        
        # Score all sequences with same model
        short_score = classifier.score_sequence(short_sequence, "bihag")
        medium_score = classifier.score_sequence(medium_sequence, "bihag")
        long_score = classifier.score_sequence(long_sequence, "bihag")
        
        # Longer sequences should generally have lower (more negative) log-likelihoods
        # Note: This is not always guaranteed due to the specific sequence content,
        # but it's generally true for most sequences
        assert short_score >= medium_score >= long_score or abs(short_score - medium_score) < 1.0
    
    def test_error_handling_in_scoring(self, temp_models_dir, trained_models):
        """Test error handling in forward algorithm scoring."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Test with invalid observation indices
        invalid_sequence = np.array([0, 5, 50, 15])  # 50 is out of range
        
        with pytest.raises(ClassificationError, match="invalid observation indices"):
            classifier.score_sequence(invalid_sequence, "bihag")
        
        # Test with negative indices
        negative_sequence = np.array([0, 5, -1, 15])
        
        with pytest.raises(ClassificationError, match="invalid observation indices"):
            classifier.score_sequence(negative_sequence, "bihag")
        
        # Test with empty sequence
        empty_sequence = np.array([])
        
        with pytest.raises(ClassificationError, match="Empty sequence"):
            classifier.score_sequence(empty_sequence, "bihag")
    
    def test_model_specific_scoring_differences(self, temp_models_dir, trained_models):
        """Test that different models produce different scores for the same sequence."""
        self.create_model_files(temp_models_dir, trained_models)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Create test sequence
        test_sequence = np.array([0, 2, 4, 7, 9, 11, 12, 14, 16, 19])
        
        # Score with all models
        scores = {}
        for raag_name in ["bihag", "darbari", "desh"]:
            scores[raag_name] = classifier.score_sequence(test_sequence, raag_name)
        
        # All scores should be different (models were trained on different patterns)
        score_values = list(scores.values())
        assert len(set(score_values)) == len(score_values), "All models should produce different scores"
        
        # All scores should be finite and negative
        for raag_name, score in scores.items():
            assert np.isfinite(score), f"Score for {raag_name} is not finite"
            assert score <= 0, f"Score for {raag_name} is positive: {score}"