"""
Unit tests for raag classification and model loading functionality.

Tests cover model loading edge cases, validation, caching, and classification accuracy.
"""

import pytest
import numpy as np
import json
import joblib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from raag_hmm.infer.classifier import ModelLoader, RaagClassifier
from raag_hmm.hmm.model import DiscreteHMM
from raag_hmm.exceptions import ClassificationError


class TestModelLoader:
    """Test cases for ModelLoader class."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for test models."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample DiscreteHMM model for testing."""
        model = DiscreteHMM(n_states=36, n_observations=36, random_state=42)
        return model
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            'model_class': 'DiscreteHMM',
            'model_parameters': {
                'n_states': 36,
                'n_observations': 36
            },
            'saved_at': '2024-01-01T12:00:00',
            'n_sequences': 10,
            'total_frames': 1000,
            'converged': True,
            'final_log_likelihood': -500.0,
            'training_time': 30.5
        }
    
    def create_test_model_files(self, models_dir, raag_name, model, metadata):
        """Helper to create model and metadata files."""
        models_path = Path(models_dir)
        
        # Save model
        model_path = models_path / f"{raag_name}.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = models_path / f"{raag_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return str(model_path), str(metadata_path)
    
    def test_init(self, temp_models_dir):
        """Test ModelLoader initialization."""
        loader = ModelLoader(temp_models_dir)
        assert loader.models_dir == Path(temp_models_dir)
        assert loader._model_cache == {}
        assert loader._metadata_cache == {}
    
    def test_load_model_success(self, temp_models_dir, sample_model, sample_metadata):
        """Test successful model loading."""
        raag_name = "test_raag"
        self.create_test_model_files(temp_models_dir, raag_name, sample_model, sample_metadata)
        
        loader = ModelLoader(temp_models_dir)
        model, metadata = loader.load_model(raag_name)
        
        assert isinstance(model, DiscreteHMM)
        assert model.n_states == 36
        assert model.n_observations == 36
        assert metadata['model_class'] == 'DiscreteHMM'
        assert metadata['n_sequences'] == 10
    
    def test_load_model_with_cache(self, temp_models_dir, sample_model, sample_metadata):
        """Test model loading with caching."""
        raag_name = "test_raag"
        self.create_test_model_files(temp_models_dir, raag_name, sample_model, sample_metadata)
        
        loader = ModelLoader(temp_models_dir)
        
        # First load
        model1, metadata1 = loader.load_model(raag_name, use_cache=True)
        
        # Second load should use cache
        model2, metadata2 = loader.load_model(raag_name, use_cache=True)
        
        # Should be the same objects from cache
        assert model1 is model2
        assert metadata1 is metadata2
        assert raag_name in loader._model_cache
        assert raag_name in loader._metadata_cache
    
    def test_load_model_without_cache(self, temp_models_dir, sample_model, sample_metadata):
        """Test model loading without caching."""
        raag_name = "test_raag"
        self.create_test_model_files(temp_models_dir, raag_name, sample_model, sample_metadata)
        
        loader = ModelLoader(temp_models_dir)
        
        # Load without cache
        model1, metadata1 = loader.load_model(raag_name, use_cache=False)
        model2, metadata2 = loader.load_model(raag_name, use_cache=False)
        
        # Should be different objects (not cached)
        assert model1 is not model2
        assert metadata1 is not metadata2
        assert raag_name not in loader._model_cache
    
    def test_load_model_file_not_found(self, temp_models_dir):
        """Test error handling when model file doesn't exist."""
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Model file not found"):
            loader.load_model("nonexistent_raag")
    
    def test_load_model_metadata_not_found(self, temp_models_dir, sample_model):
        """Test error handling when metadata file doesn't exist."""
        raag_name = "test_raag"
        models_path = Path(temp_models_dir)
        
        # Create only model file, no metadata
        model_path = models_path / f"{raag_name}.pkl"
        joblib.dump(sample_model, model_path)
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Metadata file not found"):
            loader.load_model(raag_name)
    
    def test_load_model_corrupted_metadata(self, temp_models_dir, sample_model):
        """Test error handling for corrupted metadata file."""
        raag_name = "test_raag"
        models_path = Path(temp_models_dir)
        
        # Create model file
        model_path = models_path / f"{raag_name}.pkl"
        joblib.dump(sample_model, model_path)
        
        # Create corrupted metadata file
        metadata_path = models_path / f"{raag_name}_meta.json"
        with open(metadata_path, 'w') as f:
            f.write("invalid json content {")
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Corrupted metadata file"):
            loader.load_model(raag_name)
    
    def test_load_model_missing_metadata_fields(self, temp_models_dir, sample_model):
        """Test error handling for missing required metadata fields."""
        raag_name = "test_raag"
        models_path = Path(temp_models_dir)
        
        # Create model file
        model_path = models_path / f"{raag_name}.pkl"
        joblib.dump(sample_model, model_path)
        
        # Create metadata with missing required fields
        incomplete_metadata = {'some_field': 'some_value'}
        metadata_path = models_path / f"{raag_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(incomplete_metadata, f)
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Missing required metadata field"):
            loader.load_model(raag_name)
    
    def test_load_model_wrong_model_class(self, temp_models_dir, sample_model):
        """Test error handling for wrong model class in metadata."""
        raag_name = "test_raag"
        models_path = Path(temp_models_dir)
        
        # Create model file
        model_path = models_path / f"{raag_name}.pkl"
        joblib.dump(sample_model, model_path)
        
        # Create metadata with wrong model class
        wrong_metadata = {
            'model_class': 'WrongModelClass',
            'model_parameters': {'n_states': 36, 'n_observations': 36},
            'saved_at': '2024-01-01T12:00:00'
        }
        metadata_path = models_path / f"{raag_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(wrong_metadata, f)
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Unsupported model class"):
            loader.load_model(raag_name)
    
    def test_load_model_corrupted_model_file(self, temp_models_dir, sample_metadata):
        """Test error handling for corrupted model file."""
        raag_name = "test_raag"
        models_path = Path(temp_models_dir)
        
        # Create corrupted model file
        model_path = models_path / f"{raag_name}.pkl"
        with open(model_path, 'wb') as f:
            f.write(b"corrupted pickle data")
        
        # Create valid metadata
        metadata_path = models_path / f"{raag_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Corrupted model file"):
            loader.load_model(raag_name)
    
    def test_load_model_wrong_object_type(self, temp_models_dir, sample_metadata):
        """Test error handling when loaded object is not DiscreteHMM."""
        raag_name = "test_raag"
        models_path = Path(temp_models_dir)
        
        # Save wrong object type
        model_path = models_path / f"{raag_name}.pkl"
        joblib.dump("not_a_model", model_path)
        
        # Create valid metadata
        metadata_path = models_path / f"{raag_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Loaded object is not a DiscreteHMM"):
            loader.load_model(raag_name)
    
    def test_load_model_dimension_mismatch(self, temp_models_dir):
        """Test error handling for model-metadata dimension mismatch."""
        raag_name = "test_raag"
        models_path = Path(temp_models_dir)
        
        # Create model with different dimensions
        model = DiscreteHMM(n_states=20, n_observations=20)
        model_path = models_path / f"{raag_name}.pkl"
        joblib.dump(model, model_path)
        
        # Create metadata with different dimensions
        metadata = {
            'model_class': 'DiscreteHMM',
            'model_parameters': {'n_states': 36, 'n_observations': 36},
            'saved_at': '2024-01-01T12:00:00'
        }
        metadata_path = models_path / f"{raag_name}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="n_states mismatch"):
            loader.load_model(raag_name)
    
    def test_load_all_models_success(self, temp_models_dir, sample_model, sample_metadata):
        """Test successful loading of all models."""
        raag_names = ["bihag", "darbari", "desh"]
        
        # Create multiple model files
        for raag_name in raag_names:
            self.create_test_model_files(temp_models_dir, raag_name, sample_model, sample_metadata)
        
        loader = ModelLoader(temp_models_dir)
        models = loader.load_all_models()
        
        assert len(models) == 3
        for raag_name in raag_names:
            assert raag_name in models
            model, metadata = models[raag_name]
            assert isinstance(model, DiscreteHMM)
            assert metadata['model_class'] == 'DiscreteHMM'
    
    def test_load_all_models_no_files(self, temp_models_dir):
        """Test error handling when no model files exist."""
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="No model files found"):
            loader.load_all_models()
    
    def test_load_all_models_partial_failure(self, temp_models_dir, sample_model, sample_metadata):
        """Test loading when some models fail to load."""
        # Create one valid model
        self.create_test_model_files(temp_models_dir, "valid_raag", sample_model, sample_metadata)
        
        # Create one invalid model (corrupted)
        models_path = Path(temp_models_dir)
        invalid_model_path = models_path / "invalid_raag.pkl"
        with open(invalid_model_path, 'wb') as f:
            f.write(b"corrupted")
        
        loader = ModelLoader(temp_models_dir)
        models = loader.load_all_models()
        
        # Should load only the valid model
        assert len(models) == 1
        assert "valid_raag" in models
        assert "invalid_raag" not in models
    
    def test_validate_model_compatibility_success(self, temp_models_dir, sample_metadata):
        """Test successful model compatibility validation."""
        # Create multiple compatible models
        models = {}
        for i, raag_name in enumerate(["raag1", "raag2", "raag3"]):
            model = DiscreteHMM(n_states=36, n_observations=36, random_state=i)
            models[raag_name] = (model, sample_metadata)
        
        loader = ModelLoader(temp_models_dir)
        result = loader.validate_model_compatibility(models)
        
        assert result is True
    
    def test_validate_model_compatibility_dimension_mismatch(self, temp_models_dir, sample_metadata):
        """Test model compatibility validation with dimension mismatch."""
        models = {
            "raag1": (DiscreteHMM(n_states=36, n_observations=36), sample_metadata),
            "raag2": (DiscreteHMM(n_states=20, n_observations=36), sample_metadata)  # Different n_states
        }
        
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="Model dimension mismatch"):
            loader.validate_model_compatibility(models)
    
    def test_validate_model_compatibility_empty_models(self, temp_models_dir):
        """Test model compatibility validation with empty models dict."""
        loader = ModelLoader(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="No models provided"):
            loader.validate_model_compatibility({})
    
    def test_clear_cache(self, temp_models_dir, sample_model, sample_metadata):
        """Test cache clearing functionality."""
        raag_name = "test_raag"
        self.create_test_model_files(temp_models_dir, raag_name, sample_model, sample_metadata)
        
        loader = ModelLoader(temp_models_dir)
        
        # Load model to populate cache
        loader.load_model(raag_name, use_cache=True)
        assert raag_name in loader._model_cache
        
        # Clear cache
        loader.clear_cache()
        assert len(loader._model_cache) == 0
        assert len(loader._metadata_cache) == 0
    
    def test_get_cache_info(self, temp_models_dir, sample_model, sample_metadata):
        """Test cache information retrieval."""
        raag_name = "test_raag"
        self.create_test_model_files(temp_models_dir, raag_name, sample_model, sample_metadata)
        
        loader = ModelLoader(temp_models_dir)
        
        # Initially empty cache
        cache_info = loader.get_cache_info()
        assert cache_info['cache_size'] == 0
        assert cache_info['cached_models'] == []
        
        # Load model to populate cache
        loader.load_model(raag_name, use_cache=True)
        
        cache_info = loader.get_cache_info()
        assert cache_info['cache_size'] == 1
        assert raag_name in cache_info['cached_models']
        assert cache_info['memory_usage_mb'] > 0


class TestRaagClassifier:
    """Test cases for RaagClassifier class."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for test models."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_models_and_metadata(self):
        """Create sample models and metadata for testing."""
        models_data = {}
        raag_names = ["bihag", "darbari", "desh"]
        
        for i, raag_name in enumerate(raag_names):
            model = DiscreteHMM(n_states=36, n_observations=36, random_state=i)
            metadata = {
                'model_class': 'DiscreteHMM',
                'model_parameters': {'n_states': 36, 'n_observations': 36},
                'saved_at': '2024-01-01T12:00:00',
                'n_sequences': 10 + i,
                'converged': True,
                'final_log_likelihood': -500.0 - i * 10
            }
            models_data[raag_name] = (model, metadata)
        
        return models_data
    
    def create_test_models(self, models_dir, models_data):
        """Helper to create test model files."""
        models_path = Path(models_dir)
        
        for raag_name, (model, metadata) in models_data.items():
            # Save model
            model_path = models_path / f"{raag_name}.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata_path = models_path / f"{raag_name}_meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
    
    def test_init(self, temp_models_dir):
        """Test RaagClassifier initialization."""
        classifier = RaagClassifier(temp_models_dir, use_cache=True)
        
        assert classifier.models_dir == temp_models_dir
        assert classifier.use_cache is True
        assert isinstance(classifier.model_loader, ModelLoader)
        assert classifier.models == {}
        assert classifier._is_loaded is False
    
    def test_load_models_success(self, temp_models_dir, sample_models_and_metadata):
        """Test successful model loading."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        assert classifier._is_loaded is True
        assert len(classifier.models) == 3
        assert "bihag" in classifier.models
        assert "darbari" in classifier.models
        assert "desh" in classifier.models
    
    def test_load_models_force_reload(self, temp_models_dir, sample_models_and_metadata):
        """Test force reloading of models."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        
        # First load
        classifier.load_models()
        first_models = classifier.models.copy()
        
        # Force reload
        classifier.load_models(force_reload=True)
        
        # Should have reloaded (different object instances)
        assert classifier._is_loaded is True
        assert len(classifier.models) == len(first_models)
    
    def test_load_models_no_models_available(self, temp_models_dir):
        """Test error handling when no models are available."""
        classifier = RaagClassifier(temp_models_dir)
        
        with pytest.raises(ClassificationError, match="No model files found"):
            classifier.load_models()
    
    def test_score_sequence_success(self, temp_models_dir, sample_models_and_metadata):
        """Test successful sequence scoring."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Create test sequence
        sequence = np.array([0, 5, 10, 15, 20, 25, 30, 35, 0])
        
        score = classifier.score_sequence(sequence, "bihag")
        
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)
    
    def test_score_sequence_invalid_raag(self, temp_models_dir, sample_models_and_metadata):
        """Test error handling for invalid raag name."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        sequence = np.array([0, 5, 10])
        
        with pytest.raises(ClassificationError, match="Raag model 'invalid_raag' not found"):
            classifier.score_sequence(sequence, "invalid_raag")
    
    def test_score_sequence_empty_sequence(self, temp_models_dir, sample_models_and_metadata):
        """Test error handling for empty sequence."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        empty_sequence = np.array([])
        
        with pytest.raises(ClassificationError, match="Empty sequence provided"):
            classifier.score_sequence(empty_sequence, "bihag")
    
    def test_score_sequence_invalid_observations(self, temp_models_dir, sample_models_and_metadata):
        """Test error handling for invalid observation indices."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Sequence with invalid observation indices
        invalid_sequence = np.array([0, 5, 50, 15])  # 50 is out of range [0, 35]
        
        with pytest.raises(ClassificationError, match="Sequence contains invalid observation indices"):
            classifier.score_sequence(invalid_sequence, "bihag")
    
    def test_predict_raag_success(self, temp_models_dir, sample_models_and_metadata):
        """Test successful raag prediction."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        sequence = np.array([0, 5, 10, 15, 20, 25, 30, 35, 0])
        
        # Test without returning all scores
        predicted_raag = classifier.predict_raag(sequence, return_all_scores=False)
        assert predicted_raag in ["bihag", "darbari", "desh"]
        
        # Test with returning all scores
        predicted_raag, scores = classifier.predict_raag(sequence, return_all_scores=True)
        assert predicted_raag in ["bihag", "darbari", "desh"]
        assert isinstance(scores, dict)
        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores.values())
    
    def test_predict_raag_no_models(self, temp_models_dir):
        """Test error handling when no models are loaded."""
        classifier = RaagClassifier(temp_models_dir)
        
        sequence = np.array([0, 5, 10])
        
        with pytest.raises(ClassificationError, match="No model files found"):
            classifier.predict_raag(sequence)
    
    def test_predict_with_confidence_success(self, temp_models_dir, sample_models_and_metadata):
        """Test prediction with confidence scores."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        sequence = np.array([0, 5, 10, 15, 20, 25, 30, 35, 0])
        
        # Test with normalized scores
        result = classifier.predict_with_confidence(sequence, normalize_scores=True)
        
        assert 'predicted_raag' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert 'probabilities' in result
        assert 'ranking' in result
        
        assert result['predicted_raag'] in ["bihag", "darbari", "desh"]
        assert 0 <= result['confidence'] <= 1  # Probability should be in [0, 1]
        assert len(result['scores']) == 3
        assert len(result['probabilities']) == 3
        assert len(result['ranking']) == 3
        
        # Check that probabilities sum to 1
        prob_sum = sum(result['probabilities'].values())
        assert abs(prob_sum - 1.0) < 1e-6
        
        # Test without normalized scores
        result_no_norm = classifier.predict_with_confidence(sequence, normalize_scores=False)
        assert 'probabilities' not in result_no_norm
        assert 'confidence' in result_no_norm  # Should be score margin
    
    def test_get_available_raags(self, temp_models_dir, sample_models_and_metadata):
        """Test getting available raag classes."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        
        # Before loading
        raags = classifier.get_available_raags()
        assert len(raags) == 3
        assert set(raags) == {"bihag", "darbari", "desh"}
        
        # After loading (should be the same)
        classifier.load_models()
        raags_after = classifier.get_available_raags()
        assert raags == raags_after
    
    def test_get_model_info(self, temp_models_dir, sample_models_and_metadata):
        """Test getting model information."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        model_info = classifier.get_model_info()
        
        assert len(model_info) == 3
        assert "bihag" in model_info
        
        bihag_info = model_info["bihag"]
        assert bihag_info['n_states'] == 36
        assert bihag_info['n_observations'] == 36
        assert bihag_info['n_sequences'] == 10
        assert bihag_info['converged'] is True
    
    def test_clear_cache(self, temp_models_dir, sample_models_and_metadata):
        """Test cache clearing functionality."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        assert classifier._is_loaded is True
        assert len(classifier.models) > 0
        
        classifier.clear_cache()
        
        assert classifier._is_loaded is False
        assert len(classifier.models) == 0
    
    def test_auto_load_on_prediction(self, temp_models_dir, sample_models_and_metadata):
        """Test that models are automatically loaded when needed for prediction."""
        self.create_test_models(temp_models_dir, sample_models_and_metadata)
        
        classifier = RaagClassifier(temp_models_dir)
        
        # Models not loaded yet
        assert classifier._is_loaded is False
        
        sequence = np.array([0, 5, 10, 15, 20])
        
        # This should automatically load models
        predicted_raag = classifier.predict_raag(sequence)
        
        assert classifier._is_loaded is True
        assert len(classifier.models) == 3
        assert predicted_raag in ["bihag", "darbari", "desh"]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_filename_sanitization(self):
        """Test filename sanitization for various raag names."""
        loader = ModelLoader("dummy_dir")
        
        # Test various problematic characters
        test_cases = [
            ("Raag Bihag", "raag_bihag"),
            ("Darbari-Kanada", "darbari_kanada"),
            ("Gaud Malhar", "gaud_malhar"),
            ("Raag@#$%", "raag"),
            ("123-Numeric", "123_numeric"),
            ("UPPERCASE", "uppercase")
        ]
        
        for input_name, expected_output in test_cases:
            result = loader._sanitize_filename(input_name)
            assert result == expected_output
    
    def test_numerical_stability_with_extreme_sequences(self, temp_models_dir):
        """Test numerical stability with extreme sequences."""
        # Create a simple model for testing
        model = DiscreteHMM(n_states=36, n_observations=36, random_state=42)
        metadata = {
            'model_class': 'DiscreteHMM',
            'model_parameters': {'n_states': 36, 'n_observations': 36},
            'saved_at': '2024-01-01T12:00:00'
        }
        
        models_path = Path(temp_models_dir)
        model_path = models_path / "test_raag.pkl"
        metadata_path = models_path / "test_raag_meta.json"
        
        joblib.dump(model, model_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        classifier = RaagClassifier(temp_models_dir)
        classifier.load_models()
        
        # Test with very long sequence
        long_sequence = np.random.randint(0, 36, size=10000)
        score = classifier.score_sequence(long_sequence, "test_raag")
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)
        
        # Test with single observation
        single_obs = np.array([15])
        score_single = classifier.score_sequence(single_obs, "test_raag")
        assert isinstance(score_single, float)
        assert not np.isnan(score_single)
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for test models."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)