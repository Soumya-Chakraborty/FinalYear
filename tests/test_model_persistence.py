"""
Unit tests for model persistence and serialization/deserialization consistency.

Tests the ModelPersistence class functionality including serialization,
deserialization, metadata handling, and file management.
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

from raag_hmm.train.persistence import ModelPersistence
from raag_hmm.hmm.model import DiscreteHMM
from raag_hmm.exceptions import ModelTrainingError


class TestModelPersistence:
    """Unit tests for ModelPersistence class."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def persistence(self, temp_models_dir):
        """Create ModelPersistence instance."""
        return ModelPersistence(temp_models_dir)
    
    @pytest.fixture
    def sample_model(self):
        """Create sample HMM model for testing."""
        return DiscreteHMM(n_states=6, n_observations=6, random_state=42)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata with various data types."""
        return {
            'raag_name': 'TestRaag',
            'n_sequences': 10,
            'total_frames': 500,
            'convergence_iterations': 25,
            'final_log_likelihood': -234.567,
            'converged': True,
            'training_time': 12.34,
            'hyperparameters': {
                'n_states': 6,
                'n_observations': 6,
                'max_iterations': 100,
                'convergence_tolerance': 0.1,
                'regularization_alpha': 0.01,
                'probability_floor': 1e-8,
                'random_state': 42
            },
            'training_stats': {
                'log_likelihood_history': [-300.0, -280.0, -250.0, -234.567],
                'improvement_history': [20.0, 30.0, 15.433],
                'converged': True,
                'iterations': 3
            },
            # Test numpy array serialization
            'numpy_array': np.array([1.0, 2.0, 3.0]),
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14159),
            # Test nested structures
            'nested_dict': {
                'inner_array': np.array([[1, 2], [3, 4]]),
                'inner_list': [np.float32(1.5), np.int32(10)]
            }
        }
    
    def test_initialization(self, temp_models_dir):
        """Test ModelPersistence initialization."""
        persistence = ModelPersistence(temp_models_dir)
        
        assert persistence.models_dir == Path(temp_models_dir)
        assert persistence.models_dir.exists()
        assert persistence.models_dir.is_dir()
    
    def test_initialization_creates_directory(self):
        """Test that initialization creates models directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "nonexistent" / "models"
            
            persistence = ModelPersistence(str(models_dir))
            
            assert models_dir.exists()
            assert models_dir.is_dir()
    
    def test_sanitize_filename(self, persistence):
        """Test filename sanitization."""
        # Test basic sanitization
        assert persistence._sanitize_filename("Simple") == "simple"
        assert persistence._sanitize_filename("With Spaces") == "with_spaces"
        assert persistence._sanitize_filename("With-Dashes") == "with_dashes"
        assert persistence._sanitize_filename("With_Underscores") == "with_underscores"
        
        # Test special characters
        assert persistence._sanitize_filename("Raag@#$%") == "raag"
        assert persistence._sanitize_filename("Test(123)") == "test123"
        assert persistence._sanitize_filename("Mix3d-Ch@rs_123") == "mix3d_chrs_123"
        
        # Test empty and edge cases
        assert persistence._sanitize_filename("") == ""
        assert persistence._sanitize_filename("123") == "123"
        assert persistence._sanitize_filename("_") == "_"
    
    def test_prepare_metadata_for_serialization(self, persistence, sample_metadata):
        """Test metadata preparation for JSON serialization."""
        serializable = persistence._prepare_metadata_for_serialization(sample_metadata)
        
        # Test basic types are preserved
        assert serializable['raag_name'] == sample_metadata['raag_name']
        assert serializable['n_sequences'] == sample_metadata['n_sequences']
        assert serializable['converged'] == sample_metadata['converged']
        
        # Test numpy array conversion
        assert isinstance(serializable['numpy_array'], list)
        assert serializable['numpy_array'] == [1.0, 2.0, 3.0]
        
        # Test numpy scalar conversion
        assert isinstance(serializable['numpy_int'], int)
        assert serializable['numpy_int'] == 42
        assert isinstance(serializable['numpy_float'], float)
        assert abs(serializable['numpy_float'] - 3.14159) < 1e-10
        
        # Test nested structure conversion
        nested = serializable['nested_dict']
        assert isinstance(nested['inner_array'], list)
        assert nested['inner_array'] == [[1, 2], [3, 4]]
        assert isinstance(nested['inner_list'][0], float)
        assert isinstance(nested['inner_list'][1], int)
        
        # Verify the result is JSON serializable
        json_str = json.dumps(serializable)
        assert isinstance(json_str, str)
        
        # Verify round-trip
        deserialized = json.loads(json_str)
        assert deserialized['raag_name'] == sample_metadata['raag_name']
        assert deserialized['numpy_array'] == [1.0, 2.0, 3.0]
    
    def test_save_model_basic(self, persistence, sample_model, sample_metadata):
        """Test basic model saving functionality."""
        model_path, metadata_path = persistence.save_model(
            "TestRaag", sample_model, sample_metadata
        )
        
        # Verify file paths
        assert model_path.endswith("testraag.pkl")
        assert metadata_path.endswith("testraag_meta.json")
        
        # Verify files exist
        assert Path(model_path).exists()
        assert Path(metadata_path).exists()
        
        # Verify file sizes are reasonable
        assert Path(model_path).stat().st_size > 0
        assert Path(metadata_path).stat().st_size > 0
    
    def test_save_model_metadata_enhancement(self, persistence, sample_model, sample_metadata):
        """Test that save_model adds required metadata fields."""
        model_path, metadata_path = persistence.save_model(
            "TestRaag", sample_model, sample_metadata
        )
        
        # Load and verify enhanced metadata
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        # Check added fields
        assert 'saved_at' in saved_metadata
        assert 'model_file' in saved_metadata
        assert 'metadata_file' in saved_metadata
        assert 'model_class' in saved_metadata
        assert 'model_parameters' in saved_metadata
        
        # Verify values
        assert saved_metadata['model_class'] == 'DiscreteHMM'
        assert saved_metadata['model_parameters']['n_states'] == 6
        assert saved_metadata['model_parameters']['n_observations'] == 6
        assert saved_metadata['model_file'] == "testraag.pkl"
        assert saved_metadata['metadata_file'] == "testraag_meta.json"
        
        # Verify timestamp format
        saved_at = datetime.fromisoformat(saved_metadata['saved_at'])
        assert isinstance(saved_at, datetime)
    
    def test_load_model_basic(self, persistence, sample_model, sample_metadata):
        """Test basic model loading functionality."""
        # Save model first
        persistence.save_model("TestRaag", sample_model, sample_metadata)
        
        # Load model
        loaded_model, loaded_metadata = persistence.load_model("TestRaag")
        
        # Verify model type and parameters
        assert isinstance(loaded_model, DiscreteHMM)
        assert loaded_model.n_states == sample_model.n_states
        assert loaded_model.n_observations == sample_model.n_observations
        
        # Verify stochastic matrices are valid
        loaded_model.validate_stochastic_matrices()
        
        # Verify metadata preservation
        assert loaded_metadata['raag_name'] == sample_metadata['raag_name']
        assert loaded_metadata['n_sequences'] == sample_metadata['n_sequences']
        assert loaded_metadata['final_log_likelihood'] == sample_metadata['final_log_likelihood']
    
    def test_serialization_deserialization_consistency(self, persistence, sample_model, sample_metadata):
        """Test that model parameters are preserved through save/load cycle."""
        # Get original parameters
        orig_pi, orig_A, orig_B = sample_model.get_parameters()
        
        # Save and load
        persistence.save_model("TestRaag", sample_model, sample_metadata)
        loaded_model, _ = persistence.load_model("TestRaag")
        
        # Get loaded parameters
        loaded_pi, loaded_A, loaded_B = loaded_model.get_parameters()
        
        # Verify parameter consistency
        np.testing.assert_array_almost_equal(orig_pi, loaded_pi, decimal=10)
        np.testing.assert_array_almost_equal(orig_A, loaded_A, decimal=10)
        np.testing.assert_array_almost_equal(orig_B, loaded_B, decimal=10)
        
        # Verify stochastic properties are preserved
        assert np.allclose(loaded_pi.sum(), 1.0)
        assert np.allclose(loaded_A.sum(axis=1), 1.0)
        assert np.allclose(loaded_B.sum(axis=1), 1.0)
    
    def test_model_scoring_consistency(self, persistence, sample_model, sample_metadata):
        """Test that model scoring is consistent after save/load."""
        # Create test sequence
        test_sequence = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2])
        
        # Score with original model
        original_score = sample_model.score(test_sequence)
        
        # Save and load model
        persistence.save_model("TestRaag", sample_model, sample_metadata)
        loaded_model, _ = persistence.load_model("TestRaag")
        
        # Score with loaded model
        loaded_score = loaded_model.score(test_sequence)
        
        # Verify scores are identical
        assert abs(original_score - loaded_score) < 1e-10
    
    def test_overwrite_protection(self, persistence, sample_model, sample_metadata):
        """Test overwrite protection mechanism."""
        # Save model
        persistence.save_model("TestRaag", sample_model, sample_metadata)
        
        # Try to save again without overwrite flag
        with pytest.raises(ModelTrainingError, match="already exists"):
            persistence.save_model("TestRaag", sample_model, sample_metadata, overwrite=False)
        
        # Save with overwrite should succeed
        model_path, metadata_path = persistence.save_model(
            "TestRaag", sample_model, sample_metadata, overwrite=True
        )
        
        assert Path(model_path).exists()
        assert Path(metadata_path).exists()
    
    def test_load_nonexistent_model(self, persistence):
        """Test loading non-existent model raises appropriate error."""
        with pytest.raises(ModelTrainingError, match="not found"):
            persistence.load_model("NonExistentRaag")
    
    def test_load_corrupted_model_file(self, persistence, sample_metadata):
        """Test loading corrupted model file."""
        # Create corrupted model file
        model_path = persistence.models_dir / "corrupted.pkl"
        with open(model_path, 'w') as f:
            f.write("This is not a valid pickle file")
        
        # Create valid metadata file
        metadata_path = persistence.models_dir / "corrupted_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        # Loading should fail
        with pytest.raises(ModelTrainingError):
            persistence.load_model("corrupted")
    
    def test_load_corrupted_metadata_file(self, persistence, sample_model):
        """Test loading corrupted metadata file."""
        # Save valid model
        model_path = persistence.models_dir / "corrupted.pkl"
        import joblib
        joblib.dump(sample_model, model_path)
        
        # Create corrupted metadata file
        metadata_path = persistence.models_dir / "corrupted_meta.json"
        with open(metadata_path, 'w') as f:
            f.write("This is not valid JSON")
        
        # Loading should fail
        with pytest.raises(ModelTrainingError):
            persistence.load_model("corrupted")
    
    def test_validate_model_metadata_consistency(self, persistence, sample_model, sample_metadata):
        """Test model-metadata consistency validation."""
        # Create inconsistent metadata
        inconsistent_metadata = sample_metadata.copy()
        inconsistent_metadata['model_parameters'] = {
            'n_states': 10,  # Different from model's 6 states
            'n_observations': 6
        }
        
        # Save with inconsistent metadata
        persistence.save_model("TestRaag", sample_model, inconsistent_metadata, overwrite=True)
        
        # Loading should fail validation
        with pytest.raises(ModelTrainingError, match="mismatch"):
            persistence.load_model("TestRaag")
    
    def test_save_all_models(self, persistence):
        """Test saving multiple models."""
        # Create multiple models
        trained_models = {}
        
        for i, raag in enumerate(['Bihag', 'Darbari', 'Desh']):
            model = DiscreteHMM(n_states=4, n_observations=4, random_state=42 + i)
            metadata = {
                'raag_name': raag,
                'n_sequences': 5 + i,
                'total_frames': 100 + i * 20,
                'final_log_likelihood': -150.0 - i * 10,
                'converged': True
            }
            trained_models[raag] = (model, metadata)
        
        # Save all models
        saved_paths = persistence.save_all_models(trained_models)
        
        # Verify all saved
        assert len(saved_paths) == 3
        
        for raag, (model_path, metadata_path) in saved_paths.items():
            assert Path(model_path).exists()
            assert Path(metadata_path).exists()
        
        # Verify summary file
        summary_path = persistence.models_dir / "models_summary.json"
        assert summary_path.exists()
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert summary['total_models'] == 3
        assert len(summary['models']) == 3
        
        for raag in ['Bihag', 'Darbari', 'Desh']:
            assert raag in summary['models']
            model_info = summary['models'][raag]
            assert 'model_file' in model_info
            assert 'metadata_file' in model_info
            assert 'model_size_mb' in model_info
    
    def test_load_all_models(self, persistence):
        """Test loading all available models."""
        # Save multiple models first
        trained_models = {}
        
        for raag in ['Bihag', 'Darbari']:
            model = DiscreteHMM(n_states=4, n_observations=4, random_state=42)
            metadata = {
                'raag_name': raag,
                'n_sequences': 3,
                'total_frames': 60,
                'final_log_likelihood': -120.0,
                'converged': True
            }
            trained_models[raag] = (model, metadata)
        
        persistence.save_all_models(trained_models)
        
        # Load all models
        loaded_models = persistence.load_all_models()
        
        # Verify loaded models
        assert len(loaded_models) == 2
        assert 'bihag' in loaded_models  # Sanitized names
        assert 'darbari' in loaded_models
        
        for raag, (model, metadata) in loaded_models.items():
            assert isinstance(model, DiscreteHMM)
            assert model.n_states == 4
            assert model.n_observations == 4
            assert isinstance(metadata, dict)
    
    def test_list_available_models(self, persistence):
        """Test listing available models."""
        # Initially empty
        models_info = persistence.list_available_models()
        assert len(models_info) == 0
        
        # Save some models
        for i, raag in enumerate(['Bihag', 'Darbari']):
            model = DiscreteHMM(n_states=4, n_observations=4, random_state=42)
            metadata = {
                'raag_name': raag,
                'n_sequences': 3 + i,
                'total_frames': 60 + i * 10,
                'converged': True,
                'final_log_likelihood': -100.0 - i * 5,
                'training_time': 2.0 + i * 0.5
            }
            persistence.save_model(raag, model, metadata)
        
        # List models
        models_info = persistence.list_available_models()
        assert len(models_info) == 2
        
        # Verify information
        for info in models_info:
            assert info['model_exists'] is True
            assert info['metadata_exists'] is True
            assert info['model_size_mb'] > 0
            assert 'n_sequences' in info
            assert 'total_frames' in info
            assert 'converged' in info
            assert 'saved_at' in info
    
    def test_delete_model(self, persistence, sample_model, sample_metadata):
        """Test model deletion."""
        # Save model
        persistence.save_model("TestRaag", sample_model, sample_metadata)
        
        # Verify exists
        models_info = persistence.list_available_models()
        assert len(models_info) == 1
        
        # Delete model
        success = persistence.delete_model("TestRaag")
        assert success is True
        
        # Verify deleted
        models_info = persistence.list_available_models()
        assert len(models_info) == 0
        
        # Delete non-existent model
        success = persistence.delete_model("NonExistent")
        assert success is False
    
    def test_edge_cases_empty_metadata(self, persistence, sample_model):
        """Test handling of empty or minimal metadata."""
        minimal_metadata = {'raag_name': 'MinimalRaag'}
        
        # Should work with minimal metadata
        model_path, metadata_path = persistence.save_model(
            "MinimalRaag", sample_model, minimal_metadata
        )
        
        # Should load successfully
        loaded_model, loaded_metadata = persistence.load_model("MinimalRaag")
        
        assert isinstance(loaded_model, DiscreteHMM)
        assert loaded_metadata['raag_name'] == 'MinimalRaag'
        assert 'saved_at' in loaded_metadata  # Added by save_model
    
    def test_large_metadata_handling(self, persistence, sample_model):
        """Test handling of large metadata with many arrays."""
        # Create metadata with large arrays
        large_metadata = {
            'raag_name': 'LargeMetadataRaag',
            'large_array': np.random.randn(1000).tolist(),
            'matrix': np.random.randn(50, 50).tolist(),
            'history': list(range(1000)),
            'nested_large': {
                'inner_array': np.random.randn(500).tolist(),
                'inner_matrix': np.random.randn(20, 20).tolist()
            }
        }
        
        # Should handle large metadata
        model_path, metadata_path = persistence.save_model(
            "LargeMetadataRaag", sample_model, large_metadata
        )
        
        # Should load successfully
        loaded_model, loaded_metadata = persistence.load_model("LargeMetadataRaag")
        
        assert isinstance(loaded_model, DiscreteHMM)
        assert len(loaded_metadata['large_array']) == 1000
        assert len(loaded_metadata['matrix']) == 50
        assert len(loaded_metadata['matrix'][0]) == 50