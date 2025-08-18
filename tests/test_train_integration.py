"""
Integration tests for multi-class training pipeline.

Tests the complete training workflow from dataset loading through
model training and persistence for multiple raag classes.
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from raag_hmm.train.trainer import RaagTrainer
from raag_hmm.train.persistence import ModelPersistence
from raag_hmm.hmm.model import DiscreteHMM
from raag_hmm.exceptions import ModelTrainingError


class TestRaagTrainerIntegration:
    """Integration tests for RaagTrainer class."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create dataset structure
        train_dir = Path(temp_dir) / "train"
        audio_dir = train_dir / "audio"
        metadata_dir = train_dir / "metadata"
        
        audio_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        
        # Create mock audio files (empty files for testing)
        raag_classes = ['Bihag', 'Darbari', 'Desh']
        
        for i, raag in enumerate(raag_classes):
            for j in range(2):  # 2 files per raag
                # Create audio file
                audio_file = audio_dir / f"{raag}_{j:02d}.wav"
                audio_file.touch()
                
                # Create metadata file
                metadata = {
                    'recording_id': f"{raag}_{j:02d}",
                    'raag': raag,
                    'tonic_hz': 220.0 + i * 10,  # Different tonics
                    'artist': f"Artist_{i}",
                    'instrument': 'sitar',
                    'split': 'train'
                }
                
                metadata_file = metadata_dir / f"{raag}_{j:02d}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def trainer(self):
        """Create RaagTrainer instance for testing."""
        return RaagTrainer(
            n_states=4,  # Smaller for faster testing
            n_observations=4,
            max_iterations=5,  # Fewer iterations for testing
            convergence_tolerance=1.0,  # Loose tolerance
            random_state=42
        )
    
    def test_extract_and_quantize_sequence_mock(self, trainer, temp_dataset):
        """Test sequence extraction with mocked audio processing."""
        audio_path = str(Path(temp_dataset) / "train" / "audio" / "Bihag_00.wav")
        
        # Mock the audio processing pipeline
        with patch('raag_hmm.train.trainer.load_audio') as mock_load, \
             patch('raag_hmm.train.trainer.extract_pitch_praat') as mock_praat, \
             patch('raag_hmm.train.trainer.smooth_pitch') as mock_smooth:
            
            # Setup mocks
            mock_load.return_value = np.random.randn(22050)  # 1 second of audio
            mock_praat.return_value = (
                np.array([220.0, 246.94, 261.63, 293.66] * 10),  # 40 frames
                np.array([0.8, 0.9, 0.7, 0.8] * 10)  # voicing probabilities
            )
            mock_smooth.return_value = np.array([220.0, 246.94, 261.63, 293.66] * 10)
            
            # Test extraction
            sequence = trainer.extract_and_quantize_sequence(audio_path, tonic_hz=220.0)
            
            # Verify calls
            mock_load.assert_called_once()
            mock_praat.assert_called_once()
            mock_smooth.assert_called_once()
            
            # Verify output
            assert isinstance(sequence, np.ndarray)
            assert sequence.dtype == int
            assert len(sequence) > 0
            assert np.all(sequence >= 0)
            assert np.all(sequence < 36)
    
    def test_group_sequences_by_raag_mock(self, trainer, temp_dataset):
        """Test sequence grouping with mocked processing."""
        with patch.object(trainer, 'extract_and_quantize_sequence') as mock_extract:
            # Mock sequence extraction to return different sequences for each file
            mock_extract.side_effect = [
                np.array([0, 1, 2, 3] * 5),  # Bihag_00
                np.array([1, 2, 3, 0] * 5),  # Bihag_01
                np.array([2, 3, 0, 1] * 5),  # Darbari_00
                np.array([3, 0, 1, 2] * 5),  # Darbari_01
                np.array([0, 2, 1, 3] * 5),  # Desh_00
                np.array([1, 3, 2, 0] * 5),  # Desh_01
            ]
            
            # Test grouping
            raag_sequences = trainer.group_sequences_by_raag(temp_dataset, split="train")
            
            # Verify results
            assert isinstance(raag_sequences, dict)
            assert len(raag_sequences) == 3
            assert 'Bihag' in raag_sequences
            assert 'Darbari' in raag_sequences
            assert 'Desh' in raag_sequences
            
            # Check sequence counts
            for raag, sequences in raag_sequences.items():
                assert len(sequences) == 2  # 2 files per raag
                for seq in sequences:
                    assert isinstance(seq, np.ndarray)
                    assert len(seq) == 20  # 4 * 5
    
    def test_train_raag_model(self, trainer):
        """Test training a single raag model."""
        # Create synthetic sequences
        sequences = [
            np.array([0, 1, 2, 1, 0] * 4),
            np.array([1, 2, 3, 2, 1] * 4),
            np.array([0, 2, 1, 3, 0] * 4)
        ]
        
        # Train model
        model, metadata = trainer.train_raag_model("TestRaag", sequences, verbose=False)
        
        # Verify model
        assert isinstance(model, DiscreteHMM)
        assert model.n_states == trainer.n_states
        assert model.n_observations == trainer.n_observations
        
        # Verify metadata
        assert isinstance(metadata, dict)
        assert metadata['raag_name'] == "TestRaag"
        assert metadata['n_sequences'] == 3
        assert metadata['total_frames'] == 60  # 3 * 20
        assert 'final_log_likelihood' in metadata
        assert 'training_time' in metadata
        assert 'hyperparameters' in metadata
    
    def test_train_all_raag_models_mock(self, trainer, temp_dataset):
        """Test complete multi-raag training pipeline with mocks."""
        with patch.object(trainer, 'extract_and_quantize_sequence') as mock_extract:
            # Mock sequence extraction
            mock_extract.side_effect = [
                np.array([0, 1, 2, 1] * 5),  # Bihag_00
                np.array([1, 2, 3, 2] * 5),  # Bihag_01
                np.array([2, 3, 0, 3] * 5),  # Darbari_00
                np.array([3, 0, 1, 0] * 5),  # Darbari_01
                np.array([0, 2, 1, 2] * 5),  # Desh_00
                np.array([1, 3, 2, 3] * 5),  # Desh_01
            ]
            
            # Train all models
            trained_models = trainer.train_all_raag_models(
                temp_dataset, 
                split="train", 
                verbose=False
            )
            
            # Verify results
            assert isinstance(trained_models, dict)
            assert len(trained_models) == 3
            
            expected_raags = {'Bihag', 'Darbari', 'Desh'}
            assert set(trained_models.keys()) == expected_raags
            
            # Verify each model
            for raag_name, (model, metadata) in trained_models.items():
                assert isinstance(model, DiscreteHMM)
                assert isinstance(metadata, dict)
                assert metadata['raag_name'] == raag_name
                assert metadata['n_sequences'] == 2
                assert metadata['total_frames'] == 40  # 2 * 20
            
            # Verify training statistics
            stats = trainer.get_training_summary()
            assert stats['total_sequences'] == 6
            assert stats['total_frames'] == 120
            assert len(stats['raag_summary']) == 3
    
    def test_train_empty_dataset(self, trainer):
        """Test training with empty dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty dataset structure
            train_dir = Path(temp_dir) / "train"
            (train_dir / "audio").mkdir(parents=True)
            (train_dir / "metadata").mkdir(parents=True)
            
            # Should raise error for empty dataset
            with pytest.raises(ModelTrainingError, match="No valid sequences found"):
                trainer.train_all_raag_models(temp_dir, split="train")
    
    def test_train_raag_model_empty_sequences(self, trainer):
        """Test training with empty sequence list."""
        with pytest.raises(ModelTrainingError, match="No sequences provided"):
            trainer.train_raag_model("TestRaag", [])


class TestModelPersistenceIntegration:
    """Integration tests for ModelPersistence class."""
    
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
    def sample_model_and_metadata(self):
        """Create sample model and metadata for testing."""
        model = DiscreteHMM(n_states=4, n_observations=4, random_state=42)
        
        metadata = {
            'raag_name': 'TestRaag',
            'n_sequences': 5,
            'total_frames': 100,
            'convergence_iterations': 10,
            'final_log_likelihood': -150.5,
            'converged': True,
            'training_time': 2.5,
            'hyperparameters': {
                'n_states': 4,
                'n_observations': 4,
                'max_iterations': 50,
                'convergence_tolerance': 0.1
            },
            'training_stats': {
                'log_likelihood_history': [-200.0, -180.0, -160.0, -150.5],
                'improvement_history': [20.0, 20.0, 9.5]
            }
        }
        
        return model, metadata
    
    def test_save_and_load_model(self, persistence, sample_model_and_metadata):
        """Test saving and loading a single model."""
        model, metadata = sample_model_and_metadata
        
        # Save model
        model_path, metadata_path = persistence.save_model("TestRaag", model, metadata)
        
        # Verify files exist
        assert Path(model_path).exists()
        assert Path(metadata_path).exists()
        
        # Load model
        loaded_model, loaded_metadata = persistence.load_model("TestRaag")
        
        # Verify loaded model
        assert isinstance(loaded_model, DiscreteHMM)
        assert loaded_model.n_states == model.n_states
        assert loaded_model.n_observations == model.n_observations
        
        # Verify loaded metadata
        assert loaded_metadata['raag_name'] == metadata['raag_name']
        assert loaded_metadata['n_sequences'] == metadata['n_sequences']
        assert loaded_metadata['final_log_likelihood'] == metadata['final_log_likelihood']
        assert 'saved_at' in loaded_metadata
        assert 'model_file' in loaded_metadata
    
    def test_save_all_models(self, persistence, temp_models_dir):
        """Test saving multiple models."""
        # Create multiple models
        trained_models = {}
        
        for i, raag in enumerate(['Bihag', 'Darbari', 'Desh']):
            model = DiscreteHMM(n_states=4, n_observations=4, random_state=42 + i)
            metadata = {
                'raag_name': raag,
                'n_sequences': 3 + i,
                'total_frames': 50 + i * 10,
                'final_log_likelihood': -100.0 - i * 10,
                'converged': True,
                'training_time': 1.0 + i * 0.5
            }
            trained_models[raag] = (model, metadata)
        
        # Save all models
        saved_paths = persistence.save_all_models(trained_models)
        
        # Verify all files saved
        assert len(saved_paths) == 3
        for raag, (model_path, metadata_path) in saved_paths.items():
            assert Path(model_path).exists()
            assert Path(metadata_path).exists()
        
        # Verify summary file created
        summary_path = Path(temp_models_dir) / "models_summary.json"
        assert summary_path.exists()
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert summary['total_models'] == 3
        assert len(summary['models']) == 3
    
    def test_load_all_models(self, persistence, temp_models_dir):
        """Test loading all available models."""
        # First save some models
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
        assert 'bihag' in loaded_models  # Sanitized name
        assert 'darbari' in loaded_models
        
        for raag, (model, metadata) in loaded_models.items():
            assert isinstance(model, DiscreteHMM)
            assert isinstance(metadata, dict)
    
    def test_list_available_models(self, persistence, temp_models_dir):
        """Test listing available models."""
        # Initially empty
        models_info = persistence.list_available_models()
        assert len(models_info) == 0
        
        # Save a model
        model = DiscreteHMM(n_states=4, n_observations=4, random_state=42)
        metadata = {
            'raag_name': 'TestRaag',
            'n_sequences': 5,
            'total_frames': 100,
            'converged': True
        }
        persistence.save_model("TestRaag", model, metadata)
        
        # List models
        models_info = persistence.list_available_models()
        assert len(models_info) == 1
        
        info = models_info[0]
        assert info['raag_name'] == 'testraag'  # Sanitized
        assert info['model_exists'] is True
        assert info['metadata_exists'] is True
        assert info['n_sequences'] == 5
        assert info['total_frames'] == 100
    
    def test_delete_model(self, persistence, sample_model_and_metadata):
        """Test deleting a model."""
        model, metadata = sample_model_and_metadata
        
        # Save model
        persistence.save_model("TestRaag", model, metadata)
        
        # Verify files exist
        models_info = persistence.list_available_models()
        assert len(models_info) == 1
        
        # Delete model
        success = persistence.delete_model("TestRaag")
        assert success is True
        
        # Verify files deleted
        models_info = persistence.list_available_models()
        assert len(models_info) == 0
    
    def test_overwrite_protection(self, persistence, sample_model_and_metadata):
        """Test overwrite protection."""
        model, metadata = sample_model_and_metadata
        
        # Save model
        persistence.save_model("TestRaag", model, metadata)
        
        # Try to save again without overwrite
        with pytest.raises(ModelTrainingError, match="already exists"):
            persistence.save_model("TestRaag", model, metadata, overwrite=False)
        
        # Save with overwrite should work
        persistence.save_model("TestRaag", model, metadata, overwrite=True)
    
    def test_filename_sanitization(self, persistence):
        """Test filename sanitization for special characters."""
        model = DiscreteHMM(n_states=4, n_observations=4, random_state=42)
        metadata = {'raag_name': 'Test-Raag With Spaces'}
        
        # Save with special characters in name
        model_path, metadata_path = persistence.save_model(
            "Test-Raag With Spaces", model, metadata
        )
        
        # Verify sanitized filenames
        assert "test_raag_with_spaces.pkl" in model_path
        assert "test_raag_with_spaces_meta.json" in metadata_path
        
        # Should be able to load with original name
        loaded_model, loaded_metadata = persistence.load_model("Test-Raag With Spaces")
        assert isinstance(loaded_model, DiscreteHMM)


class TestEndToEndTrainingPipeline:
    """End-to-end integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with dataset and models directories."""
        temp_dir = tempfile.mkdtemp()
        
        # Create dataset
        dataset_dir = Path(temp_dir) / "dataset"
        train_dir = dataset_dir / "train"
        audio_dir = train_dir / "audio"
        metadata_dir = train_dir / "metadata"
        
        audio_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        
        # Create models directory
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir()
        
        yield {
            'root': temp_dir,
            'dataset': str(dataset_dir),
            'models': str(models_dir)
        }
        
        shutil.rmtree(temp_dir)
    
    def test_complete_pipeline_mock(self, temp_workspace):
        """Test complete pipeline from training to persistence with mocks."""
        # Create mock dataset
        dataset_dir = temp_workspace['dataset']
        train_dir = Path(dataset_dir) / "train"
        audio_dir = train_dir / "audio"
        metadata_dir = train_dir / "metadata"
        
        # Create files for 2 raag classes
        raag_classes = ['Bihag', 'Darbari']
        
        for raag in raag_classes:
            for i in range(2):
                # Audio file
                audio_file = audio_dir / f"{raag}_{i}.wav"
                audio_file.touch()
                
                # Metadata file
                metadata = {
                    'recording_id': f"{raag}_{i}",
                    'raag': raag,
                    'tonic_hz': 220.0,
                    'split': 'train'
                }
                
                metadata_file = metadata_dir / f"{raag}_{i}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
        
        # Initialize trainer and persistence
        trainer = RaagTrainer(
            n_states=4,
            n_observations=4,
            max_iterations=3,
            random_state=42
        )
        
        persistence = ModelPersistence(temp_workspace['models'])
        
        # Mock audio processing
        with patch.object(trainer, 'extract_and_quantize_sequence') as mock_extract:
            mock_extract.side_effect = [
                np.array([0, 1, 2, 3] * 5),  # Bihag_0
                np.array([1, 2, 3, 0] * 5),  # Bihag_1
                np.array([2, 3, 0, 1] * 5),  # Darbari_0
                np.array([3, 0, 1, 2] * 5),  # Darbari_1
            ]
            
            # Step 1: Train all models
            trained_models = trainer.train_all_raag_models(
                dataset_dir, 
                split="train", 
                verbose=False
            )
            
            # Verify training results
            assert len(trained_models) == 2
            assert 'Bihag' in trained_models
            assert 'Darbari' in trained_models
            
            # Step 2: Save all models
            saved_paths = persistence.save_all_models(trained_models)
            
            # Verify persistence results
            assert len(saved_paths) == 2
            
            for raag in ['Bihag', 'Darbari']:
                model_path, metadata_path = saved_paths[raag]
                assert Path(model_path).exists()
                assert Path(metadata_path).exists()
            
            # Step 3: Load all models back
            loaded_models = persistence.load_all_models()
            
            # Verify loaded models
            assert len(loaded_models) == 2
            
            for raag_key, (model, metadata) in loaded_models.items():
                assert isinstance(model, DiscreteHMM)
                assert model.n_states == 4
                assert model.n_observations == 4
                assert metadata['n_sequences'] == 2
                assert metadata['total_frames'] == 40
            
            # Step 4: Verify model summary
            models_info = persistence.list_available_models()
            assert len(models_info) == 2
            
            for info in models_info:
                assert info['model_exists'] is True
                assert info['metadata_exists'] is True
                assert info['n_sequences'] == 2
                assert info['total_frames'] == 40