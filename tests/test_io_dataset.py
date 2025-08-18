"""
Integration tests for dataset iteration functionality.

Tests dataset structure validation, file pairing, and iteration.
"""

import pytest
import json
import numpy as np
import soundfile as sf
from pathlib import Path

from raag_hmm.io.dataset import (
    iter_dataset, DatasetIterator, DatasetItem,
    find_dataset_files, match_audio_metadata_files
)
from raag_hmm.io.audio import AudioLoader
from raag_hmm.io.metadata import MetadataParser
from raag_hmm.exceptions import AudioProcessingError


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample dataset structure for testing."""
    # Create directory structure
    train_audio = tmp_path / "train" / "audio"
    train_metadata = tmp_path / "train" / "metadata"
    test_audio = tmp_path / "test" / "audio"
    test_metadata = tmp_path / "test" / "metadata"
    
    train_audio.mkdir(parents=True)
    train_metadata.mkdir(parents=True)
    test_audio.mkdir(parents=True)
    test_metadata.mkdir(parents=True)
    
    # Create sample audio files
    sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 1 second
    
    # Train split files
    train_files = [
        ("bihag_001", "Bihag", 261.63),
        ("bihag_002", "Bihag", 293.66),
        ("yaman_001", "Yaman", 220.0),
    ]
    
    for file_id, raag, tonic in train_files:
        # Audio file
        audio_path = train_audio / f"{file_id}.wav"
        sf.write(audio_path, sample_audio, 22050)
        
        # Metadata file
        metadata = {
            "recording_id": file_id,
            "raag": raag,
            "tonic_hz": tonic,
            "split": "train",
            "artist": "Test Artist"
        }
        metadata_path = train_metadata / f"{file_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    # Test split files
    test_files = [
        ("bihag_test_001", "Bihag", 246.94),
        ("yaman_test_001", "Yaman", 196.0),
    ]
    
    for file_id, raag, tonic in test_files:
        # Audio file
        audio_path = test_audio / f"{file_id}.wav"
        sf.write(audio_path, sample_audio, 22050)
        
        # Metadata file
        metadata = {
            "recording_id": file_id,
            "raag": raag,
            "tonic_hz": tonic,
            "split": "test"
        }
        metadata_path = test_metadata / f"{file_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    # Create orphaned files (audio without metadata, metadata without audio)
    orphaned_audio = train_audio / "orphaned.wav"
    sf.write(orphaned_audio, sample_audio, 22050)
    
    orphaned_metadata = {
        "recording_id": "orphaned_meta",
        "raag": "Bihag",
        "tonic_hz": 261.63,
        "split": "train"
    }
    orphaned_meta_path = train_metadata / "orphaned_meta.json"
    with open(orphaned_meta_path, 'w') as f:
        json.dump(orphaned_metadata, f)
    
    return tmp_path


class TestIterDataset:
    """Test the iter_dataset function."""
    
    def test_iter_dataset_train_split(self, sample_dataset):
        """Test iterating through train split."""
        items = list(iter_dataset(str(sample_dataset), "train"))
        
        assert len(items) == 3  # 3 matched pairs in train
        
        # Check first item
        audio_path, metadata = items[0]
        assert Path(audio_path).exists()
        assert metadata["recording_id"] == "bihag_001"
        assert metadata["raag"] == "Bihag"
        assert metadata["tonic_hz"] == 261.63
    
    def test_iter_dataset_test_split(self, sample_dataset):
        """Test iterating through test split."""
        items = list(iter_dataset(str(sample_dataset), "test"))
        
        assert len(items) == 2  # 2 matched pairs in test
        
        # Verify all items have required fields
        for audio_path, metadata in items:
            assert Path(audio_path).exists()
            assert "recording_id" in metadata
            assert "raag" in metadata
            assert "tonic_hz" in metadata
    
    def test_iter_dataset_nonexistent_root(self):
        """Test error handling for nonexistent root directory."""
        with pytest.raises(AudioProcessingError, match="Dataset root directory not found"):
            list(iter_dataset("/nonexistent/path", "train"))
    
    def test_iter_dataset_nonexistent_split(self, sample_dataset):
        """Test error handling for nonexistent split."""
        with pytest.raises(AudioProcessingError, match="Split directory not found"):
            list(iter_dataset(str(sample_dataset), "nonexistent"))
    
    def test_iter_dataset_missing_audio_dir(self, tmp_path):
        """Test error handling for missing audio directory."""
        # Create split with only metadata directory
        split_path = tmp_path / "train"
        metadata_dir = split_path / "metadata"
        metadata_dir.mkdir(parents=True)
        
        with pytest.raises(AudioProcessingError, match="Audio directory not found"):
            list(iter_dataset(str(tmp_path), "train"))
    
    def test_iter_dataset_missing_metadata_dir(self, tmp_path):
        """Test error handling for missing metadata directory."""
        # Create split with only audio directory
        split_path = tmp_path / "train"
        audio_dir = split_path / "audio"
        audio_dir.mkdir(parents=True)
        
        with pytest.raises(AudioProcessingError, match="Metadata directory not found"):
            list(iter_dataset(str(tmp_path), "train"))
    
    def test_iter_dataset_empty_directories(self, tmp_path):
        """Test handling of empty directories."""
        # Create empty split directories
        split_path = tmp_path / "train"
        audio_dir = split_path / "audio"
        metadata_dir = split_path / "metadata"
        audio_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        
        items = list(iter_dataset(str(tmp_path), "train"))
        assert len(items) == 0


class TestDatasetIterator:
    """Test the DatasetIterator class."""
    
    def test_dataset_iterator_init(self, sample_dataset):
        """Test DatasetIterator initialization."""
        iterator = DatasetIterator(str(sample_dataset))
        
        assert iterator.root == Path(sample_dataset)
        assert isinstance(iterator.audio_loader, AudioLoader)
        assert isinstance(iterator.metadata_parser, MetadataParser)
        assert iterator.validate_files == True
    
    def test_dataset_iterator_init_nonexistent(self):
        """Test error handling for nonexistent dataset root."""
        with pytest.raises(AudioProcessingError, match="Dataset root not found"):
            DatasetIterator("/nonexistent/path")
    
    def test_iter_split_basic(self, sample_dataset):
        """Test basic split iteration."""
        iterator = DatasetIterator(str(sample_dataset))
        items = list(iterator.iter_split("train"))
        
        assert len(items) == 3
        
        for item in items:
            assert isinstance(item, DatasetItem)
            assert Path(item.audio_path).exists()
            assert Path(item.metadata_path).exists()
            assert item.metadata.recording_id is not None
    
    def test_iter_split_with_raag_filter(self, sample_dataset):
        """Test split iteration with raag filtering."""
        iterator = DatasetIterator(str(sample_dataset))
        
        # Filter for Bihag only
        bihag_items = list(iterator.iter_split("train", raag_filter="Bihag"))
        assert len(bihag_items) == 2  # 2 Bihag recordings in train
        
        for item in bihag_items:
            assert item.metadata.raag == "Bihag"
        
        # Filter for Yaman only
        yaman_items = list(iterator.iter_split("train", raag_filter="Yaman"))
        assert len(yaman_items) == 1  # 1 Yaman recording in train
        assert yaman_items[0].metadata.raag == "Yaman"
    
    def test_iter_split_nonexistent(self, sample_dataset):
        """Test error handling for nonexistent split."""
        iterator = DatasetIterator(str(sample_dataset))
        
        with pytest.raises(AudioProcessingError, match="Split directory not found"):
            list(iterator.iter_split("nonexistent"))
    
    def test_get_split_info(self, sample_dataset):
        """Test getting split information."""
        iterator = DatasetIterator(str(sample_dataset))
        
        train_info = iterator.get_split_info("train")
        
        assert train_info["split"] == "train"
        assert train_info["audio_files"] == 4  # 3 matched + 1 orphaned
        assert train_info["metadata_files"] == 4  # 3 matched + 1 orphaned
        assert train_info["matched_pairs"] == 3
        assert "Bihag" in train_info["raag_distribution"]
        assert "Yaman" in train_info["raag_distribution"]
        assert train_info["raag_distribution"]["Bihag"] == 2
        assert train_info["raag_distribution"]["Yaman"] == 1
    
    def test_get_all_splits_info(self, sample_dataset):
        """Test getting information for all splits."""
        iterator = DatasetIterator(str(sample_dataset))
        
        all_info = iterator.get_all_splits_info()
        
        assert "train" in all_info
        assert "test" in all_info
        
        assert all_info["train"]["matched_pairs"] == 3
        assert all_info["test"]["matched_pairs"] == 2
    
    def test_validate_dataset(self, sample_dataset):
        """Test dataset validation."""
        iterator = DatasetIterator(str(sample_dataset))
        
        validation = iterator.validate_dataset()
        
        assert validation["valid"] == False  # Has orphaned files
        assert len(validation["issues"]) > 0  # Should report orphaned files
        assert validation["total_matched"] == 5  # 3 train + 2 test
        assert "train" in validation["splits"]
        assert "test" in validation["splits"]
    
    def test_validate_dataset_clean(self, tmp_path):
        """Test validation of clean dataset (no orphaned files)."""
        # Create clean dataset
        train_audio = tmp_path / "train" / "audio"
        train_metadata = tmp_path / "train" / "metadata"
        train_audio.mkdir(parents=True)
        train_metadata.mkdir(parents=True)
        
        # Create matched pair
        sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        audio_path = train_audio / "test_001.wav"
        sf.write(audio_path, sample_audio, 22050)
        
        metadata = {
            "recording_id": "test_001",
            "raag": "Bihag",
            "tonic_hz": 261.63,
            "split": "train"
        }
        metadata_path = train_metadata / "test_001.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        iterator = DatasetIterator(str(tmp_path))
        validation = iterator.validate_dataset()
        
        assert validation["valid"] == True
        assert len(validation["issues"]) == 0


class TestDatasetItem:
    """Test the DatasetItem dataclass."""
    
    def test_dataset_item_creation(self, sample_dataset):
        """Test creating DatasetItem with valid files."""
        audio_path = sample_dataset / "train" / "audio" / "bihag_001.wav"
        metadata_path = sample_dataset / "train" / "metadata" / "bihag_001.json"
        
        # Load metadata
        parser = MetadataParser()
        metadata = parser.parse(str(metadata_path))
        
        item = DatasetItem(
            audio_path=str(audio_path),
            metadata_path=str(metadata_path),
            metadata=metadata
        )
        
        assert item.audio_path == str(audio_path)
        assert item.metadata_path == str(metadata_path)
        assert item.metadata.recording_id == "bihag_001"
    
    def test_dataset_item_missing_audio(self, sample_dataset):
        """Test error handling for missing audio file."""
        metadata_path = sample_dataset / "train" / "metadata" / "bihag_001.json"
        
        parser = MetadataParser()
        metadata = parser.parse(str(metadata_path))
        
        with pytest.raises(AudioProcessingError, match="Audio file not found"):
            DatasetItem(
                audio_path="/nonexistent/audio.wav",
                metadata_path=str(metadata_path),
                metadata=metadata
            )
    
    def test_dataset_item_missing_metadata(self, sample_dataset):
        """Test error handling for missing metadata file."""
        audio_path = sample_dataset / "train" / "audio" / "bihag_001.wav"
        
        parser = MetadataParser()
        metadata = parser.parse(str(sample_dataset / "train" / "metadata" / "bihag_001.json"))
        
        with pytest.raises(AudioProcessingError, match="Metadata file not found"):
            DatasetItem(
                audio_path=str(audio_path),
                metadata_path="/nonexistent/metadata.json",
                metadata=metadata
            )


class TestUtilityFunctions:
    """Test utility functions for dataset handling."""
    
    def test_find_dataset_files(self, sample_dataset):
        """Test finding dataset files."""
        audio_files, metadata_files = find_dataset_files(str(sample_dataset), "train")
        
        assert len(audio_files) == 4  # 3 matched + 1 orphaned
        assert len(metadata_files) == 4  # 3 matched + 1 orphaned
        
        # Check file extensions
        for audio_file in audio_files:
            assert Path(audio_file).suffix == ".wav"
        
        for metadata_file in metadata_files:
            assert Path(metadata_file).suffix == ".json"
    
    def test_find_dataset_files_custom_extensions(self, sample_dataset):
        """Test finding files with custom extensions."""
        # Should find no files with .flac extension
        audio_files, metadata_files = find_dataset_files(
            str(sample_dataset), "train", extensions={".flac"}
        )
        
        assert len(audio_files) == 0
        assert len(metadata_files) == 4  # Metadata files still found
    
    def test_find_dataset_files_nonexistent_split(self, sample_dataset):
        """Test error handling for nonexistent split."""
        with pytest.raises(AudioProcessingError, match="Split directory not found"):
            find_dataset_files(str(sample_dataset), "nonexistent")
    
    def test_match_audio_metadata_files(self, sample_dataset):
        """Test matching audio and metadata files."""
        audio_files, metadata_files = find_dataset_files(str(sample_dataset), "train")
        
        matched_pairs = match_audio_metadata_files(audio_files, metadata_files)
        
        assert len(matched_pairs) == 3  # 3 matched pairs
        
        for audio_path, metadata_path in matched_pairs:
            audio_base = Path(audio_path).stem
            metadata_base = Path(metadata_path).stem
            assert audio_base == metadata_base
    
    def test_match_audio_metadata_files_no_matches(self):
        """Test matching with no common base names."""
        audio_files = ["/path/audio1.wav", "/path/audio2.wav"]
        metadata_files = ["/path/meta1.json", "/path/meta2.json"]
        
        matched_pairs = match_audio_metadata_files(audio_files, metadata_files)
        
        assert len(matched_pairs) == 0


class TestDatasetSubsetCreation:
    """Test dataset subset creation functionality."""
    
    def test_create_subset_basic(self, sample_dataset, tmp_path):
        """Test creating a basic dataset subset."""
        iterator = DatasetIterator(str(sample_dataset))
        
        output_dir = tmp_path / "subset"
        splits = {"train": 2, "test": 1}
        
        created_counts = iterator.create_subset(
            str(output_dir), splits, copy_files=True
        )
        
        assert created_counts["train"] == 2
        assert created_counts["test"] == 1
        
        # Verify files were created
        train_audio = output_dir / "train" / "audio"
        train_metadata = output_dir / "train" / "metadata"
        
        assert len(list(train_audio.glob("*.wav"))) == 2
        assert len(list(train_metadata.glob("*.json"))) == 2
    
    def test_create_subset_with_raag_filter(self, sample_dataset, tmp_path):
        """Test creating subset with raag filtering."""
        iterator = DatasetIterator(str(sample_dataset))
        
        output_dir = tmp_path / "bihag_subset"
        splits = {"train": 5}  # Request more than available
        
        created_counts = iterator.create_subset(
            str(output_dir), splits, raag_filter="Bihag", copy_files=True
        )
        
        # Should only get 2 Bihag files from train
        assert created_counts["train"] == 2
        
        # Verify all files are Bihag
        subset_iterator = DatasetIterator(str(output_dir))
        items = list(subset_iterator.iter_split("train"))
        
        for item in items:
            assert item.metadata.raag == "Bihag"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_dataset_structure(self, tmp_path):
        """Test handling of invalid dataset structure."""
        # Create directory without proper structure
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        
        iterator = DatasetIterator(str(invalid_dir))
        
        with pytest.raises(AudioProcessingError):
            list(iterator.iter_split("train"))
    
    def test_corrupted_metadata_files(self, tmp_path):
        """Test handling of corrupted metadata files."""
        # Create dataset with corrupted metadata
        train_audio = tmp_path / "train" / "audio"
        train_metadata = tmp_path / "train" / "metadata"
        train_audio.mkdir(parents=True)
        train_metadata.mkdir(parents=True)
        
        # Create audio file
        sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        audio_path = train_audio / "test.wav"
        sf.write(audio_path, sample_audio, 22050)
        
        # Create corrupted metadata
        metadata_path = train_metadata / "test.json"
        metadata_path.write_text("{ invalid json }")
        
        iterator = DatasetIterator(str(tmp_path))
        
        # Should handle gracefully and skip corrupted files
        items = list(iterator.iter_split("train"))
        assert len(items) == 0  # No valid items
    
    def test_mixed_audio_formats(self, tmp_path):
        """Test handling of mixed audio formats."""
        # Create dataset with different audio formats
        train_audio = tmp_path / "train" / "audio"
        train_metadata = tmp_path / "train" / "metadata"
        train_audio.mkdir(parents=True)
        train_metadata.mkdir(parents=True)
        
        # Create audio files with different extensions
        sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        
        for i, ext in enumerate([".wav", ".flac"]):
            try:
                audio_path = train_audio / f"test_{i}{ext}"
                if ext == ".wav":
                    sf.write(audio_path, sample_audio, 22050, format="WAV")
                elif ext == ".flac":
                    sf.write(audio_path, sample_audio, 22050, format="FLAC")
                
                # Create corresponding metadata
                metadata = {
                    "recording_id": f"test_{i}",
                    "raag": "Bihag",
                    "tonic_hz": 261.63,
                    "split": "train"
                }
                metadata_path = train_metadata / f"test_{i}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
                    
            except Exception:
                # Skip if format not supported by soundfile
                continue
        
        iterator = DatasetIterator(str(tmp_path))
        items = list(iterator.iter_split("train"))
        
        # Should handle all supported formats
        assert len(items) >= 1