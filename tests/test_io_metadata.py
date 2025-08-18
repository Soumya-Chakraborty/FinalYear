"""
Unit tests for metadata parsing and validation functionality.

Tests JSON schema validation, metadata parsing, and error handling.
"""

import pytest
import json
import tempfile
from pathlib import Path

from raag_hmm.io.metadata import (
    load_metadata, MetadataParser, AudioMetadata,
    validate_raag_name, validate_tonic_frequency, create_metadata_template,
    METADATA_SCHEMA
)
from raag_hmm.exceptions import AudioProcessingError


class TestLoadMetadata:
    """Test the load_metadata function."""
    
    def test_load_metadata_valid(self, tmp_path):
        """Test loading valid metadata."""
        metadata = {
            "recording_id": "test_001",
            "raag": "Bihag",
            "tonic_hz": 261.63,
            "artist": "Test Artist",
            "split": "train"
        }
        
        metadata_path = tmp_path / "test.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        loaded = load_metadata(str(metadata_path))
        
        assert loaded["recording_id"] == "test_001"
        assert loaded["raag"] == "Bihag"
        assert loaded["tonic_hz"] == 261.63
        assert loaded["artist"] == "Test Artist"
    
    def test_load_metadata_minimal(self, tmp_path):
        """Test loading metadata with only required fields."""
        metadata = {
            "recording_id": "minimal_001",
            "raag": "Yaman",
            "tonic_hz": 220.0
        }
        
        metadata_path = tmp_path / "minimal.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        loaded = load_metadata(str(metadata_path))
        
        assert loaded["recording_id"] == "minimal_001"
        assert loaded["raag"] == "Yaman"
        assert loaded["tonic_hz"] == 220.0
    
    def test_load_metadata_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(AudioProcessingError, match="Metadata file not found"):
            load_metadata("nonexistent.json")
    
    def test_load_metadata_invalid_json(self, tmp_path):
        """Test error handling for invalid JSON."""
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("{ invalid json }")
        
        with pytest.raises(AudioProcessingError, match="Invalid JSON"):
            load_metadata(str(invalid_path))
    
    def test_load_metadata_missing_required_field(self, tmp_path):
        """Test validation error for missing required fields."""
        metadata = {
            "recording_id": "test_001",
            "raag": "Bihag"
            # Missing tonic_hz
        }
        
        metadata_path = tmp_path / "incomplete.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        with pytest.raises(AudioProcessingError, match="Metadata validation failed"):
            load_metadata(str(metadata_path))
    
    def test_load_metadata_invalid_raag(self, tmp_path):
        """Test validation error for invalid raag."""
        metadata = {
            "recording_id": "test_001",
            "raag": "InvalidRaag",
            "tonic_hz": 261.63
        }
        
        metadata_path = tmp_path / "invalid_raag.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        with pytest.raises(AudioProcessingError, match="Metadata validation failed"):
            load_metadata(str(metadata_path))
    
    def test_load_metadata_invalid_tonic(self, tmp_path):
        """Test validation error for invalid tonic frequency."""
        metadata = {
            "recording_id": "test_001",
            "raag": "Bihag",
            "tonic_hz": 1000.0  # Too high
        }
        
        metadata_path = tmp_path / "invalid_tonic.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        with pytest.raises(AudioProcessingError, match="Metadata validation failed"):
            load_metadata(str(metadata_path))


class TestAudioMetadata:
    """Test the AudioMetadata dataclass."""
    
    def test_audio_metadata_creation(self):
        """Test creating AudioMetadata with valid data."""
        metadata = AudioMetadata(
            recording_id="test_001",
            raag="Bihag",
            tonic_hz=261.63,
            artist="Test Artist"
        )
        
        assert metadata.recording_id == "test_001"
        assert metadata.raag == "Bihag"
        assert metadata.tonic_hz == 261.63
        assert metadata.artist == "Test Artist"
        assert metadata.split == "train"  # Default value
    
    def test_audio_metadata_validation_invalid_raag(self):
        """Test validation error for invalid raag."""
        with pytest.raises(ValueError, match="Invalid raag"):
            AudioMetadata(
                recording_id="test_001",
                raag="InvalidRaag",
                tonic_hz=261.63
            )
    
    def test_audio_metadata_validation_invalid_tonic(self):
        """Test validation error for invalid tonic frequency."""
        with pytest.raises(ValueError, match="Tonic frequency.*out of valid range"):
            AudioMetadata(
                recording_id="test_001",
                raag="Bihag",
                tonic_hz=50.0  # Too low
            )
    
    def test_audio_metadata_validation_invalid_split(self):
        """Test validation error for invalid split."""
        with pytest.raises(ValueError, match="Invalid split"):
            AudioMetadata(
                recording_id="test_001",
                raag="Bihag",
                tonic_hz=261.63,
                split="invalid"
            )
    
    def test_audio_metadata_validation_negative_duration(self):
        """Test validation error for negative duration."""
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            AudioMetadata(
                recording_id="test_001",
                raag="Bihag",
                tonic_hz=261.63,
                duration_sec=-1.0
            )
    
    def test_audio_metadata_to_dict(self):
        """Test conversion to dictionary."""
        metadata = AudioMetadata(
            recording_id="test_001",
            raag="Bihag",
            tonic_hz=261.63,
            artist="Test Artist",
            duration_sec=120.5
        )
        
        result = metadata.to_dict()
        
        assert result["recording_id"] == "test_001"
        assert result["raag"] == "Bihag"
        assert result["tonic_hz"] == 261.63
        assert result["artist"] == "Test Artist"
        assert result["duration_sec"] == 120.5
        assert result["split"] == "train"
    
    def test_audio_metadata_to_dict_minimal(self):
        """Test conversion to dictionary with minimal fields."""
        metadata = AudioMetadata(
            recording_id="test_001",
            raag="Bihag",
            tonic_hz=261.63
        )
        
        result = metadata.to_dict()
        
        # Should only contain required fields and split
        expected_keys = {"recording_id", "raag", "tonic_hz", "split"}
        assert set(result.keys()) == expected_keys


class TestMetadataParser:
    """Test the MetadataParser class."""
    
    def test_metadata_parser_init(self):
        """Test MetadataParser initialization."""
        parser = MetadataParser(strict_validation=False)
        assert parser.strict_validation == False
        assert parser.schema == METADATA_SCHEMA
    
    def test_metadata_parser_parse(self, tmp_path):
        """Test parsing metadata file."""
        metadata = {
            "recording_id": "test_001",
            "raag": "Darbari",
            "tonic_hz": 196.0,
            "instrument": "Sitar"
        }
        
        metadata_path = tmp_path / "test.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        parser = MetadataParser()
        result = parser.parse(str(metadata_path))
        
        assert isinstance(result, AudioMetadata)
        assert result.recording_id == "test_001"
        assert result.raag == "Darbari"
        assert result.tonic_hz == 196.0
        assert result.instrument == "Sitar"
    
    def test_metadata_parser_parse_batch(self, tmp_path):
        """Test batch parsing of multiple files."""
        # Create multiple metadata files
        metadata_files = []
        for i in range(3):
            metadata = {
                "recording_id": f"test_{i:03d}",
                "raag": "Yaman",
                "tonic_hz": 261.63 + i * 10
            }
            
            metadata_path = tmp_path / f"test_{i}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            metadata_files.append(str(metadata_path))
        
        parser = MetadataParser()
        results = parser.parse_batch(metadata_files)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.recording_id == f"test_{i:03d}"
            assert result.raag == "Yaman"
    
    def test_metadata_parser_batch_with_errors_strict(self, tmp_path):
        """Test batch parsing with errors in strict mode."""
        # Create valid and invalid files
        valid_metadata = {
            "recording_id": "valid_001",
            "raag": "Bihag",
            "tonic_hz": 261.63
        }
        
        invalid_metadata = {
            "recording_id": "invalid_001",
            "raag": "InvalidRaag",  # Invalid raag
            "tonic_hz": 261.63
        }
        
        valid_path = tmp_path / "valid.json"
        invalid_path = tmp_path / "invalid.json"
        
        with open(valid_path, 'w') as f:
            json.dump(valid_metadata, f)
        with open(invalid_path, 'w') as f:
            json.dump(invalid_metadata, f)
        
        parser = MetadataParser(strict_validation=True)
        
        with pytest.raises(AudioProcessingError, match="Batch parsing failed"):
            parser.parse_batch([str(valid_path), str(invalid_path)])
    
    def test_metadata_parser_batch_with_errors_lenient(self, tmp_path):
        """Test batch parsing with errors in lenient mode."""
        # Create valid and invalid files
        valid_metadata = {
            "recording_id": "valid_001",
            "raag": "Bihag",
            "tonic_hz": 261.63
        }
        
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("{ invalid json }")
        
        valid_path = tmp_path / "valid.json"
        with open(valid_path, 'w') as f:
            json.dump(valid_metadata, f)
        
        parser = MetadataParser(strict_validation=False)
        results = parser.parse_batch([str(valid_path), str(invalid_path)])
        
        # Should return only valid results
        assert len(results) == 1
        assert results[0].recording_id == "valid_001"
    
    def test_metadata_parser_validate_metadata(self):
        """Test metadata validation method."""
        parser = MetadataParser()
        
        valid_metadata = {
            "recording_id": "test_001",
            "raag": "Bihag",
            "tonic_hz": 261.63
        }
        
        invalid_metadata = {
            "recording_id": "test_001",
            "raag": "InvalidRaag",
            "tonic_hz": 261.63
        }
        
        assert parser.validate_metadata(valid_metadata) == True
        
        with pytest.raises(AudioProcessingError):
            parser.validate_metadata(invalid_metadata)
    
    def test_metadata_parser_custom_schema(self):
        """Test parser with custom schema."""
        custom_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "number"}
            },
            "required": ["id", "value"]
        }
        
        parser = MetadataParser(custom_schema=custom_schema)
        assert parser.schema == custom_schema
    
    def test_metadata_parser_update_schema(self):
        """Test updating parser schema."""
        parser = MetadataParser()
        original_schema = parser.get_schema()
        
        new_schema = {"type": "object", "properties": {}}
        parser.update_schema(new_schema)
        
        assert parser.schema == new_schema
        assert parser.schema != original_schema


class TestValidationFunctions:
    """Test standalone validation functions."""
    
    @pytest.mark.parametrize("raag,expected", [
        ("Bihag", True),
        ("Darbari", True),
        ("Desh", True),
        ("Gaud_Malhar", True),
        ("Yaman", True),
        ("InvalidRaag", False),
        ("bihag", False),  # Case sensitive
        ("", False)
    ])
    def test_validate_raag_name(self, raag, expected):
        """Test raag name validation."""
        assert validate_raag_name(raag) == expected
    
    @pytest.mark.parametrize("tonic_hz,expected", [
        (80.0, True),
        (261.63, True),
        (800.0, True),
        (79.9, False),
        (800.1, False),
        (0.0, False),
        (-100.0, False)
    ])
    def test_validate_tonic_frequency(self, tonic_hz, expected):
        """Test tonic frequency validation."""
        assert validate_tonic_frequency(tonic_hz) == expected


class TestMetadataTemplate:
    """Test metadata template creation."""
    
    def test_create_metadata_template_basic(self):
        """Test creating basic metadata template."""
        template = create_metadata_template(
            recording_id="test_001",
            raag="Bihag",
            tonic_hz=261.63
        )
        
        assert template["recording_id"] == "test_001"
        assert template["raag"] == "Bihag"
        assert template["tonic_hz"] == 261.63
        assert template["split"] == "train"
    
    def test_create_metadata_template_with_optional(self):
        """Test creating template with optional fields."""
        template = create_metadata_template(
            recording_id="test_001",
            raag="Bihag",
            tonic_hz=261.63,
            artist="Test Artist",
            instrument="Sitar",
            split="test",
            duration_sec=120.5
        )
        
        assert template["artist"] == "Test Artist"
        assert template["instrument"] == "Sitar"
        assert template["split"] == "test"
        assert template["duration_sec"] == 120.5
    
    def test_create_metadata_template_invalid_raag(self):
        """Test template creation with invalid raag."""
        with pytest.raises(ValueError, match="Invalid raag name"):
            create_metadata_template(
                recording_id="test_001",
                raag="InvalidRaag",
                tonic_hz=261.63
            )
    
    def test_create_metadata_template_invalid_tonic(self):
        """Test template creation with invalid tonic."""
        with pytest.raises(ValueError, match="Invalid tonic frequency"):
            create_metadata_template(
                recording_id="test_001",
                raag="Bihag",
                tonic_hz=1000.0
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_metadata_file(self, tmp_path):
        """Test handling of empty metadata file."""
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("")
        
        with pytest.raises(AudioProcessingError, match="Invalid JSON"):
            load_metadata(str(empty_path))
    
    def test_metadata_with_extra_fields(self, tmp_path):
        """Test metadata with additional fields (should be allowed)."""
        metadata = {
            "recording_id": "test_001",
            "raag": "Bihag",
            "tonic_hz": 261.63,
            "extra_field": "extra_value",
            "another_field": 123
        }
        
        metadata_path = tmp_path / "extra.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Should not raise error (additionalProperties: True)
        loaded = load_metadata(str(metadata_path))
        assert loaded["extra_field"] == "extra_value"
        assert loaded["another_field"] == 123
    
    def test_unicode_metadata(self, tmp_path):
        """Test metadata with unicode characters."""
        metadata = {
            "recording_id": "test_001",
            "raag": "Bihag",
            "tonic_hz": 261.63,
            "artist": "राग गायक",  # Hindi text
            "notes": "Special characters: àáâãäå"
        }
        
        metadata_path = tmp_path / "unicode.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
        
        loaded = load_metadata(str(metadata_path))
        assert loaded["artist"] == "राग गायक"
        assert loaded["notes"] == "Special characters: àáâãäå"