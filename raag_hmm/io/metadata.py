"""
Metadata parsing and validation functionality.

This module provides functions and classes for loading and validating
JSON metadata files with schema validation for required fields.
"""

import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..exceptions import AudioProcessingError
from ..logger import get_logger

logger = get_logger(__name__)


# JSON schema for metadata validation
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "recording_id": {
            "type": "string",
            "minLength": 1,
            "description": "Unique identifier for the recording"
        },
        "raag": {
            "type": "string",
            "enum": ["Bihag", "Darbari", "Desh", "Gaud_Malhar", "Yaman"],
            "description": "Raag class name"
        },
        "tonic_hz": {
            "type": "number",
            "minimum": 80.0,
            "maximum": 800.0,
            "description": "Tonic frequency in Hz"
        },
        "artist": {
            "type": "string",
            "description": "Artist name (optional)"
        },
        "instrument": {
            "type": "string",
            "description": "Instrument name (optional)"
        },
        "split": {
            "type": "string",
            "enum": ["train", "test", "val"],
            "default": "train",
            "description": "Dataset split"
        },
        "notes": {
            "type": "string",
            "description": "Additional notes (optional)"
        },
        "duration_sec": {
            "type": "number",
            "minimum": 0.0,
            "description": "Recording duration in seconds (optional)"
        }
    },
    "required": ["recording_id", "raag", "tonic_hz"],
    "additionalProperties": True
}


@dataclass
class AudioMetadata:
    """
    Structured representation of audio metadata.
    
    Contains all required and optional fields for audio recordings
    with proper type annotations and validation.
    """
    recording_id: str
    raag: str
    tonic_hz: float
    artist: Optional[str] = None
    instrument: Optional[str] = None
    split: str = "train"
    notes: Optional[str] = None
    duration_sec: Optional[float] = None
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate metadata fields."""
        # Validate raag
        valid_raags = {"Bihag", "Darbari", "Desh", "Gaud_Malhar", "Yaman"}
        if self.raag not in valid_raags:
            raise ValueError(f"Invalid raag '{self.raag}'. Must be one of: {valid_raags}")
        
        # Validate tonic frequency
        if not (80.0 <= self.tonic_hz <= 800.0):
            raise ValueError(f"Tonic frequency {self.tonic_hz} Hz out of valid range [80, 800] Hz")
        
        # Validate split
        valid_splits = {"train", "test", "val"}
        if self.split not in valid_splits:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of: {valid_splits}")
        
        # Validate duration if provided
        if self.duration_sec is not None and self.duration_sec < 0:
            raise ValueError(f"Duration cannot be negative: {self.duration_sec}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "recording_id": self.recording_id,
            "raag": self.raag,
            "tonic_hz": self.tonic_hz,
            "split": self.split
        }
        
        # Add optional fields if present
        if self.artist is not None:
            result["artist"] = self.artist
        if self.instrument is not None:
            result["instrument"] = self.instrument
        if self.notes is not None:
            result["notes"] = self.notes
        if self.duration_sec is not None:
            result["duration_sec"] = self.duration_sec
            
        return result


def load_metadata(path: str) -> Dict[str, Any]:
    """
    Load and validate metadata from JSON file.
    
    Loads JSON metadata file and validates against schema to ensure
    all required fields are present and properly formatted.
    
    Args:
        path: Path to JSON metadata file
        
    Returns:
        Dictionary containing validated metadata
        
    Raises:
        AudioProcessingError: If file cannot be loaded or validation fails
    """
    try:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Metadata file not found: {path}")
        
        logger.debug(f"Loading metadata from: {path}")
        
        # Load JSON content
        with open(path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Validate against schema
        try:
            jsonschema.validate(metadata, METADATA_SCHEMA)
        except jsonschema.ValidationError as e:
            raise AudioProcessingError(f"Metadata validation failed for {path}: {e.message}")
        
        logger.debug(f"Loaded metadata for recording: {metadata.get('recording_id', 'unknown')}")
        return metadata
        
    except json.JSONDecodeError as e:
        raise AudioProcessingError(f"Invalid JSON in metadata file {path}: {str(e)}")
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Failed to load metadata from {path}: {str(e)}")


class MetadataParser:
    """
    Metadata parsing class with configurable validation and batch processing.
    
    Provides more control over metadata parsing with support for
    custom validation rules and batch processing of multiple files.
    """
    
    def __init__(self, strict_validation: bool = True, custom_schema: Optional[Dict] = None):
        """
        Initialize MetadataParser with validation options.
        
        Args:
            strict_validation: Whether to enforce strict schema validation
            custom_schema: Optional custom JSON schema (uses default if None)
        """
        self.strict_validation = strict_validation
        self.schema = custom_schema or METADATA_SCHEMA
        logger.debug(f"MetadataParser initialized: strict={strict_validation}")
    
    def parse(self, path: str) -> AudioMetadata:
        """
        Parse metadata file into structured AudioMetadata object.
        
        Args:
            path: Path to JSON metadata file
            
        Returns:
            AudioMetadata object with validated fields
            
        Raises:
            AudioProcessingError: If parsing or validation fails
        """
        try:
            # Load raw metadata
            raw_metadata = self._load_raw(path)
            
            # Convert to structured object
            metadata = AudioMetadata(
                recording_id=raw_metadata["recording_id"],
                raag=raw_metadata["raag"],
                tonic_hz=float(raw_metadata["tonic_hz"]),
                artist=raw_metadata.get("artist"),
                instrument=raw_metadata.get("instrument"),
                split=raw_metadata.get("split", "train"),
                notes=raw_metadata.get("notes"),
                duration_sec=raw_metadata.get("duration_sec")
            )
            
            logger.info(f"Parsed metadata for {metadata.recording_id}: {metadata.raag} at {metadata.tonic_hz} Hz")
            return metadata
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, ValueError)):
                raise AudioProcessingError(f"Failed to parse metadata {path}: {str(e)}")
            raise
    
    def parse_batch(self, paths: List[str]) -> List[AudioMetadata]:
        """
        Parse multiple metadata files in batch.
        
        Args:
            paths: List of paths to metadata files
            
        Returns:
            List of AudioMetadata objects
            
        Raises:
            AudioProcessingError: If any file fails to parse (in strict mode)
        """
        results = []
        errors = []
        
        for path in paths:
            try:
                metadata = self.parse(path)
                results.append(metadata)
            except Exception as e:
                error_msg = f"Failed to parse {path}: {str(e)}"
                errors.append(error_msg)
                
                if self.strict_validation:
                    raise AudioProcessingError(f"Batch parsing failed: {error_msg}")
                else:
                    logger.warning(error_msg)
        
        if errors and not self.strict_validation:
            logger.warning(f"Batch parsing completed with {len(errors)} errors out of {len(paths)} files")
        
        logger.info(f"Successfully parsed {len(results)} metadata files")
        return results
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata dictionary against schema.
        
        Args:
            metadata: Dictionary to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            AudioProcessingError: If validation fails in strict mode
        """
        try:
            jsonschema.validate(metadata, self.schema)
            return True
        except jsonschema.ValidationError as e:
            if self.strict_validation:
                raise AudioProcessingError(f"Metadata validation failed: {e.message}")
            logger.warning(f"Metadata validation warning: {e.message}")
            return False
    
    def _load_raw(self, path: str) -> Dict[str, Any]:
        """Load raw JSON metadata without conversion to AudioMetadata."""
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Metadata file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate if strict mode enabled
            if self.strict_validation:
                self.validate_metadata(metadata)
            
            return metadata
            
        except json.JSONDecodeError as e:
            raise AudioProcessingError(f"Invalid JSON in {path}: {str(e)}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get current validation schema."""
        return self.schema.copy()
    
    def update_schema(self, schema: Dict[str, Any]):
        """Update validation schema."""
        self.schema = schema
        logger.debug("Metadata validation schema updated")


def validate_raag_name(raag: str) -> bool:
    """
    Validate raag name against supported classes.
    
    Args:
        raag: Raag name to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_raags = {"Bihag", "Darbari", "Desh", "Gaud_Malhar", "Yaman"}
    return raag in valid_raags


def validate_tonic_frequency(tonic_hz: float) -> bool:
    """
    Validate tonic frequency is within reasonable range.
    
    Args:
        tonic_hz: Tonic frequency in Hz
        
    Returns:
        True if valid, False otherwise
    """
    return 80.0 <= tonic_hz <= 800.0


def create_metadata_template(recording_id: str, raag: str, tonic_hz: float, **kwargs) -> Dict[str, Any]:
    """
    Create metadata template with required fields.
    
    Args:
        recording_id: Unique recording identifier
        raag: Raag class name
        tonic_hz: Tonic frequency in Hz
        **kwargs: Additional optional fields
        
    Returns:
        Dictionary with metadata template
        
    Raises:
        ValueError: If required fields are invalid
    """
    if not validate_raag_name(raag):
        raise ValueError(f"Invalid raag name: {raag}")
    
    if not validate_tonic_frequency(tonic_hz):
        raise ValueError(f"Invalid tonic frequency: {tonic_hz}")
    
    template = {
        "recording_id": recording_id,
        "raag": raag,
        "tonic_hz": tonic_hz,
        "split": kwargs.get("split", "train")
    }
    
    # Add optional fields
    for field in ["artist", "instrument", "notes", "duration_sec"]:
        if field in kwargs:
            template[field] = kwargs[field]
    
    return template