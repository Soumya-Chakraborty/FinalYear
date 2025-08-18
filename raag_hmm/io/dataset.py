"""
Dataset iteration functionality.

This module provides functions and classes for iterating through
datasets with proper file pairing between audio and metadata files.
"""

import os
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, List, Optional, Set
from dataclasses import dataclass

from .audio import AudioLoader
from .metadata import MetadataParser, AudioMetadata
from ..exceptions import AudioProcessingError
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetItem:
    """
    Represents a single item in the dataset with audio and metadata.
    
    Contains paths to both audio and metadata files along with
    parsed metadata for convenient access.
    """
    audio_path: str
    metadata_path: str
    metadata: AudioMetadata
    
    def __post_init__(self):
        """Validate that files exist."""
        if not Path(self.audio_path).exists():
            raise AudioProcessingError(f"Audio file not found: {self.audio_path}")
        if not Path(self.metadata_path).exists():
            raise AudioProcessingError(f"Metadata file not found: {self.metadata_path}")


def iter_dataset(root: str, split: str = "train") -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Iterate through dataset files for specified split.
    
    Yields paired audio and metadata files from the dataset directory.
    Expects directory structure:
    root/
    ├── train/
    │   ├── audio/
    │   │   ├── file1.wav
    │   │   └── file2.wav
    │   └── metadata/
    │       ├── file1.json
    │       └── file2.json
    └── test/
        ├── audio/
        └── metadata/
    
    Args:
        root: Root directory of the dataset
        split: Dataset split ('train', 'test', 'val')
        
    Yields:
        Tuple of (audio_path, metadata_dict) for each matched pair
        
    Raises:
        AudioProcessingError: If dataset structure is invalid or files missing
    """
    try:
        root_path = Path(root)
        if not root_path.exists():
            raise AudioProcessingError(f"Dataset root directory not found: {root}")
        
        split_path = root_path / split
        if not split_path.exists():
            raise AudioProcessingError(f"Split directory not found: {split_path}")
        
        audio_dir = split_path / "audio"
        metadata_dir = split_path / "metadata"
        
        if not audio_dir.exists():
            raise AudioProcessingError(f"Audio directory not found: {audio_dir}")
        if not metadata_dir.exists():
            raise AudioProcessingError(f"Metadata directory not found: {metadata_dir}")
        
        logger.debug(f"Iterating dataset: {root} split={split}")
        
        # Find all audio files
        audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return
        
        logger.debug(f"Found {len(audio_files)} audio files")
        
        # Match with metadata files
        matched_count = 0
        for audio_path in sorted(audio_files):
            # Find corresponding metadata file
            base_name = audio_path.stem
            metadata_path = metadata_dir / f"{base_name}.json"
            
            if metadata_path.exists():
                try:
                    # Load metadata
                    from .metadata import load_metadata
                    metadata = load_metadata(str(metadata_path))
                    
                    yield str(audio_path), metadata
                    matched_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {audio_path}: {e}")
            else:
                logger.warning(f"No metadata found for audio file: {audio_path}")
        
        logger.info(f"Successfully matched {matched_count} audio-metadata pairs")
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Failed to iterate dataset {root}: {str(e)}")


class DatasetIterator:
    """
    Dataset iterator class with advanced filtering and validation options.
    
    Provides more control over dataset iteration with support for
    filtering by raag, validation, and batch processing.
    """
    
    def __init__(self, 
                 root: str,
                 audio_loader: Optional[AudioLoader] = None,
                 metadata_parser: Optional[MetadataParser] = None,
                 validate_files: bool = True,
                 supported_formats: Optional[Set[str]] = None):
        """
        Initialize DatasetIterator with configuration options.
        
        Args:
            root: Root directory of the dataset
            audio_loader: AudioLoader instance (creates default if None)
            metadata_parser: MetadataParser instance (creates default if None)
            validate_files: Whether to validate file existence and format
            supported_formats: Set of supported audio formats (uses default if None)
        """
        self.root = Path(root)
        self.audio_loader = audio_loader or AudioLoader()
        self.metadata_parser = metadata_parser or MetadataParser()
        self.validate_files = validate_files
        self.supported_formats = supported_formats or {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
        
        if not self.root.exists():
            raise AudioProcessingError(f"Dataset root not found: {root}")
        
        logger.debug(f"DatasetIterator initialized for: {root}")
    
    def iter_split(self, split: str, raag_filter: Optional[str] = None) -> Iterator[DatasetItem]:
        """
        Iterate through dataset split with optional raag filtering.
        
        Args:
            split: Dataset split ('train', 'test', 'val')
            raag_filter: Optional raag name to filter by
            
        Yields:
            DatasetItem objects for each matched pair
            
        Raises:
            AudioProcessingError: If split directory structure is invalid
        """
        split_path = self.root / split
        if not split_path.exists():
            raise AudioProcessingError(f"Split directory not found: {split_path}")
        
        audio_dir = split_path / "audio"
        metadata_dir = split_path / "metadata"
        
        self._validate_directory_structure(audio_dir, metadata_dir)
        
        # Find and match files
        audio_files = self._find_audio_files(audio_dir)
        logger.debug(f"Found {len(audio_files)} audio files in {split}")
        
        matched_count = 0
        filtered_count = 0
        
        for audio_path in sorted(audio_files):
            try:
                # Find corresponding metadata
                base_name = audio_path.stem
                metadata_path = metadata_dir / f"{base_name}.json"
                
                if not metadata_path.exists():
                    logger.warning(f"No metadata found for: {audio_path.name}")
                    continue
                
                # Parse metadata
                metadata = self.metadata_parser.parse(str(metadata_path))
                
                # Apply raag filter if specified
                if raag_filter and metadata.raag != raag_filter:
                    filtered_count += 1
                    continue
                
                # Create dataset item
                item = DatasetItem(
                    audio_path=str(audio_path),
                    metadata_path=str(metadata_path),
                    metadata=metadata
                )
                
                yield item
                matched_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
        
        logger.info(f"Split {split}: {matched_count} items yielded, {filtered_count} filtered")
    
    def get_split_info(self, split: str) -> Dict[str, Any]:
        """
        Get information about a dataset split.
        
        Args:
            split: Dataset split name
            
        Returns:
            Dictionary with split statistics
            
        Raises:
            AudioProcessingError: If split directory not found
        """
        split_path = self.root / split
        if not split_path.exists():
            raise AudioProcessingError(f"Split directory not found: {split_path}")
        
        audio_dir = split_path / "audio"
        metadata_dir = split_path / "metadata"
        
        # Count files
        audio_files = self._find_audio_files(audio_dir) if audio_dir.exists() else []
        metadata_files = list(metadata_dir.glob("*.json")) if metadata_dir.exists() else []
        
        # Count by raag
        raag_counts = {}
        matched_pairs = 0
        
        for audio_path in audio_files:
            base_name = audio_path.stem
            metadata_path = metadata_dir / f"{base_name}.json"
            
            if metadata_path.exists():
                try:
                    metadata = self.metadata_parser.parse(str(metadata_path))
                    raag_counts[metadata.raag] = raag_counts.get(metadata.raag, 0) + 1
                    matched_pairs += 1
                except Exception:
                    pass  # Skip invalid metadata
        
        return {
            'split': split,
            'audio_files': len(audio_files),
            'metadata_files': len(metadata_files),
            'matched_pairs': matched_pairs,
            'raag_distribution': raag_counts,
            'audio_directory': str(audio_dir),
            'metadata_directory': str(metadata_dir)
        }
    
    def get_all_splits_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available splits.
        
        Returns:
            Dictionary mapping split names to their info
        """
        splits_info = {}
        
        # Find all split directories
        for split_dir in self.root.iterdir():
            if split_dir.is_dir() and not split_dir.name.startswith('.'):
                try:
                    splits_info[split_dir.name] = self.get_split_info(split_dir.name)
                except AudioProcessingError:
                    logger.warning(f"Invalid split directory: {split_dir}")
        
        return splits_info
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate entire dataset structure and report issues.
        
        Returns:
            Dictionary with validation results and issues found
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'splits': {},
            'total_files': 0,
            'total_matched': 0
        }
        
        try:
            splits_info = self.get_all_splits_info()
            
            for split_name, split_info in splits_info.items():
                validation_results['splits'][split_name] = split_info
                validation_results['total_files'] += split_info['audio_files']
                validation_results['total_matched'] += split_info['matched_pairs']
                
                # Check for issues
                unmatched = split_info['audio_files'] - split_info['matched_pairs']
                if unmatched > 0:
                    validation_results['issues'].append(
                        f"Split {split_name}: {unmatched} audio files without metadata"
                    )
                
                orphaned_metadata = split_info['metadata_files'] - split_info['matched_pairs']
                if orphaned_metadata > 0:
                    validation_results['issues'].append(
                        f"Split {split_name}: {orphaned_metadata} metadata files without audio"
                    )
            
            if validation_results['issues']:
                validation_results['valid'] = False
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Validation failed: {str(e)}")
        
        return validation_results
    
    def _validate_directory_structure(self, audio_dir: Path, metadata_dir: Path):
        """Validate that required directories exist."""
        if not audio_dir.exists():
            raise AudioProcessingError(f"Audio directory not found: {audio_dir}")
        if not metadata_dir.exists():
            raise AudioProcessingError(f"Metadata directory not found: {metadata_dir}")
    
    def _find_audio_files(self, audio_dir: Path) -> List[Path]:
        """Find all supported audio files in directory."""
        audio_files = []
        
        for ext in self.supported_formats:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        # Validate formats if enabled
        if self.validate_files:
            valid_files = []
            for audio_path in audio_files:
                if self.audio_loader.validate_format(str(audio_path)):
                    valid_files.append(audio_path)
                else:
                    logger.warning(f"Unsupported audio format: {audio_path}")
            return valid_files
        
        return audio_files
    
    def create_subset(self, 
                     output_root: str,
                     splits: Dict[str, int],
                     raag_filter: Optional[str] = None,
                     copy_files: bool = False) -> Dict[str, int]:
        """
        Create a subset of the dataset with specified number of files per split.
        
        Args:
            output_root: Output directory for subset
            splits: Dictionary mapping split names to number of files to include
            raag_filter: Optional raag to filter by
            copy_files: Whether to copy files (True) or create symlinks (False)
            
        Returns:
            Dictionary with actual number of files created per split
            
        Raises:
            AudioProcessingError: If subset creation fails
        """
        import shutil
        
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        
        created_counts = {}
        
        for split_name, target_count in splits.items():
            logger.info(f"Creating subset for {split_name}: {target_count} files")
            
            # Create output directories
            split_output = output_path / split_name
            audio_output = split_output / "audio"
            metadata_output = split_output / "metadata"
            
            audio_output.mkdir(parents=True, exist_ok=True)
            metadata_output.mkdir(parents=True, exist_ok=True)
            
            # Collect items from split
            items = list(self.iter_split(split_name, raag_filter=raag_filter))
            
            # Limit to target count
            selected_items = items[:target_count]
            
            # Copy/link files
            for item in selected_items:
                audio_src = Path(item.audio_path)
                metadata_src = Path(item.metadata_path)
                
                audio_dst = audio_output / audio_src.name
                metadata_dst = metadata_output / metadata_src.name
                
                if copy_files:
                    shutil.copy2(audio_src, audio_dst)
                    shutil.copy2(metadata_src, metadata_dst)
                else:
                    audio_dst.symlink_to(audio_src.resolve())
                    metadata_dst.symlink_to(metadata_src.resolve())
            
            created_counts[split_name] = len(selected_items)
            logger.info(f"Created {len(selected_items)} files for {split_name}")
        
        return created_counts


def find_dataset_files(root: str, 
                      split: str,
                      extensions: Optional[Set[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Find all audio and metadata files in a dataset split.
    
    Args:
        root: Dataset root directory
        split: Split name ('train', 'test', 'val')
        extensions: Set of audio extensions to look for
        
    Returns:
        Tuple of (audio_files, metadata_files) lists
        
    Raises:
        AudioProcessingError: If directories not found
    """
    root_path = Path(root)
    split_path = root_path / split
    
    if not split_path.exists():
        raise AudioProcessingError(f"Split directory not found: {split_path}")
    
    audio_dir = split_path / "audio"
    metadata_dir = split_path / "metadata"
    
    if not audio_dir.exists():
        raise AudioProcessingError(f"Audio directory not found: {audio_dir}")
    if not metadata_dir.exists():
        raise AudioProcessingError(f"Metadata directory not found: {metadata_dir}")
    
    # Find audio files
    extensions = extensions or {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(str(p) for p in audio_dir.glob(f"*{ext}"))
    
    # Find metadata files
    metadata_files = [str(p) for p in metadata_dir.glob("*.json")]
    
    return sorted(audio_files), sorted(metadata_files)


def match_audio_metadata_files(audio_files: List[str], 
                              metadata_files: List[str]) -> List[Tuple[str, str]]:
    """
    Match audio files with their corresponding metadata files.
    
    Args:
        audio_files: List of audio file paths
        metadata_files: List of metadata file paths
        
    Returns:
        List of (audio_path, metadata_path) tuples for matched pairs
    """
    # Create mapping from base names to metadata files
    metadata_map = {}
    for metadata_path in metadata_files:
        base_name = Path(metadata_path).stem
        metadata_map[base_name] = metadata_path
    
    # Match audio files
    matched_pairs = []
    for audio_path in audio_files:
        base_name = Path(audio_path).stem
        if base_name in metadata_map:
            matched_pairs.append((audio_path, metadata_map[base_name]))
    
    return matched_pairs