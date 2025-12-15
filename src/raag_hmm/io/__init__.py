"""
Audio I/O and dataset management module.

Handles audio file loading, metadata parsing, and dataset iteration.
"""

from .audio import load_audio, AudioLoader
from .metadata import load_metadata, MetadataParser
from .dataset import iter_dataset, DatasetIterator

__all__ = [
    "load_audio",
    "AudioLoader", 
    "load_metadata",
    "MetadataParser",
    "iter_dataset",
    "DatasetIterator"
]