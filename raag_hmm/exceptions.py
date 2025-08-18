"""
Exception hierarchy for RaagHMM system.
"""


class RaagHMMError(Exception):
    """Base exception for RaagHMM system."""
    pass


class AudioProcessingError(RaagHMMError):
    """Audio loading or processing failures."""
    pass


class PitchExtractionError(RaagHMMError):
    """Pitch detection algorithm failures."""
    pass


class QuantizationError(RaagHMMError):
    """Feature quantization issues."""
    pass


class ModelTrainingError(RaagHMMError):
    """HMM training convergence or numerical issues."""
    pass


class ClassificationError(RaagHMMError):
    """Inference and prediction failures."""
    pass


class MetadataValidationError(RaagHMMError):
    """Metadata parsing and validation errors."""
    pass


class AudioFormatError(AudioProcessingError):
    """Unsupported audio format errors."""
    pass