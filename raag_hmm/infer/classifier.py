"""
Raag classification and model loading functionality.

This module implements model loading with validation and caching,
and provides classification using forward algorithm scoring.
"""

import json
import joblib
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from datetime import datetime
import numpy as np

from ..hmm.model import DiscreteHMM
from ..exceptions import ClassificationError
from ..logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Handles model loading, validation, and caching for performance.
    
    Provides functionality to load trained HMM models with error handling
    for corrupted files, metadata validation, and version compatibility checking.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelLoader with models directory.
        
        Args:
            models_dir: Directory containing trained models (default: "models")
        """
        self.models_dir = Path(models_dir)
        self._model_cache = {}  # Cache for loaded models
        self._metadata_cache = {}  # Cache for metadata
        
        logger.debug(f"ModelLoader initialized: {self.models_dir}")
    
    def load_model(self, raag_name: str, use_cache: bool = True) -> Tuple[DiscreteHMM, Dict[str, Any]]:
        """
        Load HMM model and metadata with error handling and validation.
        
        Args:
            raag_name: Name of the raag class
            use_cache: Whether to use cached models (default: True)
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            ClassificationError: If loading fails or validation errors occur
        """
        try:
            # Check cache first
            if use_cache and raag_name in self._model_cache:
                logger.debug(f"Loading model from cache: {raag_name}")
                return self._model_cache[raag_name], self._metadata_cache[raag_name]
            
            # Sanitize raag name for filename
            safe_raag_name = self._sanitize_filename(raag_name)
            
            # Define file paths
            model_path = self.models_dir / f"{safe_raag_name}.pkl"
            metadata_path = self.models_dir / f"{safe_raag_name}_meta.json"
            
            # Check file existence
            if not model_path.exists():
                raise ClassificationError(f"Model file not found: {model_path}")
            if not metadata_path.exists():
                raise ClassificationError(f"Metadata file not found: {metadata_path}")
            
            # Load and validate metadata first
            metadata = self._load_and_validate_metadata(metadata_path)
            
            # Load model with corruption handling
            model = self._load_and_validate_model(model_path, metadata)
            
            # Cache the loaded model and metadata
            if use_cache:
                self._model_cache[raag_name] = model
                self._metadata_cache[raag_name] = metadata
            
            logger.info(f"Successfully loaded model for raag: {raag_name}")
            
            return model, metadata
            
        except Exception as e:
            if isinstance(e, ClassificationError):
                raise
            raise ClassificationError(f"Failed to load model for raag {raag_name}: {str(e)}")
    
    def load_all_models(self, use_cache: bool = True) -> Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]]:
        """
        Load all available models from the models directory.
        
        Args:
            use_cache: Whether to use cached models (default: True)
            
        Returns:
            Dictionary mapping raag names to (model, metadata) tuples
            
        Raises:
            ClassificationError: If no models can be loaded
        """
        try:
            # Find all model files
            model_files = list(self.models_dir.glob("*.pkl"))
            
            if not model_files:
                raise ClassificationError(f"No model files found in {self.models_dir}")
            
            loaded_models = {}
            failed_models = []
            
            logger.info(f"Loading {len(model_files)} models from {self.models_dir}")
            
            for model_file in model_files:
                # Extract raag name from filename
                raag_name = model_file.stem  # Remove .pkl extension
                
                try:
                    model, metadata = self.load_model(raag_name, use_cache=use_cache)
                    loaded_models[raag_name] = (model, metadata)
                except Exception as e:
                    logger.warning(f"Failed to load model {model_file}: {e}")
                    failed_models.append((raag_name, str(e)))
            
            if not loaded_models:
                error_details = "; ".join([f"{name}: {error}" for name, error in failed_models])
                raise ClassificationError(f"No models could be loaded successfully. Errors: {error_details}")
            
            if failed_models:
                logger.warning(f"Failed to load {len(failed_models)} models: {failed_models}")
            
            logger.info(f"Successfully loaded {len(loaded_models)} models")
            
            return loaded_models
            
        except Exception as e:
            if isinstance(e, ClassificationError):
                raise
            raise ClassificationError(f"Failed to load all models: {str(e)}")
    
    def validate_model_compatibility(self, models: Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]]) -> bool:
        """
        Validate that all loaded models are compatible for classification.
        
        Args:
            models: Dictionary of loaded models
            
        Returns:
            True if all models are compatible
            
        Raises:
            ClassificationError: If models are incompatible
        """
        if not models:
            raise ClassificationError("No models provided for compatibility check")
        
        # Get reference model parameters
        reference_raag = next(iter(models.keys()))
        reference_model, reference_metadata = models[reference_raag]
        
        reference_n_states = reference_model.n_states
        reference_n_observations = reference_model.n_observations
        
        # Check all models have same dimensions
        for raag_name, (model, metadata) in models.items():
            if model.n_states != reference_n_states:
                raise ClassificationError(
                    f"Model dimension mismatch: {raag_name} has {model.n_states} states, "
                    f"expected {reference_n_states}"
                )
            
            if model.n_observations != reference_n_observations:
                raise ClassificationError(
                    f"Model dimension mismatch: {raag_name} has {model.n_observations} observations, "
                    f"expected {reference_n_observations}"
                )
            
            # Validate stochastic matrices
            try:
                model.validate_stochastic_matrices()
            except Exception as e:
                raise ClassificationError(f"Model {raag_name} has invalid stochastic matrices: {e}")
        
        logger.debug(f"All {len(models)} models are compatible for classification")
        return True
    
    def clear_cache(self) -> None:
        """Clear the model and metadata cache."""
        self._model_cache.clear()
        self._metadata_cache.clear()
        logger.debug("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached models.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_models': list(self._model_cache.keys()),
            'cache_size': len(self._model_cache),
            'memory_usage_mb': self._estimate_cache_memory_usage()
        }
    
    def _load_and_validate_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """
        Load and validate metadata file.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            Validated metadata dictionary
            
        Raises:
            ClassificationError: If metadata is invalid or corrupted
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ClassificationError(f"Corrupted metadata file {metadata_path}: {e}")
        except Exception as e:
            raise ClassificationError(f"Failed to read metadata file {metadata_path}: {e}")
        
        # Validate required metadata fields
        required_fields = ['model_class', 'model_parameters', 'saved_at']
        for field in required_fields:
            if field not in metadata:
                raise ClassificationError(f"Missing required metadata field: {field}")
        
        # Validate model class
        if metadata['model_class'] != 'DiscreteHMM':
            raise ClassificationError(
                f"Unsupported model class: {metadata['model_class']}, expected DiscreteHMM"
            )
        
        # Validate model parameters
        model_params = metadata['model_parameters']
        if 'n_states' not in model_params or 'n_observations' not in model_params:
            raise ClassificationError("Missing model parameters in metadata")
        
        # Version compatibility check (if version info is available)
        if 'version' in metadata:
            self._check_version_compatibility(metadata['version'])
        
        logger.debug(f"Metadata validated: {metadata_path}")
        return metadata
    
    def _load_and_validate_model(self, model_path: Path, metadata: Dict[str, Any]) -> DiscreteHMM:
        """
        Load and validate model file.
        
        Args:
            model_path: Path to model pickle file
            metadata: Associated metadata for validation
            
        Returns:
            Loaded and validated DiscreteHMM model
            
        Raises:
            ClassificationError: If model is corrupted or invalid
        """
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise ClassificationError(f"Corrupted model file {model_path}: {e}")
        
        # Validate model type
        if not isinstance(model, DiscreteHMM):
            raise ClassificationError(
                f"Loaded object is not a DiscreteHMM: {type(model)}, expected DiscreteHMM"
            )
        
        # Validate model consistency with metadata
        model_params = metadata['model_parameters']
        
        if model.n_states != model_params['n_states']:
            raise ClassificationError(
                f"Model n_states mismatch: file={model.n_states}, "
                f"metadata={model_params['n_states']}"
            )
        
        if model.n_observations != model_params['n_observations']:
            raise ClassificationError(
                f"Model n_observations mismatch: file={model.n_observations}, "
                f"metadata={model_params['n_observations']}"
            )
        
        # Validate stochastic matrices
        try:
            model.validate_stochastic_matrices()
        except Exception as e:
            raise ClassificationError(f"Model has invalid stochastic matrices: {e}")
        
        logger.debug(f"Model validated: {model_path}")
        return model
    
    def _check_version_compatibility(self, model_version: str) -> None:
        """
        Check version compatibility between model and current system.
        
        Args:
            model_version: Version string from model metadata
            
        Raises:
            ClassificationError: If version is incompatible
        """
        # For now, we'll just log the version info
        # In a real system, you might implement semantic version checking
        logger.debug(f"Model version: {model_version}")
        
        # Example version compatibility check:
        # if not self._is_version_compatible(model_version):
        #     raise ClassificationError(f"Incompatible model version: {model_version}")
    
    def _sanitize_filename(self, raag_name: str) -> str:
        """
        Sanitize raag name for use as filename.
        
        Args:
            raag_name: Original raag name
            
        Returns:
            Sanitized filename-safe string
        """
        # Replace spaces and special characters with underscores
        safe_name = raag_name.replace(' ', '_').replace('-', '_')
        
        # Remove any characters that aren't alphanumeric or underscore
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        
        # Convert to lowercase for consistency
        safe_name = safe_name.lower()
        
        return safe_name
    
    def _estimate_cache_memory_usage(self) -> float:
        """
        Estimate memory usage of cached models in MB.
        
        Returns:
            Estimated memory usage in megabytes
        """
        # This is a rough estimate - in practice you might use more sophisticated methods
        total_size = 0
        
        for model in self._model_cache.values():
            # Estimate size of model parameters
            total_size += model.pi.nbytes
            total_size += model.A.nbytes
            total_size += model.B.nbytes
        
        return total_size / (1024 * 1024)  # Convert to MB


class RaagClassifier:
    """
    Multi-model inference engine for raag classification.
    
    Provides functionality to classify unknown audio recordings using
    forward algorithm scoring across all trained raag models.
    """
    
    def __init__(self, models_dir: str = "models", use_cache: bool = True):
        """
        Initialize RaagClassifier with model directory.
        
        Args:
            models_dir: Directory containing trained models (default: "models")
            use_cache: Whether to use model caching (default: True)
        """
        self.models_dir = models_dir
        self.use_cache = use_cache
        self.model_loader = ModelLoader(models_dir)
        self.models = {}  # Will be loaded on first use
        self._is_loaded = False
        
        logger.debug(f"RaagClassifier initialized: {models_dir}")
    
    def load_models(self, force_reload: bool = False) -> None:
        """
        Load all available models for classification.
        
        Args:
            force_reload: Force reloading even if models are already loaded
            
        Raises:
            ClassificationError: If models cannot be loaded
        """
        if self._is_loaded and not force_reload:
            logger.debug("Models already loaded, skipping reload")
            return
        
        try:
            logger.info("Loading models for classification...")
            
            # Load all models
            self.models = self.model_loader.load_all_models(use_cache=self.use_cache)
            
            # Validate compatibility
            self.model_loader.validate_model_compatibility(self.models)
            
            self._is_loaded = True
            
            raag_names = list(self.models.keys())
            logger.info(f"Successfully loaded {len(self.models)} models: {raag_names}")
            
        except Exception as e:
            self._is_loaded = False
            if isinstance(e, ClassificationError):
                raise
            raise ClassificationError(f"Failed to load models: {str(e)}")
    
    def score_sequence(self, sequence: np.ndarray, raag_name: str) -> float:
        """
        Compute log-likelihood score for a sequence using specific raag model.
        
        Args:
            sequence: Quantized pitch sequence [T]
            raag_name: Name of the raag model to use
            
        Returns:
            Log-likelihood score for the sequence
            
        Raises:
            ClassificationError: If scoring fails
        """
        try:
            # Ensure models are loaded
            if not self._is_loaded:
                self.load_models()
            
            # Check if raag model exists
            if raag_name not in self.models:
                available_raags = list(self.models.keys())
                raise ClassificationError(
                    f"Raag model '{raag_name}' not found. Available models: {available_raags}"
                )
            
            # Validate sequence
            sequence = np.array(sequence)
            if len(sequence) == 0:
                raise ClassificationError("Empty sequence provided")
            
            model, metadata = self.models[raag_name]
            
            # Validate sequence values
            if np.any(sequence < 0) or np.any(sequence >= model.n_observations):
                raise ClassificationError(
                    f"Sequence contains invalid observation indices. "
                    f"Expected range [0, {model.n_observations-1}], "
                    f"got range [{sequence.min()}, {sequence.max()}]"
                )
            
            # Compute log-likelihood using forward algorithm
            log_likelihood = model.score(sequence)
            
            logger.debug(f"Scored sequence for {raag_name}: {log_likelihood:.6f}")
            
            return log_likelihood
            
        except Exception as e:
            if isinstance(e, ClassificationError):
                raise
            raise ClassificationError(f"Failed to score sequence for raag {raag_name}: {str(e)}")
    
    def predict_raag(self, sequence: np.ndarray, return_all_scores: bool = False) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Predict raag class using argmax over all model scores.
        
        Args:
            sequence: Quantized pitch sequence [T]
            return_all_scores: Whether to return scores for all classes (default: False)
            
        Returns:
            If return_all_scores is False: predicted raag name
            If return_all_scores is True: tuple of (predicted_raag, all_scores_dict)
            
        Raises:
            ClassificationError: If prediction fails
        """
        try:
            # Ensure models are loaded
            if not self._is_loaded:
                self.load_models()
            
            if not self.models:
                raise ClassificationError("No models available for prediction")
            
            # Validate sequence
            sequence = np.array(sequence)
            if len(sequence) == 0:
                raise ClassificationError("Empty sequence provided for prediction")
            
            # Compute scores for all raag models
            scores = {}
            
            for raag_name in self.models.keys():
                try:
                    score = self.score_sequence(sequence, raag_name)
                    scores[raag_name] = score
                except Exception as e:
                    logger.warning(f"Failed to score sequence for {raag_name}: {e}")
                    scores[raag_name] = float('-inf')  # Assign very low score
            
            if not scores or all(score == float('-inf') for score in scores.values()):
                raise ClassificationError("Failed to compute scores for any raag model")
            
            # Find raag with highest score (argmax)
            predicted_raag = max(scores, key=scores.get)
            
            logger.debug(f"Prediction: {predicted_raag} (score: {scores[predicted_raag]:.6f})")
            logger.debug(f"All scores: {scores}")
            
            if return_all_scores:
                return predicted_raag, scores
            else:
                return predicted_raag
                
        except Exception as e:
            if isinstance(e, ClassificationError):
                raise
            raise ClassificationError(f"Failed to predict raag: {str(e)}")
    
    def predict_with_confidence(self, sequence: np.ndarray, normalize_scores: bool = True) -> Dict[str, Any]:
        """
        Predict raag with confidence scores and ranking for all classes.
        
        Args:
            sequence: Quantized pitch sequence [T]
            normalize_scores: Whether to normalize scores to probabilities (default: True)
            
        Returns:
            Dictionary containing:
            - 'predicted_raag': Most likely raag
            - 'confidence': Confidence score for prediction
            - 'scores': Dictionary of raw log-likelihood scores
            - 'probabilities': Dictionary of normalized probabilities (if normalize_scores=True)
            - 'ranking': List of raags sorted by score (highest first)
            
        Raises:
            ClassificationError: If prediction fails
        """
        try:
            # Get prediction and all scores
            predicted_raag, scores = self.predict_raag(sequence, return_all_scores=True)
            
            # Create ranking (sorted by score, highest first)
            ranking = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            
            # Compute confidence and probabilities
            if normalize_scores:
                # Convert log-likelihoods to probabilities using softmax
                # Subtract max for numerical stability
                max_score = max(scores.values())
                exp_scores = {raag: np.exp(score - max_score) for raag, score in scores.items()}
                total_exp = sum(exp_scores.values())
                probabilities = {raag: exp_score / total_exp for raag, exp_score in exp_scores.items()}
                
                confidence = probabilities[predicted_raag]
            else:
                probabilities = None
                # Use relative score difference as confidence measure
                sorted_scores = sorted(scores.values(), reverse=True)
                if len(sorted_scores) > 1:
                    confidence = sorted_scores[0] - sorted_scores[1]  # Score margin
                else:
                    confidence = sorted_scores[0]
            
            result = {
                'predicted_raag': predicted_raag,
                'confidence': confidence,
                'scores': scores,
                'ranking': ranking
            }
            
            if probabilities is not None:
                result['probabilities'] = probabilities
            
            logger.debug(f"Prediction with confidence: {predicted_raag} (confidence: {confidence:.4f})")
            
            return result
            
        except Exception as e:
            if isinstance(e, ClassificationError):
                raise
            raise ClassificationError(f"Failed to predict raag with confidence: {str(e)}")
    
    def get_available_raags(self) -> List[str]:
        """
        Get list of available raag classes.
        
        Returns:
            List of raag names for which models are available
        """
        if not self._is_loaded:
            try:
                self.load_models()
            except Exception as e:
                logger.warning(f"Failed to load models: {e}")
                return []
        
        return list(self.models.keys())
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary mapping raag names to model information
        """
        if not self._is_loaded:
            try:
                self.load_models()
            except Exception as e:
                logger.warning(f"Failed to load models: {e}")
                return {}
        
        model_info = {}
        
        for raag_name, (model, metadata) in self.models.items():
            model_info[raag_name] = {
                'n_states': model.n_states,
                'n_observations': model.n_observations,
                'n_sequences': metadata.get('n_sequences', 'unknown'),
                'total_frames': metadata.get('total_frames', 'unknown'),
                'converged': metadata.get('converged', 'unknown'),
                'final_log_likelihood': metadata.get('final_log_likelihood', 'unknown'),
                'training_time': metadata.get('training_time', 'unknown'),
                'saved_at': metadata.get('saved_at', 'unknown')
            }
        
        return model_info
    
    def clear_cache(self) -> None:
        """Clear model cache and force reload on next use."""
        self.model_loader.clear_cache()
        self.models.clear()
        self._is_loaded = False
        logger.debug("Classifier cache cleared")