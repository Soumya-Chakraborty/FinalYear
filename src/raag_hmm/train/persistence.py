"""
Model persistence and metadata storage for trained HMM models.

This module handles serialization/deserialization of HMM models using joblib
and manages JSON metadata storage with training statistics and hyperparameters.
"""

import json
import joblib
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import numpy as np

from ..hmm.model import DiscreteHMM
from ..exceptions import ModelTrainingError
from ..logger import get_logger

logger = get_logger(__name__)


class ModelPersistence:
    """
    Handles model serialization, deserialization, and metadata management.
    
    Provides functionality to save and load trained HMM models with their
    associated training metadata in a structured directory format.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelPersistence with target directory.
        
        Args:
            models_dir: Directory to store models and metadata (default: "models")
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"ModelPersistence initialized: {self.models_dir}")
    
    def save_model(self, 
                   raag_name: str, 
                   model: DiscreteHMM, 
                   metadata: Dict[str, Any],
                   overwrite: bool = False) -> Tuple[str, str]:
        """
        Save HMM model and metadata to disk.
        
        Args:
            raag_name: Name of the raag class
            model: Trained DiscreteHMM model
            metadata: Training metadata dictionary
            overwrite: Whether to overwrite existing files (default: False)
            
        Returns:
            Tuple of (model_path, metadata_path) for saved files
            
        Raises:
            ModelTrainingError: If saving fails or files exist without overwrite
        """
        try:
            # Sanitize raag name for filename
            safe_raag_name = self._sanitize_filename(raag_name)
            
            # Define file paths
            model_path = self.models_dir / f"{safe_raag_name}.pkl"
            metadata_path = self.models_dir / f"{safe_raag_name}_meta.json"
            
            # Check for existing files
            if not overwrite:
                if model_path.exists():
                    raise ModelTrainingError(f"Model file already exists: {model_path}")
                if metadata_path.exists():
                    raise ModelTrainingError(f"Metadata file already exists: {metadata_path}")
            
            # Prepare metadata for serialization
            serializable_metadata = self._prepare_metadata_for_serialization(metadata)
            
            # Add save timestamp and file information
            serializable_metadata.update({
                'saved_at': datetime.now().isoformat(),
                'model_file': model_path.name,
                'metadata_file': metadata_path.name,
                'model_class': model.__class__.__name__,
                'model_parameters': {
                    'n_states': model.n_states,
                    'n_observations': model.n_observations
                }
            })
            
            # Save model using joblib
            logger.debug(f"Saving model to: {model_path}")
            joblib.dump(model, model_path, compress=3)
            
            # Save metadata as JSON
            logger.debug(f"Saving metadata to: {metadata_path}")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved model for raag: {raag_name}")
            logger.info(f"  Model: {model_path}")
            logger.info(f"  Metadata: {metadata_path}")
            
            return str(model_path), str(metadata_path)
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to save model for raag {raag_name}: {str(e)}")
    
    def load_model(self, raag_name: str) -> Tuple[DiscreteHMM, Dict[str, Any]]:
        """
        Load HMM model and metadata from disk.
        
        Args:
            raag_name: Name of the raag class
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            ModelTrainingError: If loading fails or files not found
        """
        try:
            # Sanitize raag name for filename
            safe_raag_name = self._sanitize_filename(raag_name)
            
            # Define file paths
            model_path = self.models_dir / f"{safe_raag_name}.pkl"
            metadata_path = self.models_dir / f"{safe_raag_name}_meta.json"
            
            # Check file existence
            if not model_path.exists():
                raise ModelTrainingError(f"Model file not found: {model_path}")
            if not metadata_path.exists():
                raise ModelTrainingError(f"Metadata file not found: {metadata_path}")
            
            # Load model
            logger.debug(f"Loading model from: {model_path}")
            model = joblib.load(model_path)
            
            # Validate model type
            if not isinstance(model, DiscreteHMM):
                raise ModelTrainingError(f"Loaded object is not a DiscreteHMM: {type(model)}")
            
            # Load metadata
            logger.debug(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate model consistency with metadata
            self._validate_model_metadata_consistency(model, metadata)
            
            logger.info(f"Successfully loaded model for raag: {raag_name}")
            
            return model, metadata
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to load model for raag {raag_name}: {str(e)}")
    
    def save_all_models(self, 
                       trained_models: Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]],
                       overwrite: bool = False) -> Dict[str, Tuple[str, str]]:
        """
        Save all trained models and their metadata.
        
        Args:
            trained_models: Dictionary mapping raag names to (model, metadata) tuples
            overwrite: Whether to overwrite existing files (default: False)
            
        Returns:
            Dictionary mapping raag names to (model_path, metadata_path) tuples
            
        Raises:
            ModelTrainingError: If saving any model fails
        """
        try:
            saved_paths = {}
            
            logger.info(f"Saving {len(trained_models)} trained models")
            
            for raag_name, (model, metadata) in trained_models.items():
                model_path, metadata_path = self.save_model(
                    raag_name, model, metadata, overwrite=overwrite
                )
                saved_paths[raag_name] = (model_path, metadata_path)
            
            # Create summary file
            summary_path = self._create_models_summary(trained_models, saved_paths)
            
            logger.info(f"All models saved successfully")
            logger.info(f"Summary created: {summary_path}")
            
            return saved_paths
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to save all models: {str(e)}")
    
    def load_all_models(self) -> Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]]:
        """
        Load all available models from the models directory.
        
        Returns:
            Dictionary mapping raag names to (model, metadata) tuples
            
        Raises:
            ModelTrainingError: If loading fails
        """
        try:
            # Find all model files
            model_files = list(self.models_dir.glob("*.pkl"))
            
            if not model_files:
                raise ModelTrainingError(f"No model files found in {self.models_dir}")
            
            loaded_models = {}
            
            logger.info(f"Loading {len(model_files)} models from {self.models_dir}")
            
            for model_file in model_files:
                # Extract raag name from filename
                raag_name = model_file.stem  # Remove .pkl extension
                
                try:
                    model, metadata = self.load_model(raag_name)
                    loaded_models[raag_name] = (model, metadata)
                except Exception as e:
                    logger.warning(f"Failed to load model {model_file}: {e}")
            
            if not loaded_models:
                raise ModelTrainingError("No models could be loaded successfully")
            
            logger.info(f"Successfully loaded {len(loaded_models)} models")
            
            return loaded_models
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to load all models: {str(e)}")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their basic information.
        
        Returns:
            List of dictionaries with model information
        """
        try:
            model_files = list(self.models_dir.glob("*.pkl"))
            models_info = []
            
            for model_file in sorted(model_files):
                raag_name = model_file.stem
                metadata_file = self.models_dir / f"{raag_name}_meta.json"
                
                info = {
                    'raag_name': raag_name,
                    'model_file': str(model_file),
                    'model_exists': model_file.exists(),
                    'metadata_file': str(metadata_file),
                    'metadata_exists': metadata_file.exists(),
                    'model_size_mb': model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0
                }
                
                # Try to load basic metadata
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        info.update({
                            'n_sequences': metadata.get('n_sequences', 'unknown'),
                            'total_frames': metadata.get('total_frames', 'unknown'),
                            'converged': metadata.get('converged', 'unknown'),
                            'final_log_likelihood': metadata.get('final_log_likelihood', 'unknown'),
                            'training_time': metadata.get('training_time', 'unknown'),
                            'saved_at': metadata.get('saved_at', 'unknown')
                        })
                    except Exception:
                        info['metadata_error'] = True
                
                models_info.append(info)
            
            return models_info
            
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            return []
    
    def delete_model(self, raag_name: str) -> bool:
        """
        Delete model and metadata files for a raag.
        
        Args:
            raag_name: Name of the raag class
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            safe_raag_name = self._sanitize_filename(raag_name)
            
            model_path = self.models_dir / f"{safe_raag_name}.pkl"
            metadata_path = self.models_dir / f"{safe_raag_name}_meta.json"
            
            deleted_files = []
            
            if model_path.exists():
                model_path.unlink()
                deleted_files.append(str(model_path))
            
            if metadata_path.exists():
                metadata_path.unlink()
                deleted_files.append(str(metadata_path))
            
            if deleted_files:
                logger.info(f"Deleted files for raag {raag_name}: {deleted_files}")
                return True
            else:
                logger.warning(f"No files found to delete for raag: {raag_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete model for raag {raag_name}: {e}")
            return False
    
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
    
    def _prepare_metadata_for_serialization(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata dictionary for JSON serialization.
        
        Converts numpy arrays and other non-serializable objects to serializable formats.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Serializable metadata dictionary
        """
        serializable = {}
        
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable[key] = int(value)
            elif isinstance(value, np.floating):
                serializable[key] = float(value)
            elif isinstance(value, dict):
                serializable[key] = self._prepare_metadata_for_serialization(value)
            elif isinstance(value, list):
                serializable[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else
                    int(item) if isinstance(item, np.integer) else
                    float(item) if isinstance(item, np.floating) else
                    item for item in value
                ]
            else:
                serializable[key] = value
        
        return serializable
    
    def _validate_model_metadata_consistency(self, model: DiscreteHMM, metadata: Dict[str, Any]):
        """
        Validate that loaded model is consistent with its metadata.
        
        Args:
            model: Loaded DiscreteHMM model
            metadata: Loaded metadata dictionary
            
        Raises:
            ModelTrainingError: If inconsistencies are found
        """
        # Check model parameters
        model_params = metadata.get('model_parameters', {})
        
        if model_params.get('n_states') != model.n_states:
            raise ModelTrainingError(
                f"Model n_states mismatch: metadata={model_params.get('n_states')}, "
                f"model={model.n_states}"
            )
        
        if model_params.get('n_observations') != model.n_observations:
            raise ModelTrainingError(
                f"Model n_observations mismatch: metadata={model_params.get('n_observations')}, "
                f"model={model.n_observations}"
            )
        
        # Validate stochastic matrices
        try:
            model.validate_stochastic_matrices()
        except Exception as e:
            raise ModelTrainingError(f"Loaded model has invalid stochastic matrices: {e}")
    
    def _create_models_summary(self, 
                              trained_models: Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]],
                              saved_paths: Dict[str, Tuple[str, str]]) -> str:
        """
        Create a summary file with information about all saved models.
        
        Args:
            trained_models: Dictionary of trained models
            saved_paths: Dictionary of saved file paths
            
        Returns:
            Path to created summary file
        """
        summary_path = self.models_dir / "models_summary.json"
        
        summary = {
            'created_at': datetime.now().isoformat(),
            'total_models': len(trained_models),
            'models_directory': str(self.models_dir),
            'models': {}
        }
        
        for raag_name, (model, metadata) in trained_models.items():
            model_path, metadata_path = saved_paths[raag_name]
            
            summary['models'][raag_name] = {
                'model_file': Path(model_path).name,
                'metadata_file': Path(metadata_path).name,
                'n_sequences': metadata.get('n_sequences'),
                'total_frames': metadata.get('total_frames'),
                'converged': metadata.get('converged'),
                'final_log_likelihood': metadata.get('final_log_likelihood'),
                'training_time': metadata.get('training_time'),
                'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
            }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(summary_path)