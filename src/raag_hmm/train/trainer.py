"""
RaagTrainer for multi-class HMM training pipeline.

This module implements the training system for multiple raag classes,
handling sequence grouping, per-raag model training, and batch processing.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from ..hmm.model import DiscreteHMM
from ..io.dataset import DatasetIterator
from ..io.audio import load_audio
from ..pitch.extractor import extract_pitch_praat, extract_pitch_librosa
from ..pitch.smoother import smooth_pitch
from ..quantize.sequence import quantize_sequence
from ..exceptions import ModelTrainingError, AudioProcessingError
from ..logger import get_logger

logger = get_logger(__name__)


class RaagTrainer:
    """
    Multi-class HMM trainer for raag classification.
    
    Handles the complete training pipeline from dataset loading through
    model training for all raag classes in the dataset.
    """
    
    def __init__(self, 
                 n_states: int = 36,
                 n_observations: int = 36,
                 max_iterations: int = 200,
                 convergence_tolerance: float = 0.1,
                 regularization_alpha: float = 0.01,
                 probability_floor: float = 1e-8,
                 random_state: Optional[int] = None):
        """
        Initialize RaagTrainer with HMM configuration.
        
        Args:
            n_states: Number of HMM hidden states (default: 36)
            n_observations: Number of observation symbols (default: 36)
            max_iterations: Maximum Baum-Welch iterations (default: 200)
            convergence_tolerance: Log-likelihood improvement threshold (default: 0.1)
            regularization_alpha: Dirichlet regularization parameter (default: 0.01)
            probability_floor: Minimum probability value (default: 1e-8)
            random_state: Random seed for reproducible initialization
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.regularization_alpha = regularization_alpha
        self.probability_floor = probability_floor
        self.random_state = random_state
        
        # Training statistics
        self.training_stats = {}
        
        logger.debug(f"RaagTrainer initialized: {n_states} states, {n_observations} observations")
    
    def extract_and_quantize_sequence(self, 
                                    audio_path: str, 
                                    tonic_hz: float,
                                    frame_sec: float = 0.0464,
                                    hop_sec: float = 0.01) -> np.ndarray:
        """
        Extract pitch from audio and quantize to chromatic sequence.
        
        Args:
            audio_path: Path to audio file
            tonic_hz: Tonic frequency for normalization
            frame_sec: Frame size in seconds (default: 0.0464)
            hop_sec: Hop size in seconds (default: 0.01)
            
        Returns:
            Quantized chromatic sequence as integer array
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        try:
            # Load audio
            y = load_audio(audio_path, sr=22050)
            
            # Extract pitch using Praat (primary method)
            try:
                f0_hz, voicing_prob = extract_pitch_praat(y, sr=22050, 
                                                        frame_sec=frame_sec, 
                                                        hop_sec=hop_sec)
            except Exception as e:
                logger.warning(f"Praat extraction failed for {audio_path}: {e}")
                # Fallback to librosa
                f0_hz, voicing_prob = extract_pitch_librosa(y, sr=22050, method='pyin')
            
            # Apply smoothing
            f0_smoothed = smooth_pitch(f0_hz, voicing_prob)
            
            # Remove unvoiced frames (f0 = 0)
            voiced_mask = f0_smoothed > 0
            if not np.any(voiced_mask):
                raise AudioProcessingError(f"No voiced frames found in {audio_path}")
            
            f0_voiced = f0_smoothed[voiced_mask]
            
            # Quantize sequence
            quantized = quantize_sequence(f0_voiced, tonic_hz)
            
            logger.debug(f"Extracted sequence from {audio_path}: {len(quantized)} frames")
            return quantized
            
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            raise AudioProcessingError(f"Failed to process {audio_path}: {str(e)}")
    
    def group_sequences_by_raag(self, 
                               dataset_root: str, 
                               split: str = "train") -> Dict[str, List[np.ndarray]]:
        """
        Load dataset and group quantized sequences by raag class.
        
        Args:
            dataset_root: Root directory of the dataset
            split: Dataset split to use (default: "train")
            
        Returns:
            Dictionary mapping raag names to lists of quantized sequences
            
        Raises:
            ModelTrainingError: If dataset loading or processing fails
        """
        try:
            # Initialize dataset iterator
            dataset_iter = DatasetIterator(dataset_root)
            
            # Group sequences by raag
            raag_sequences = defaultdict(list)
            processed_count = 0
            error_count = 0
            
            logger.info(f"Loading sequences from {dataset_root} split={split}")
            
            for item in dataset_iter.iter_split(split):
                try:
                    # Extract and quantize sequence
                    sequence = self.extract_and_quantize_sequence(
                        item.audio_path, 
                        item.metadata.tonic_hz
                    )
                    
                    # Skip very short sequences
                    if len(sequence) < 10:
                        logger.warning(f"Skipping short sequence: {item.audio_path} ({len(sequence)} frames)")
                        continue
                    
                    # Add to raag group
                    raag_sequences[item.metadata.raag].append(sequence)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logger.debug(f"Processed {processed_count} sequences")
                    
                except Exception as e:
                    logger.error(f"Failed to process {item.audio_path}: {e}")
                    error_count += 1
            
            # Convert to regular dict and log statistics
            raag_sequences = dict(raag_sequences)
            
            logger.info(f"Sequence grouping completed:")
            logger.info(f"  Processed: {processed_count} sequences")
            logger.info(f"  Errors: {error_count} sequences")
            logger.info(f"  Raag classes: {len(raag_sequences)}")
            
            for raag, sequences in raag_sequences.items():
                total_frames = sum(len(seq) for seq in sequences)
                logger.info(f"  {raag}: {len(sequences)} sequences, {total_frames} total frames")
            
            if not raag_sequences:
                raise ModelTrainingError("No valid sequences found in dataset")
            
            return raag_sequences
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to group sequences by raag: {str(e)}")
    
    def train_raag_model(self, 
                        raag_name: str, 
                        sequences: List[np.ndarray],
                        verbose: bool = False) -> Tuple[DiscreteHMM, Dict[str, Any]]:
        """
        Train HMM model for a single raag class.
        
        Args:
            raag_name: Name of the raag class
            sequences: List of quantized sequences for this raag
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (trained_model, training_metadata)
            
        Raises:
            ModelTrainingError: If training fails
        """
        try:
            if not sequences:
                raise ModelTrainingError(f"No sequences provided for raag {raag_name}")
            
            logger.info(f"Training HMM for raag: {raag_name}")
            logger.info(f"  Sequences: {len(sequences)}")
            
            total_frames = sum(len(seq) for seq in sequences)
            logger.info(f"  Total frames: {total_frames}")
            
            # Initialize HMM model
            model = DiscreteHMM(
                n_states=self.n_states,
                n_observations=self.n_observations,
                random_state=self.random_state
            )
            
            # Record training start time
            start_time = time.time()
            
            # Train model using Baum-Welch
            training_stats = model.train(
                observations_list=sequences,
                max_iterations=self.max_iterations,
                convergence_tolerance=self.convergence_tolerance,
                regularization_alpha=self.regularization_alpha,
                probability_floor=self.probability_floor,
                verbose=verbose
            )
            
            # Record training time
            training_time = time.time() - start_time
            
            # Create training metadata
            metadata = {
                'raag_name': raag_name,
                'n_sequences': len(sequences),
                'total_frames': total_frames,
                'convergence_iterations': training_stats['iterations'],
                'final_log_likelihood': training_stats['final_log_likelihood'],
                'converged': training_stats['converged'],
                'training_time': training_time,
                'hyperparameters': {
                    'n_states': self.n_states,
                    'n_observations': self.n_observations,
                    'max_iterations': self.max_iterations,
                    'convergence_tolerance': self.convergence_tolerance,
                    'regularization_alpha': self.regularization_alpha,
                    'probability_floor': self.probability_floor,
                    'random_state': self.random_state
                },
                'training_stats': training_stats
            }
            
            logger.info(f"Training completed for {raag_name}:")
            logger.info(f"  Converged: {training_stats['converged']}")
            logger.info(f"  Iterations: {training_stats['iterations']}")
            logger.info(f"  Final log-likelihood: {training_stats['final_log_likelihood']:.6f}")
            logger.info(f"  Training time: {training_time:.2f}s")
            
            return model, metadata
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to train model for raag {raag_name}: {str(e)}")
    
    def train_all_raag_models(self, 
                             dataset_root: str,
                             split: str = "train",
                             verbose: bool = False) -> Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]]:
        """
        Train HMM models for all raag classes in the dataset.
        
        This is the main function for batch training that implements the complete
        pipeline from dataset loading to model training for all raag classes.
        
        Args:
            dataset_root: Root directory of the dataset
            split: Dataset split to use for training (default: "train")
            verbose: Whether to print detailed training progress
            
        Returns:
            Dictionary mapping raag names to (model, metadata) tuples
            
        Raises:
            ModelTrainingError: If training pipeline fails
        """
        try:
            logger.info(f"Starting multi-raag training pipeline")
            logger.info(f"Dataset: {dataset_root}")
            logger.info(f"Split: {split}")
            
            # Step 1: Group sequences by raag class
            raag_sequences = self.group_sequences_by_raag(dataset_root, split)
            
            # Validate that we have the expected raag classes
            expected_raags = {'Bihag', 'Darbari', 'Desh', 'Gaud_Malhar', 'Yaman'}
            found_raags = set(raag_sequences.keys())
            
            logger.info(f"Expected raags: {expected_raags}")
            logger.info(f"Found raags: {found_raags}")
            
            missing_raags = expected_raags - found_raags
            if missing_raags:
                logger.warning(f"Missing raag classes: {missing_raags}")
            
            extra_raags = found_raags - expected_raags
            if extra_raags:
                logger.info(f"Additional raag classes found: {extra_raags}")
            
            # Step 2: Train separate HMM for each raag class
            trained_models = {}
            training_summary = {}
            
            for raag_name in sorted(raag_sequences.keys()):
                sequences = raag_sequences[raag_name]
                
                logger.info(f"\n{'='*50}")
                logger.info(f"Training raag: {raag_name}")
                logger.info(f"{'='*50}")
                
                # Train model for this raag
                model, metadata = self.train_raag_model(
                    raag_name, 
                    sequences, 
                    verbose=verbose
                )
                
                # Store results
                trained_models[raag_name] = (model, metadata)
                training_summary[raag_name] = {
                    'n_sequences': metadata['n_sequences'],
                    'total_frames': metadata['total_frames'],
                    'converged': metadata['converged'],
                    'iterations': metadata['convergence_iterations'],
                    'final_log_likelihood': metadata['final_log_likelihood'],
                    'training_time': metadata['training_time']
                }
            
            # Step 3: Log overall training summary
            logger.info(f"\n{'='*50}")
            logger.info(f"TRAINING SUMMARY")
            logger.info(f"{'='*50}")
            
            total_sequences = sum(s['n_sequences'] for s in training_summary.values())
            total_frames = sum(s['total_frames'] for s in training_summary.values())
            total_time = sum(s['training_time'] for s in training_summary.values())
            converged_count = sum(1 for s in training_summary.values() if s['converged'])
            
            logger.info(f"Total raag classes trained: {len(trained_models)}")
            logger.info(f"Total sequences processed: {total_sequences}")
            logger.info(f"Total frames processed: {total_frames}")
            logger.info(f"Total training time: {total_time:.2f}s")
            logger.info(f"Models converged: {converged_count}/{len(trained_models)}")
            
            # Store training statistics for later access
            self.training_stats = {
                'raag_summary': training_summary,
                'total_sequences': total_sequences,
                'total_frames': total_frames,
                'total_time': total_time,
                'converged_count': converged_count,
                'dataset_root': dataset_root,
                'split': split
            }
            
            logger.info(f"Multi-raag training pipeline completed successfully")
            
            return trained_models
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Multi-raag training pipeline failed: {str(e)}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of the last training run.
        
        Returns:
            Dictionary with training statistics and summary
        """
        return self.training_stats.copy()