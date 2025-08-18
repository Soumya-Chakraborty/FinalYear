#!/usr/bin/env python3
"""
Demonstration script for the multi-raag HMM training pipeline.

This script demonstrates the complete workflow from training multiple
raag models to saving and loading them using the implemented system.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import json

from raag_hmm.train.trainer import RaagTrainer
from raag_hmm.train.persistence import ModelPersistence
from raag_hmm.hmm.model import DiscreteHMM


def create_demo_dataset(dataset_root: str):
    """Create a minimal demo dataset for testing."""
    print("Creating demo dataset...")
    
    dataset_path = Path(dataset_root)
    train_dir = dataset_path / "train"
    audio_dir = train_dir / "audio"
    metadata_dir = train_dir / "metadata"
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Create demo files for 3 raag classes
    raag_classes = ['Bihag', 'Darbari', 'Desh']
    
    for raag in raag_classes:
        for i in range(3):  # 3 files per raag
            # Create dummy audio file
            audio_file = audio_dir / f"{raag}_{i:02d}.wav"
            audio_file.touch()
            
            # Create metadata
            metadata = {
                'recording_id': f"{raag}_{i:02d}",
                'raag': raag,
                'tonic_hz': 220.0 + hash(raag) % 50,  # Different tonic per raag
                'artist': f"Artist_{raag}",
                'instrument': 'sitar',
                'split': 'train'
            }
            
            metadata_file = metadata_dir / f"{raag}_{i:02d}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    print(f"Created demo dataset with {len(raag_classes)} raag classes")
    return raag_classes


def mock_audio_processing(trainer):
    """Mock the audio processing pipeline for demonstration."""
    print("Setting up mock audio processing...")
    
    # Create a simple mock that returns different patterns for each raag
    def mock_extract_sequence(audio_path, tonic_hz, **kwargs):
        # Extract raag name from path
        raag_name = Path(audio_path).stem.split('_')[0]
        file_idx = int(Path(audio_path).stem.split('_')[1])
        
        # Generate different patterns for each raag
        if raag_name == 'Bihag':
            base_pattern = [0, 2, 4, 5, 7, 9, 11]  # Major-like pattern
        elif raag_name == 'Darbari':
            base_pattern = [0, 1, 3, 5, 6, 8, 10]  # Minor-like pattern
        elif raag_name == 'Desh':
            base_pattern = [0, 2, 3, 5, 7, 8, 11]  # Mixed pattern
        else:
            base_pattern = [0, 1, 2, 3, 4, 5, 6]   # Default pattern
        
        # Add some variation based on file index
        pattern = [(note + file_idx) % 12 for note in base_pattern]
        
        # Repeat pattern to create longer sequence
        sequence = np.array(pattern * 8)  # 56 frames
        
        return sequence
    
    # Replace the method
    trainer.extract_and_quantize_sequence = mock_extract_sequence
    print("Mock audio processing configured")


def demonstrate_training_pipeline():
    """Demonstrate the complete training pipeline."""
    print("=" * 60)
    print("RAAG HMM TRAINING PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = Path(temp_dir) / "dataset"
        models_dir = Path(temp_dir) / "models"
        
        # Step 1: Create demo dataset
        print("\n1. DATASET PREPARATION")
        print("-" * 30)
        raag_classes = create_demo_dataset(str(dataset_dir))
        
        # Step 2: Initialize trainer
        print("\n2. TRAINER INITIALIZATION")
        print("-" * 30)
        trainer = RaagTrainer(
            n_states=12,  # 12 chromatic states
            n_observations=12,  # 12 chromatic observations
            max_iterations=10,  # Fewer iterations for demo
            convergence_tolerance=1.0,  # Loose tolerance for demo
            random_state=42  # Reproducible results
        )
        print(f"Trainer initialized: {trainer.n_states} states, {trainer.n_observations} observations")
        
        # Step 3: Mock audio processing
        print("\n3. AUDIO PROCESSING SETUP")
        print("-" * 30)
        mock_audio_processing(trainer)
        
        # Step 4: Train all models
        print("\n4. MODEL TRAINING")
        print("-" * 30)
        trained_models = trainer.train_all_raag_models(
            str(dataset_dir),
            split="train",
            verbose=True
        )
        
        print(f"\nTraining completed for {len(trained_models)} raag classes:")
        for raag_name, (model, metadata) in trained_models.items():
            print(f"  {raag_name}: {metadata['n_sequences']} sequences, "
                  f"{metadata['total_frames']} frames, "
                  f"converged={metadata['converged']}")
        
        # Step 5: Model persistence
        print("\n5. MODEL PERSISTENCE")
        print("-" * 30)
        persistence = ModelPersistence(str(models_dir))
        saved_paths = persistence.save_all_models(trained_models)
        
        print(f"Saved {len(saved_paths)} models:")
        for raag, (model_path, metadata_path) in saved_paths.items():
            model_size = Path(model_path).stat().st_size / 1024  # KB
            print(f"  {raag}: {model_size:.1f} KB")
        
        # Step 6: Model loading and validation
        print("\n6. MODEL LOADING AND VALIDATION")
        print("-" * 30)
        loaded_models = persistence.load_all_models()
        
        print(f"Loaded {len(loaded_models)} models:")
        for raag, (model, metadata) in loaded_models.items():
            # Test model scoring
            test_sequence = np.array([0, 2, 4, 5, 7, 9, 11, 0])
            score = model.score(test_sequence)
            print(f"  {raag}: score={score:.2f}, "
                  f"training_time={metadata.get('training_time', 0):.2f}s")
        
        # Step 7: Model information
        print("\n7. MODEL INFORMATION")
        print("-" * 30)
        models_info = persistence.list_available_models()
        
        for info in models_info:
            print(f"  {info['raag_name']}:")
            print(f"    Sequences: {info.get('n_sequences', 'N/A')}")
            print(f"    Frames: {info.get('total_frames', 'N/A')}")
            print(f"    Converged: {info.get('converged', 'N/A')}")
            print(f"    Size: {info['model_size_mb']:.2f} MB")
        
        # Step 8: Training statistics
        print("\n8. TRAINING STATISTICS")
        print("-" * 30)
        stats = trainer.get_training_summary()
        
        print(f"Total sequences processed: {stats['total_sequences']}")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Total training time: {stats['total_time']:.2f}s")
        print(f"Models converged: {stats['converged_count']}/{len(trained_models)}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return True


def demonstrate_model_comparison():
    """Demonstrate model comparison and scoring."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Create test sequences representing different raag patterns
    test_sequences = {
        'Bihag_pattern': np.array([0, 2, 4, 5, 7, 9, 11, 9, 7, 5, 4, 2, 0]),
        'Darbari_pattern': np.array([0, 1, 3, 5, 6, 8, 10, 8, 6, 5, 3, 1, 0]),
        'Desh_pattern': np.array([0, 2, 3, 5, 7, 8, 11, 8, 7, 5, 3, 2, 0]),
        'Random_pattern': np.array([0, 6, 1, 8, 3, 9, 4, 10, 5, 11, 2, 7, 0])
    }
    
    # Create and train simple models for demonstration
    models = {}
    
    for raag in ['Bihag', 'Darbari', 'Desh']:
        print(f"\nTraining model for {raag}...")
        model = DiscreteHMM(n_states=12, n_observations=12, random_state=42)
        
        # Create training sequences based on raag patterns
        if raag == 'Bihag':
            training_seqs = [test_sequences['Bihag_pattern'] for _ in range(3)]
        elif raag == 'Darbari':
            training_seqs = [test_sequences['Darbari_pattern'] for _ in range(3)]
        else:  # Desh
            training_seqs = [test_sequences['Desh_pattern'] for _ in range(3)]
        
        # Train model
        model.train(training_seqs, max_iterations=5, verbose=False)
        models[raag] = model
    
    # Test each sequence against all models
    print("\nModel Scoring Results:")
    print("-" * 40)
    
    for seq_name, sequence in test_sequences.items():
        print(f"\nTesting {seq_name}:")
        scores = {}
        
        for raag, model in models.items():
            score = model.score(sequence)
            scores[raag] = score
        
        # Find best match
        best_raag = max(scores, key=scores.get)
        
        for raag, score in scores.items():
            marker = " <-- BEST MATCH" if raag == best_raag else ""
            print(f"  {raag}: {score:.2f}{marker}")
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        # Run main demonstration
        success = demonstrate_training_pipeline()
        
        if success:
            # Run model comparison demonstration
            demonstrate_model_comparison()
            
        print("\nAll demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()