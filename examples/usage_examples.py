"""
Examples of using the RaagHMM system.

This file demonstrates various ways to use the RaagHMM system,
from basic usage to advanced configurations.
"""

import numpy as np
from pathlib import Path
import tempfile

# Training examples
def example_train_models():
    """Example of training models with the RaagTrainer."""
    from raag_hmm.train import RaagTrainer, ModelPersistence
    
    print("Example: Training models")
    print("-" * 40)
    
    # Initialize trainer with custom parameters
    trainer = RaagTrainer(
        n_states=36,
        n_observations=36,
        max_iterations=100,
        convergence_tolerance=0.05,
        regularization_alpha=0.01,
        random_state=42
    )
    
    # Train models (in practice, replace with actual dataset path)
    # trained_models = trainer.train_all_raag_models(
    #     dataset_root="path/to/dataset",
    #     split="train",
    #     verbose=True
    # )
    
    # Save models
    # persistence = ModelPersistence("models/")
    # persistence.save_all_models(trained_models)
    
    print("Training example would use:")
    print("- RaagTrainer for model training")
    print("- ModelPersistence for saving/loading")
    print("- Dataset in expected format")
    print()


def example_predict_audio():
    """Example of predicting raag for audio file."""
    from raag_hmm.infer import RaagClassifier
    from raag_hmm.io import load_audio
    from raag_hmm.pitch import extract_pitch_with_fallback, smooth_pitch
    from raag_hmm.quantize import quantize_sequence
    
    print("Example: Predicting raag for audio")
    print("-" * 40)
    
    # In practice, you would:
    # 1. Load audio
    # audio_data = load_audio("path/to/audio.wav")
    # 
    # 2. Extract pitch
    # f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)
    # 
    # 3. Smooth pitch
    # f0_hz = smooth_pitch(f0_hz, voicing_prob)
    # 
    # 4. Quantize with known tonic
    # tonic_hz = 261.63  # Example: C4
    # quantized_sequence = quantize_sequence(f0_hz, tonic_hz)
    # 
    # 5. Load classifier and predict
    # classifier = RaagClassifier("path/to/models")
    # result = classifier.predict_with_confidence(quantized_sequence)
    # print(f"Predicted raag: {result['predicted_raag']}")
    
    print("Prediction example would use:")
    print("- Audio loading and pitch extraction")
    print("- Tonic normalization and quantization") 
    print("- RaagClassifier for prediction")
    print()


def example_comprehensive_workflow():
    """Example of complete workflow from training to evaluation."""
    print("Example: Complete workflow")
    print("-" * 40)
    
    print("1. Prepare dataset:")
    print("   - Organize audio files with JSON metadata")
    print("   - Validate dataset structure")
    print("   - Resample audio to consistent format")
    
    print("\n2. Train models:")
    print("   - Initialize RaagTrainer")
    print("   - Call train_all_raag_models()")
    print("   - Save trained models")
    
    print("\n3. Make predictions:")
    print("   - Load trained models")
    print("   - Process new audio files")
    print("   - Get raag predictions with confidence")
    
    print("\n4. Evaluate results:")
    print("   - Test on held-out dataset")
    print("   - Compute accuracy metrics")
    print("   - Generate confusion matrices")
    print()


def example_cli_commands():
    """Example CLI commands."""
    print("Example: CLI Commands")
    print("-" * 40)
    
    cli_examples = [
        "# Prepare dataset",
        "raag-hmm dataset prepare /raw/dataset /processed/dataset",
        "",
        "# Train all models",
        "raag-hmm train models /dataset /models",
        "",
        "# Train single model",
        "raag-hmm train single bihag /dataset bihag_model.pkl",
        "",
        "# Predict single audio file",
        "raag-hmm predict single audio.wav /models --tonic 261.63",
        "",
        "# Batch prediction", 
        "raag-hmm predict batch /audio/dir /models results.json",
        "",
        "# Evaluate models",
        "raag-hmm evaluate test /dataset /models /results",
        "",
        "# View system info",
        "raag-hmm info",
        "raag-hmm examples"
    ]
    
    for example in cli_examples:
        if example == "":
            print()
        else:
            print(f"  {example}")
    print()


def example_custom_configuration():
    """Example of custom configuration."""
    from raag_hmm.config import get_config, set_config, update_config
    
    print("Example: Custom Configuration")
    print("-" * 40)
    
    # Get current configuration
    current_sr = get_config('audio', 'sample_rate')
    print(f"Current sample rate: {current_sr}")
    
    # Update configuration
    set_config('hmm', 'max_iterations', 300)
    set_config('pitch', 'voicing_threshold', 0.6)
    
    # Bulk update
    custom_config = {
        'hmm': {
            'n_states': 48,
            'convergence_tolerance': 0.01
        },
        'training': {
            'early_stopping': True
        }
    }
    update_config(custom_config)
    
    print("Configuration updated:")
    print("- HMM max iterations: 300")
    print("- Pitch voicing threshold: 0.6") 
    print("- HMM states: 48")
    print("- Training early stopping: True")
    print()


def example_error_handling():
    """Example of proper error handling."""
    from raag_hmm.exceptions import (
        AudioProcessingError, 
        PitchExtractionError, 
        ModelTrainingError,
        ClassificationError
    )
    
    print("Example: Error Handling")
    print("-" * 40)
    
    try:
        # Operation that might fail
        pass
        # result = some_raag_operation()
    except AudioProcessingError as e:
        print(f"Audio processing failed: {e}")
    except PitchExtractionError as e:
        print(f"Pitch extraction failed: {e}")
    except ModelTrainingError as e:
        print(f"Model training failed: {e}")
    except ClassificationError as e:
        print(f"Classification failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Always handle specific RaagHMM exceptions first")
    print()


def example_evaluation():
    """Example of evaluation workflow."""
    print("Example: Evaluation")
    print("-" * 40)
    
    # This would typically involve:
    # 1. Loading test dataset
    # 2. Making predictions on test set
    # 3. Computing metrics
    # 4. Generating reports
    
    print("Evaluation typically includes:")
    print("- Overall accuracy: percentage of correct predictions")
    print("- Per-class accuracy: accuracy for each raag individually") 
    print("- Confusion matrix: detailed breakdown of predictions")
    print("- Top-K accuracy: accuracy considering top K predictions")
    print("- Confidence analysis: distribution of prediction confidence")
    print()


if __name__ == "__main__":
    print("RaagHMM Usage Examples")
    print("=" * 50)
    print()
    
    example_train_models()
    example_predict_audio()
    example_comprehensive_workflow()
    example_cli_commands()
    example_custom_configuration()
    example_error_handling()
    example_evaluation()
    
    print("For more details, see:")
    print("- Documentation: docs/usage/usage_guide.md")
    print("- API Reference: docs/api/api_reference.md")
    print("- CLI Examples: raag-hmm examples")