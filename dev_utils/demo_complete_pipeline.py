#!/usr/bin/env python3
"""
Complete RaagHMM Pipeline Demonstration

This script demonstrates the complete end-to-end workflow of the RaagHMM system:
1. Create sample dataset
2. Prepare dataset
3. Train models
4. Make predictions
5. Evaluate results
"""

import os
import json
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import subprocess
import sys

def create_sample_dataset(base_dir: Path):
    """Create a sample dataset with synthetic audio for demonstration."""
    print("🎵 Creating sample dataset...")
    
    # Create directory structure
    train_dir = base_dir / "dataset" / "train"
    test_dir = base_dir / "dataset" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    # Audio parameters
    sample_rate = 22050
    duration = 2.0  # 2 seconds per file
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Define raag characteristics (simplified for demo)
    raags = {
        "Bihag": {"base_freq": 220, "pattern": [1, 1.125, 1.25, 1.5]},  # A, B, C#, F#
        "Darbari": {"base_freq": 196, "pattern": [1, 1.067, 1.2, 1.33]},  # G, Ab, Bb, C
        "Desh": {"base_freq": 261.63, "pattern": [1, 1.125, 1.33, 1.5]}  # C, D, F, G
    }
    
    files_created = 0
    
    for raag_name, raag_info in raags.items():
        base_freq = raag_info["base_freq"]
        pattern = raag_info["pattern"]
        
        # Create training files (3 per raag)
        for i in range(3):
            # Create melodic pattern
            segment_length = len(t) // len(pattern)
            audio_data = np.zeros(len(t))
            
            for j, freq_mult in enumerate(pattern):
                start_idx = j * segment_length
                end_idx = min((j + 1) * segment_length, len(t))
                segment_t = t[start_idx:end_idx]
                
                # Add some vibrato and variation
                freq = base_freq * freq_mult * (1 + 0.02 * np.sin(2 * np.pi * 5 * segment_t))
                segment_audio = 0.3 * np.sin(2 * np.pi * freq * segment_t)
                
                # Add envelope
                envelope = np.exp(-2 * (segment_t - segment_t[0]))
                segment_audio *= envelope
                
                audio_data[start_idx:end_idx] = segment_audio
            
            # Add some noise for realism
            noise = 0.02 * np.random.randn(len(audio_data))
            audio_data += noise
            
            # Save training file
            audio_file = train_dir / f"{raag_name.lower()}_train_{i}.wav"
            sf.write(str(audio_file), audio_data, sample_rate)
            
            # Create metadata
            metadata = {
                "recording_id": f"{raag_name.lower()}_train_{i}",
                "raag": raag_name,
                "tonic_hz": base_freq,
                "artist": f"Synthetic Artist {i+1}",
                "instrument": "synthetic",
                "split": "train",
                "duration": duration,
                "notes": f"Synthetic {raag_name} pattern for demonstration"
            }
            
            metadata_file = train_dir / f"{raag_name.lower()}_train_{i}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            files_created += 1
            
            # Create test file (1 per raag)
            if i == 0:
                # Create slightly different version for testing
                test_audio = audio_data * 0.8 + 0.01 * np.random.randn(len(audio_data))
                
                audio_file = test_dir / f"{raag_name.lower()}_test.wav"
                sf.write(str(audio_file), test_audio, sample_rate)
                
                metadata["recording_id"] = f"{raag_name.lower()}_test"
                metadata["split"] = "test"
                
                metadata_file = test_dir / f"{raag_name.lower()}_test.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                files_created += 1
    
    print(f"✅ Created {files_created} audio files with metadata")
    return base_dir / "dataset"

def run_cli_command(command: list, description: str):
    """Run a CLI command and display results."""
    print(f"\n🔧 {description}")
    print(f"Command: python -m raag_hmm.cli.main {' '.join(command)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "raag_hmm.cli.main"] + command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Run the complete RaagHMM pipeline demonstration."""
    print("🎼 RaagHMM Complete Pipeline Demonstration")
    print("=" * 50)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create sample dataset
        dataset_dir = create_sample_dataset(temp_path)
        
        # Step 2: Validate dataset
        success = run_cli_command(
            ["dataset", "validate", str(dataset_dir)],
            "Validating dataset structure"
        )
        if not success:
            print("❌ Dataset validation failed, stopping demo")
            return
        
        # Step 3: Extract pitch from a sample file
        sample_audio = dataset_dir / "train" / "bihag_train_0.wav"
        sample_metadata = dataset_dir / "train" / "bihag_train_0.json"
        
        success = run_cli_command(
            ["dataset", "extract-pitch", str(sample_audio), 
             "--metadata", str(sample_metadata), "--quantize"],
            "Extracting pitch from sample audio"
        )
        
        # Step 4: Train models
        models_dir = temp_path / "models"
        success = run_cli_command(
            ["train", "models", str(dataset_dir), str(models_dir),
             "--max-iter", "10", "--tolerance", "1.0"],  # Quick training for demo
            "Training HMM models"
        )
        if not success:
            print("❌ Model training failed, stopping demo")
            return
        
        # Step 5: Make a prediction
        test_audio = dataset_dir / "test" / "bihag_test.wav"
        test_metadata = dataset_dir / "test" / "bihag_test.json"
        
        success = run_cli_command(
            ["predict", "single", str(test_audio), str(models_dir),
             "--metadata", str(test_metadata), "--top-k", "3"],
            "Making prediction on test audio"
        )
        
        # Step 6: Batch prediction
        results_file = temp_path / "batch_results.json"
        success = run_cli_command(
            ["predict", "batch", str(dataset_dir / "test"), str(models_dir),
             str(results_file)],
            "Running batch prediction"
        )
        
        # Step 7: Evaluate models
        eval_dir = temp_path / "evaluation"
        success = run_cli_command(
            ["evaluate", "test", str(dataset_dir), str(models_dir), str(eval_dir),
             "--top-k", "1", "--top-k", "3"],
            "Evaluating model performance"
        )
        
        # Step 8: Show examples and help
        run_cli_command(["examples"], "Showing usage examples")
        
        print("\n" + "=" * 50)
        print("🎉 Complete pipeline demonstration finished!")
        print("\nThe RaagHMM system includes:")
        print("✅ Dataset preparation and validation")
        print("✅ Pitch extraction and quantization")
        print("✅ HMM model training")
        print("✅ Single and batch prediction")
        print("✅ Comprehensive evaluation")
        print("✅ Rich CLI with help and examples")
        
        print(f"\nDemo files were created in: {temp_dir}")
        print("(Note: Temporary directory will be cleaned up automatically)")

if __name__ == "__main__":
    main()