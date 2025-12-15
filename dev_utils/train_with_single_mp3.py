#!/usr/bin/env python3
"""
Train RaagHMM with a single MP3 file (demonstration).

This script shows how to work with your MP3 file and explains what you need for proper training.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import librosa

def get_mp3_info(mp3_file):
    """Get basic information about the MP3 file."""
    try:
        duration = librosa.get_duration(path=str(mp3_file))
        sr = librosa.get_samplerate(str(mp3_file))
        return duration, sr
    except Exception as e:
        print(f"Error reading MP3: {e}")
        return None, None

def create_demo_dataset(mp3_file, raag_name, tonic_hz):
    """Create a demo dataset structure with your MP3 file."""
    mp3_path = Path(mp3_file)
    
    if not mp3_path.exists():
        print(f"❌ MP3 file not found: {mp3_file}")
        return None
    
    # Create dataset structure
    dataset_dir = Path("demo_dataset")
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    
    # Clean up existing dataset
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    # Get MP3 info
    duration, sample_rate = get_mp3_info(mp3_path)
    
    # Copy MP3 to train directory
    train_mp3 = train_dir / f"{raag_name.lower()}_demo.mp3"
    shutil.copy2(mp3_path, train_mp3)
    
    # Create metadata for training file
    train_metadata = {
        "recording_id": f"{raag_name.lower()}_demo",
        "raag": raag_name,
        "tonic_hz": tonic_hz,
        "artist": "Demo Artist",
        "instrument": "unknown",
        "split": "train",
        "notes": f"Demo file from {mp3_path.name}"
    }
    
    if duration:
        train_metadata["duration"] = round(duration, 2)
    
    # Save training metadata
    train_metadata_file = train_dir / f"{raag_name.lower()}_demo.json"
    with open(train_metadata_file, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    # For demo purposes, create a duplicate as test file
    # (In real training, you'd use different recordings)
    test_mp3 = test_dir / f"{raag_name.lower()}_test.mp3"
    shutil.copy2(mp3_path, test_mp3)
    
    test_metadata = train_metadata.copy()
    test_metadata["recording_id"] = f"{raag_name.lower()}_test"
    test_metadata["split"] = "test"
    test_metadata["notes"] = f"Test file (duplicate of {mp3_path.name})"
    
    test_metadata_file = test_dir / f"{raag_name.lower()}_test.json"
    with open(test_metadata_file, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"✅ Created demo dataset in: {dataset_dir}")
    print(f"   - Training file: {train_mp3}")
    print(f"   - Test file: {test_mp3}")
    
    if duration:
        print(f"   - Duration: {duration:.1f} seconds")
    
    return dataset_dir

def run_raag_command(cmd_list, description):
    """Run a RaagHMM CLI command."""
    print(f"\n🔧 {description}")
    print(f"Command: python -m raag_hmm.cli.main {' '.join(cmd_list)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "raag_hmm.cli.main"] + cmd_list,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function for single MP3 training demo."""
    print("🎵 RaagHMM Single MP3 Training Demo")
    print("=" * 40)
    
    # Get MP3 file path
    mp3_file = input("Enter path to your MP3 file: ").strip()
    
    if not mp3_file:
        print("❌ No MP3 file provided")
        return
    
    if not Path(mp3_file).exists():
        print(f"❌ File not found: {mp3_file}")
        return
    
    # Get raag information
    print("\nAvailable raags:")
    raags = ["Bihag", "Darbari", "Desh", "Gaud_Malhar", "Yaman"]
    for i, raag in enumerate(raags, 1):
        print(f"  {i}. {raag}")
    
    while True:
        try:
            choice = int(input("Select raag (1-5): "))
            if 1 <= choice <= 5:
                raag_name = raags[choice - 1]
                break
            else:
                print("Please select 1-5")
        except ValueError:
            print("Please enter a number")
    
    # Get tonic frequency
    print("\nCommon tonic frequencies:")
    tonics = {
        "C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23,
        "G": 392.00, "A": 440.00, "B": 493.88
    }
    
    for note, freq in tonics.items():
        print(f"  {note}: {freq} Hz")
    
    while True:
        tonic_input = input("Enter tonic frequency (Hz) or note: ").strip()
        try:
            if tonic_input.upper() in tonics:
                tonic_hz = tonics[tonic_input.upper()]
                break
            else:
                tonic_hz = float(tonic_input)
                if 80 <= tonic_hz <= 800:
                    break
                else:
                    print("Frequency should be between 80-800 Hz")
        except ValueError:
            print("Please enter a valid frequency or note")
    
    print(f"\n📋 Summary:")
    print(f"   MP3 file: {mp3_file}")
    print(f"   Raag: {raag_name}")
    print(f"   Tonic: {tonic_hz} Hz")
    
    # Create demo dataset
    print(f"\n📁 Creating demo dataset...")
    dataset_dir = create_demo_dataset(mp3_file, raag_name, tonic_hz)
    
    if not dataset_dir:
        return
    
    # Validate dataset
    print(f"\n✅ Validating dataset...")
    if not run_raag_command(["dataset", "validate", str(dataset_dir)], "Dataset validation"):
        return
    
    # Extract pitch (demonstration)
    print(f"\n🎼 Extracting pitch (demo)...")
    sample_file = dataset_dir / "train" / f"{raag_name.lower()}_demo.mp3"
    sample_metadata = dataset_dir / "train" / f"{raag_name.lower()}_demo.json"
    
    run_raag_command([
        "dataset", "extract-pitch", 
        str(sample_file),
        "--metadata", str(sample_metadata),
        "--quantize"
    ], "Pitch extraction demo")
    
    # Important note about training
    print(f"\n⚠️  IMPORTANT NOTE ABOUT TRAINING:")
    print(f"   A single MP3 file is NOT enough for proper model training!")
    print(f"   You need:")
    print(f"   - At least 2-3 different raags")
    print(f"   - At least 3-5 recordings per raag")
    print(f"   - Different artists/performances for variety")
    
    # Ask if user wants to proceed anyway
    proceed = input(f"\nProceed with demo training anyway? (y/N): ").strip().lower()
    
    if proceed == 'y':
        print(f"\n🚀 Training demo model...")
        models_dir = "demo_models"
        
        # This will likely fail or produce poor results with just one file
        success = run_raag_command([
            "train", "models",
            str(dataset_dir),
            models_dir,
            "--max-iter", "5",  # Very few iterations for demo
            "--tolerance", "1.0"
        ], "Demo model training")
        
        if success:
            print(f"\n🎯 Testing prediction...")
            test_file = dataset_dir / "test" / f"{raag_name.lower()}_test.mp3"
            test_metadata = dataset_dir / "test" / f"{raag_name.lower()}_test.json"
            
            run_raag_command([
                "predict", "single",
                str(test_file),
                models_dir,
                "--metadata", str(test_metadata)
            ], "Demo prediction")
    
    # Provide guidance for real training
    print(f"\n🎓 For Real Training:")
    print(f"1. Collect more MP3 files (at least 10-15 total)")
    print(f"2. Organize them by raag in train/ and test/ directories")
    print(f"3. Run: python create_metadata.py")
    print(f"4. Run: python train_with_my_data.py")
    print(f"")
    print(f"📚 Example dataset structure:")
    print(f"my_dataset/")
    print(f"├── train/")
    print(f"│   ├── bihag_song1.mp3")
    print(f"│   ├── bihag_song1.json")
    print(f"│   ├── bihag_song2.mp3")
    print(f"│   ├── bihag_song2.json")
    print(f"│   ├── darbari_song1.mp3")
    print(f"│   ├── darbari_song1.json")
    print(f"│   └── ...")
    print(f"└── test/")
    print(f"    ├── bihag_test.mp3")
    print(f"    ├── bihag_test.json")
    print(f"    └── ...")

if __name__ == "__main__":
    main()