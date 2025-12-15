#!/usr/bin/env python3
"""
Complete workflow for training RaagHMM with your own MP3 files.

This script guides you through the entire process from data preparation to model training.
"""

import subprocess
import sys
import json
from pathlib import Path
import shutil

def run_raag_command(cmd_list, description, timeout=300):
    """Run a RaagHMM CLI command."""
    print(f"\n🔧 {description}")
    print(f"Command: python -m raag_hmm.cli.main {' '.join(cmd_list)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "raag_hmm.cli.main"] + cmd_list,
            timeout=timeout,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def check_dataset_structure(dataset_dir):
    """Check if dataset has the required structure."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return False
    
    # Count files
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
    train_audio = []
    train_metadata = []
    
    for ext in audio_extensions:
        train_audio.extend(train_dir.glob(f"*{ext}"))
    
    train_metadata = list(train_dir.glob("*.json"))
    
    print(f"📊 Dataset Statistics:")
    print(f"  Train audio files: {len(train_audio)}")
    print(f"  Train metadata files: {len(train_metadata)}")
    
    if test_dir.exists():
        test_audio = []
        for ext in audio_extensions:
            test_audio.extend(test_dir.glob(f"*{ext}"))
        test_metadata = list(test_dir.glob("*.json"))
        print(f"  Test audio files: {len(test_audio)}")
        print(f"  Test metadata files: {len(test_metadata)}")
    
    if len(train_audio) == 0:
        print("❌ No audio files found in train directory")
        return False
    
    if len(train_metadata) == 0:
        print("❌ No metadata files found in train directory")
        return False
    
    # Check for raag distribution
    raag_counts = {}
    for metadata_file in train_metadata:
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                raag = metadata.get('raag', 'Unknown')
                raag_counts[raag] = raag_counts.get(raag, 0) + 1
        except:
            continue
    
    print(f"\n📈 Raag Distribution:")
    for raag, count in raag_counts.items():
        print(f"  {raag}: {count} files")
    
    # Check minimum requirements
    if len(raag_counts) < 2:
        print("⚠️  Warning: You need at least 2 different raags for meaningful training")
    
    min_files_per_raag = min(raag_counts.values()) if raag_counts else 0
    if min_files_per_raag < 3:
        print("⚠️  Warning: Recommend at least 3 files per raag for better training")
    
    return True

def main():
    """Main training workflow."""
    print("🎼 RaagHMM Training with Your MP3 Files")
    print("=" * 50)
    
    # Step 1: Get dataset directory
    print("\n📁 Step 1: Dataset Location")
    dataset_dir = input("Enter path to your dataset directory: ").strip()
    
    if not dataset_dir:
        print("❌ No dataset directory provided")
        return
    
    # Step 2: Check dataset structure
    print("\n🔍 Step 2: Checking Dataset Structure")
    if not check_dataset_structure(dataset_dir):
        print("\n💡 To fix dataset issues:")
        print("1. Organize files into train/ and test/ directories")
        print("2. Run: python create_metadata.py")
        print("3. Ensure each audio file has a corresponding .json metadata file")
        return
    
    # Step 3: Validate dataset
    print("\n✅ Step 3: Validating Dataset")
    if not run_raag_command(["dataset", "validate", dataset_dir], "Dataset validation"):
        print("❌ Dataset validation failed. Please fix the issues and try again.")
        return
    
    # Step 4: Get output directory for models
    print("\n📂 Step 4: Model Output Directory")
    models_dir = input("Enter directory to save trained models [./models]: ").strip() or "./models"
    
    # Create models directory if it doesn't exist
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 5: Training parameters
    print("\n⚙️  Step 5: Training Parameters")
    print("Default parameters work well for most cases.")
    
    use_defaults = input("Use default training parameters? (Y/n): ").strip().lower()
    
    if use_defaults in ['n', 'no']:
        max_iter = input("Maximum iterations [200]: ").strip() or "200"
        tolerance = input("Convergence tolerance [0.1]: ").strip() or "0.1"
        states = input("Number of HMM states [36]: ").strip() or "36"
    else:
        max_iter = "200"
        tolerance = "0.1" 
        states = "36"
    
    # Step 6: Train models
    print(f"\n🚀 Step 6: Training Models")
    print("This may take several minutes depending on your dataset size...")
    
    training_cmd = [
        "train", "models",
        dataset_dir,
        models_dir,
        "--max-iter", max_iter,
        "--tolerance", tolerance,
        "--states", states
    ]
    
    if not run_raag_command(training_cmd, "Model training", timeout=1800):  # 30 min timeout
        print("❌ Model training failed")
        return
    
    # Step 7: Test prediction (if test data exists)
    test_dir = Path(dataset_dir) / "test"
    if test_dir.exists():
        test_files = list(test_dir.glob("*.mp3")) + list(test_dir.glob("*.wav"))
        if test_files:
            print(f"\n🎯 Step 7: Testing Prediction")
            test_file = test_files[0]  # Use first test file
            
            # Check if metadata exists
            metadata_file = test_file.with_suffix('.json')
            if metadata_file.exists():
                run_raag_command([
                    "predict", "single",
                    str(test_file),
                    models_dir,
                    "--metadata", str(metadata_file),
                    "--top-k", "3"
                ], f"Test prediction on {test_file.name}")
            else:
                print(f"⚠️  No metadata found for {test_file.name}, skipping test prediction")
    
    # Step 8: Evaluation (if test data exists)
    if test_dir.exists():
        print(f"\n📊 Step 8: Model Evaluation")
        eval_dir = Path(models_dir).parent / "evaluation_results"
        
        run_raag_command([
            "evaluate", "test",
            dataset_dir,
            models_dir,
            str(eval_dir)
        ], "Model evaluation")
    
    # Step 9: Success summary
    print(f"\n🎉 Training Complete!")
    print(f"✅ Models saved in: {models_dir}")
    
    if Path(models_dir).exists():
        model_files = list(Path(models_dir).glob("*.pkl"))
        print(f"✅ Trained models: {len(model_files)}")
        for model_file in model_files:
            print(f"   - {model_file.name}")
    
    print(f"\n🎵 Next Steps:")
    print(f"1. Test your models:")
    print(f"   python -m raag_hmm.cli.main predict single your_audio.mp3 {models_dir} --tonic 261.63")
    print(f"")
    print(f"2. Batch prediction:")
    print(f"   python -m raag_hmm.cli.main predict batch audio_folder/ {models_dir} results.json")
    print(f"")
    print(f"3. Get help:")
    print(f"   python -m raag_hmm.cli.main examples")

if __name__ == "__main__":
    main()