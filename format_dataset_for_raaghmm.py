#!/usr/bin/env python3
"""
Dataset formatter for RaagHMM system.

This script takes audio files organized by raga and creates the proper
dataset structure with metadata files required by RaagHMM.

Usage:
1. Organize your audio files by raga in a source directory
2. Run this script to create the RaagHMM-compatible structure
3. Fill in metadata including tonic frequencies

Example source structure:
source_dir/
├── Bihag/
│   ├── song1.wav
│   ├── song2.mp3
│   └── ...
├── Darbari/
│   ├── song1.wav
│   └── ...
└── ...

This will be converted to:
target_dir/
├── train/
│   ├── audio/
│   └── metadata/
├── test/
│   ├── audio/
│   └── metadata/
└── val/
    ├── audio/
    └── metadata/
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


SUPPORTED_RAGAS = {"Bihag", "Darbari", "Desh", "Gaud_Malhar", "Yaman"}


def validate_raga_name(raag: str) -> bool:
    """Validate if the raga name is supported by RaagHMM."""
    return raag in SUPPORTED_RAGAS


def identify_tonic_interactive(audio_path: str, raag: str) -> float:
    """
    Interactive function to identify tonic frequency.
    In practice, this might use audio analysis tools or expert input.
    """
    print(f"\nAnalyzing audio file: {audio_path}")
    print(f"Raga: {raag}")
    print("Please identify the tonic (Sa) frequency in Hz.")
    print("You can use tools like Sonic Visualiser, Praat, or a tuner app.")
    
    # Provide some guidance
    if 'Bihag' == raag:
        print("Bihag is often performed in C (261.63 Hz) or nearby frequencies.")
    elif 'Darbari' == raag:
        print("Darbari is often performed in A (220 Hz) or G (196 Hz) ranges.")
    elif 'Desh' == raag:
        print("Desh is often performed in D (293.66 Hz) or C (261.63 Hz) ranges.")
    elif 'Gaud_Malhar' == raag:
        print("Gaud Malhar is often in G (196 Hz) or C (261.63 Hz) ranges.")
    elif 'Yaman' == raag:
        print("Yaman is often performed in C (261.63 Hz) or F (174.61 Hz) ranges.")
    
    while True:
        try:
            tonic_hz = float(input("Enter tonic frequency in Hz: "))
            if 80.0 <= tonic_hz <= 800.0:
                return tonic_hz
            else:
                print("Frequency should be between 80 and 800 Hz. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def create_metadata_file(output_path: str, recording_id: str, raag: str, tonic_hz: float, 
                        artist: str = "", instrument: str = "", split: str = "train", 
                        duration_sec: float = 0.0, notes: str = ""):
    """
    Create a metadata JSON file for RaagHMM.
    
    Args:
        output_path: Path for the metadata file
        recording_id: Unique identifier for the recording
        raag: Raag name
        tonic_hz: Tonic frequency in Hz
        artist: Artist name (optional)
        instrument: Instrument name (optional)
        split: Dataset split (train/test/val)
        duration_sec: Duration in seconds (optional)
        notes: Additional notes (optional)
    """
    metadata = {
        "recording_id": recording_id,
        "raag": raag,
        "tonic_hz": tonic_hz,
        "split": split
    }
    
    if artist:
        metadata["artist"] = artist
    if instrument:
        metadata["instrument"] = instrument
    if duration_sec:
        metadata["duration_sec"] = duration_sec
    if notes:
        metadata["notes"] = notes
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Created metadata: {output_path}")


def get_audio_files_by_raga(source_dir: str) -> Dict[str, List[str]]:
    """
    Get all audio files organized by raga directory.
    
    Args:
        source_dir: Source directory with raga subdirectories
        
    Returns:
        Dictionary mapping raga names to lists of audio file paths
    """
    raga_files = defaultdict(list)
    
    for raga_dir in Path(source_dir).iterdir():
        if raga_dir.is_dir():
            raga_name = raga_dir.name
            
            # Validate raga name
            if not validate_raga_name(raga_name):
                print(f"Warning: Skipping unsupported raga directory: {raga_name}")
                continue
            
            # Get all audio files in the raga directory
            for file_path in raga_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}:
                    raga_files[raga_name].append(str(file_path))
    
    return dict(raga_files)


def distribute_files_to_splits(file_list: List[str], train_ratio: float = 0.7, 
                              test_ratio: float = 0.2, val_ratio: float = 0.1) -> Dict[str, List[str]]:
    """
    Distribute files among train, test, and validation splits.
    
    Args:
        file_list: List of file paths
        train_ratio: Proportion for training set
        test_ratio: Proportion for test set
        val_ratio: Proportion for validation set
        
    Returns:
        Dictionary mapping split names to lists of file paths
    """
    import random
    # Set seed for reproducibility
    random.seed(42)
    
    # Shuffle the list
    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)
    
    # Calculate split points
    n_total = len(shuffled_files)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    
    splits = {
        "train": shuffled_files[:n_train],
        "test": shuffled_files[n_train:n_train+n_test],
        "val": shuffled_files[n_train+n_test:]
    }
    
    return splits


def format_dataset(source_dir: str, target_dir: str, interactive: bool = True, 
                   train_ratio: float = 0.7, test_ratio: float = 0.2, val_ratio: float = 0.1):
    """
    Format dataset for RaagHMM system.
    
    Args:
        source_dir: Source directory with raga subdirectories
        target_dir: Target directory for formatted dataset
        interactive: Whether to ask for tonic frequencies interactively
        train_ratio: Proportion for training set
        test_ratio: Proportion for test set
        val_ratio: Proportion for validation set
    """
    # Validate ratios
    total_ratio = train_ratio + test_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get audio files by raga
    raga_files = get_audio_files_by_raga(source_dir)
    
    if not raga_files:
        print("No valid raga directories found in source directory.")
        return
    
    print(f"Found {len(raga_files)} ragas:")
    for raga, files in raga_files.items():
        print(f"  {raga}: {len(files)} files")
    
    # Create target directory structure
    splits = ["train", "test", "val"]
    for split in splits:
        audio_dir = Path(target_dir) / split / "audio"
        metadata_dir = Path(target_dir) / split / "metadata"
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFormatting dataset to: {target_dir}")
    
    # Process each raga
    total_processed = 0
    for raga, files in raga_files.items():
        print(f"\nProcessing raga: {raga}")
        
        # Distribute files to splits
        split_mapping = distribute_files_to_splits(files, train_ratio, test_ratio, val_ratio)
        
        for split_name, split_files in split_mapping.items():
            if not split_files:
                continue
                
            print(f"  {split_name}: {len(split_files)} files")
            
            for i, file_path in enumerate(split_files):
                source_file = Path(file_path)
                base_name = source_file.stem
                
                # Create unique recording ID
                recording_id = f"{raga.lower()}_{base_name}_{i+1:03d}"
                
                # Define paths in target structure
                target_audio_path = Path(target_dir) / split_name / "audio" / f"{recording_id}{source_file.suffix}"
                target_metadata_path = Path(target_dir) / split_name / "metadata" / f"{recording_id}.json"
                
                # Copy audio file to target
                shutil.copy2(source_file, target_audio_path)
                
                # Get or set tonic frequency
                if interactive:
                    tonic_hz = identify_tonic_interactive(str(source_file), raga)
                else:
                    # Use placeholder - in practice, you'd need to determine this
                    tonic_hz = 261.63  # Placeholder value
                    print(f"  Using placeholder tonic: {tonic_hz} Hz for {source_file.name}")
                
                # Create metadata
                create_metadata_file(
                    str(target_metadata_path),
                    recording_id,
                    raga,
                    tonic_hz,
                    split=split_name,
                    notes=f"Original file: {source_file.name}"
                )
                
                total_processed += 1
    
    print(f"\nDataset formatting completed!")
    print(f"Total files processed: {total_processed}")
    print(f"Formatted dataset location: {target_dir}")
    
    # Show final structure
    print("\nDataset structure:")
    for split in splits:
        split_path = Path(target_dir) / split
        audio_count = len(list((split_path / "audio").glob("*")))
        metadata_count = len(list((split_path / "metadata").glob("*")))
        print(f"  {split}: {audio_count} audio files, {metadata_count} metadata files")


def validate_formatted_dataset(dataset_dir: str) -> Dict[str, Any]:
    """
    Validate the formatted dataset structure and content.
    
    Args:
        dataset_dir: Path to the formatted dataset
        
    Returns:
        Dictionary with validation results
    """
    from raag_hmm.io.dataset import DatasetIterator
    
    try:
        iterator = DatasetIterator(dataset_dir)
        validation = iterator.validate_dataset()
        
        print("\nDataset Validation Results:")
        print(f"Valid: {validation['valid']}")
        if validation['issues']:
            print(f"Issues found: {validation['issues']}")
        
        for split_name, split_info in validation['splits'].items():
            print(f"{split_name} split:")
            print(f"  - Audio files: {split_info['audio_files']}")
            print(f"  - Metadata files: {split_info['metadata_files']}")
            print(f"  - Matched pairs: {split_info['matched_pairs']}")
            print(f"  - Raag distribution: {split_info['raag_distribution']}")
        
        return validation
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return {"valid": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Format Indian classical music dataset for RaagHMM system"
    )
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True,
        help="Source directory containing audio files organized by raga"
    )
    parser.add_argument(
        "--target_dir", 
        type=str, 
        required=True,
        help="Target directory for formatted dataset"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Interactively identify tonic frequencies (default: False)"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.7,
        help="Proportion for training set (default: 0.7)"
    )
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.2,
        help="Proportion for test set (default: 0.2)"
    )
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.1,
        help="Proportion for validation set (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.exists(args.source_dir):
        print(f"Source directory does not exist: {args.source_dir}")
        return
    
    # Format the dataset
    format_dataset(
        args.source_dir,
        args.target_dir,
        args.interactive,
        args.train_ratio,
        args.test_ratio,
        args.val_ratio
    )
    
    # Validate the formatted dataset
    print("\nValidating formatted dataset...")
    validation_result = validate_formatted_dataset(args.target_dir)


if __name__ == "__main__":
    print("Dataset Formatter for RaagHMM System")
    print("=" * 50)
    print("\nThis script helps format audio files into the structure required by RaagHMM.")
    print("\nBefore running:")
    print("1. Organize your audio files in subdirectories by raga name")
    print("2. Supported ragas: Bihag, Darbari, Desh, Gaud_Malhar, Yaman")
    print("3. Supported formats: WAV, MP3, FLAC, M4A, OGG")
    print("\nExample source structure:")
    print("source_dir/")
    print("├── Bihag/")
    print("│   ├── song1.wav")
    print("│   └── song2.mp3")
    print("└── Darbari/")
    print("    └── song1.wav")
    print("\nThis will be formatted to the RaagHMM structure with proper metadata.")