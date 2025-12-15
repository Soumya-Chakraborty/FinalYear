#!/usr/bin/env python3
"""
Helper script to create metadata files for your MP3 collection.

This script helps you generate the required JSON metadata files for each audio file.
"""

import json
import os
from pathlib import Path
import librosa

# Common tonic frequencies (in Hz) for reference
COMMON_TONICS = {
    "C": 261.63,
    "C#/Db": 277.18,
    "D": 293.66,
    "D#/Eb": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#/Gb": 369.99,
    "G": 392.00,
    "G#/Ab": 415.30,
    "A": 440.00,
    "A#/Bb": 466.16,
    "B": 493.88
}

VALID_RAAGS = ["Bihag", "Darbari", "Desh", "Gaud_Malhar", "Yaman"]

def get_audio_duration(audio_file):
    """Get duration of audio file."""
    try:
        duration = librosa.get_duration(path=str(audio_file))
        return duration
    except:
        return None

def create_metadata_interactive(audio_file, split="train"):
    """Create metadata interactively for an audio file."""
    print(f"\n📁 Creating metadata for: {audio_file.name}")
    
    # Get basic info
    recording_id = input(f"Recording ID [{audio_file.stem}]: ").strip() or audio_file.stem
    
    # Select raag
    print("\nAvailable raags:")
    for i, raag in enumerate(VALID_RAAGS, 1):
        print(f"  {i}. {raag}")
    
    while True:
        try:
            raag_choice = input("Select raag (1-5): ").strip()
            raag_idx = int(raag_choice) - 1
            if 0 <= raag_idx < len(VALID_RAAGS):
                raag = VALID_RAAGS[raag_idx]
                break
            else:
                print("Invalid choice. Please select 1-5.")
        except ValueError:
            print("Please enter a number 1-5.")
    
    # Select tonic
    print(f"\nCommon tonic frequencies:")
    for note, freq in COMMON_TONICS.items():
        print(f"  {note}: {freq} Hz")
    
    while True:
        tonic_input = input("Enter tonic frequency (Hz) or note name: ").strip()
        try:
            if tonic_input.upper() in [k.upper() for k in COMMON_TONICS.keys()]:
                # Find the matching key (case insensitive)
                for note, freq in COMMON_TONICS.items():
                    if note.upper() == tonic_input.upper():
                        tonic_hz = freq
                        break
            else:
                tonic_hz = float(tonic_input)
            
            if 80 <= tonic_hz <= 800:  # Reasonable range
                break
            else:
                print("Tonic frequency should be between 80-800 Hz")
        except ValueError:
            print("Please enter a valid frequency or note name")
    
    # Optional fields
    artist = input("Artist name (optional): ").strip() or "Unknown"
    instrument = input("Instrument (optional): ").strip() or "unknown"
    notes = input("Notes (optional): ").strip() or ""
    
    # Get duration
    duration = get_audio_duration(audio_file)
    
    # Create metadata
    metadata = {
        "recording_id": recording_id,
        "raag": raag,
        "tonic_hz": tonic_hz,
        "artist": artist,
        "instrument": instrument,
        "split": split,
        "notes": notes
    }
    
    if duration:
        metadata["duration"] = round(duration, 2)
    
    return metadata

def process_directory(directory, split="train"):
    """Process all audio files in a directory."""
    directory = Path(directory)
    
    # Find audio files
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(directory.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"No audio files found in {directory}")
        return
    
    print(f"Found {len(audio_files)} audio files in {directory}")
    
    for audio_file in audio_files:
        metadata_file = audio_file.with_suffix('.json')
        
        if metadata_file.exists():
            overwrite = input(f"Metadata exists for {audio_file.name}. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                continue
        
        try:
            metadata = create_metadata_interactive(audio_file, split)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Created: {metadata_file.name}")
            
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            break
        except Exception as e:
            print(f"❌ Error creating metadata for {audio_file.name}: {e}")

def main():
    """Main function."""
    print("🎵 RaagHMM Metadata Creator")
    print("=" * 40)
    
    print("\nThis script helps you create metadata files for your audio collection.")
    print("Make sure your audio files are organized in train/ and test/ directories.")
    
    while True:
        print("\nOptions:")
        print("1. Process train directory")
        print("2. Process test directory")
        print("3. Process custom directory")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            train_dir = input("Enter path to train directory [./train]: ").strip() or "./train"
            if Path(train_dir).exists():
                process_directory(train_dir, "train")
            else:
                print(f"Directory not found: {train_dir}")
        
        elif choice == "2":
            test_dir = input("Enter path to test directory [./test]: ").strip() or "./test"
            if Path(test_dir).exists():
                process_directory(test_dir, "test")
            else:
                print(f"Directory not found: {test_dir}")
        
        elif choice == "3":
            custom_dir = input("Enter directory path: ").strip()
            if Path(custom_dir).exists():
                split = input("Split type (train/test) [train]: ").strip() or "train"
                process_directory(custom_dir, split)
            else:
                print(f"Directory not found: {custom_dir}")
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()