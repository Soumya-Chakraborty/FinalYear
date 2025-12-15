#!/usr/bin/env python3
"""
Utility script to help create a custom Indian classical music dataset
for RaagHMM system by downloading from YouTube with proper attribution.

This script is provided for educational and research purposes.
Make sure to comply with YouTube's terms of service and copyright laws.

Usage:
1. Install required packages: pip install yt-dlp pydub
2. Run this script with a list of raga-specific YouTube URLs
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
from pydub import AudioSegment


def download_audio_from_youtube(url: str, output_path: str, start_time: str = None, end_time: str = None) -> bool:
    """
    Download audio from YouTube URL and save as WAV file.
    
    Args:
        url: YouTube video URL
        output_path: Output path for the audio file
        start_time: Start time in format 'MM:SS' (optional)
        end_time: End time in format 'MM:SS' (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Prepare yt-dlp command
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',  # Output format
            '--audio-quality', '0',  # Best quality
            '-o', output_path,  # Output filename
            url
        ]
        
        if start_time and end_time:
            # Add time range if specified
            cmd.extend(['--download-sections', f'*{start_time}-{end_time}'])
        
        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error downloading {url}: {result.stderr}")
            return False
            
        print(f"Successfully downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"Exception while downloading {url}: {e}")
        return False


def identify_tonic_frequency(audio_path: str) -> float:
    """
    A simplified approach to estimate tonic frequency.
    For production use, this should be done by an expert or with advanced algorithms.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Estimated tonic frequency in Hz
    """
    # This is a placeholder - in practice you'd use advanced audio analysis
    # or have an expert identify the tonic
    print(f"Please manually identify the tonic frequency for: {audio_path}")
    print("You can use tools like Sonic Visualiser with Vamp plugins or manually identify the Sa (tonic).")
    return float(input("Enter tonic frequency in Hz: "))


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
    
    print(f"Metadata created: {output_path}")


def validate_raag_name(raag: str) -> bool:
    """
    Validate if the raga name is one of the supported ragas for RaagHMM.
    
    Args:
        raag: Raag name to validate
    
    Returns:
        True if valid, False otherwise
    """
    valid_ragas = {"Bihag", "Darbari", "Desh", "Gaud_Malhar", "Yaman"}
    return raag in valid_ragas


def create_dataset_structure(base_dir: str):
    """
    Create the required directory structure for RaagHMM.
    
    Args:
        base_dir: Base directory for the dataset
    """
    splits = ["train", "test", "val"]
    
    for split in splits:
        audio_dir = Path(base_dir) / split / "audio"
        metadata_dir = Path(base_dir) / split / "metadata"
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset structure created at: {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Utility to help create a custom Indian classical music dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                       help="Directory to create the dataset in")
    parser.add_argument("--urls_file", type=str, required=True,
                       help="Path to a text file containing YouTube URLs and metadata")
    
    args = parser.parse_args()
    
    # Read URLs and metadata from file
    # Format: URL|RAAG|ARTIST|INSTRUMENT|START_TIME|END_TIME|SPLIT
    # Example: https://www.youtube.com/watch?v=abc123|Bihag|Artist Name|Sitar|01:30|05:30|train
    with open(args.urls_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Create dataset structure
    create_dataset_structure(args.dataset_dir)
    
    for i, line in enumerate(lines):
        parts = line.split('|')
        if len(parts) < 4:
            print(f"Invalid line format: {line}")
            continue
        
        url = parts[0].strip()
        raag = parts[1].strip()
        artist = parts[2].strip()
        instrument = parts[3].strip()
        start_time = parts[4].strip() if len(parts) > 4 else None
        end_time = parts[5].strip() if len(parts) > 5 else None
        split = parts[6].strip() if len(parts) > 6 else "train"
        
        if not validate_raag_name(raag):
            print(f"Invalid raga name: {raag} (must be one of: Bihag, Darbari, Desh, Gaud_Malhar, Yaman)")
            continue
        
        # Create file names
        recording_id = f"{raag.lower()}_recording_{i+1:04d}"
        audio_filename = f"{recording_id}.wav"
        metadata_filename = f"{recording_id}.json"
        
        # Define paths
        audio_path = os.path.join(args.dataset_dir, split, "audio", audio_filename)
        metadata_path = os.path.join(args.dataset_dir, split, "metadata", metadata_filename)
        
        # Download audio
        if download_audio_from_youtube(url, audio_path, start_time, end_time):
            # Identify tonic (placeholder - would need manual input)
            print(f"Processing recording: {recording_id}")
            print(f"Raga: {raag}, Artist: {artist}, Instrument: {instrument}")
            print(f"Split: {split}")
            
            # For demo purposes, we'll use a placeholder tonic frequency
            # In practice, you'd identify this manually or with analysis
            tonic_hz = 261.63  # Placeholder - you should identify the real tonic
            
            # Create metadata
            create_metadata_file(
                metadata_path, recording_id, raag, tonic_hz, 
                artist, instrument, split,
                notes=f"Downloaded from YouTube: {url}"
            )


if __name__ == "__main__":
    # Example usage:
    # Create an example URLs file
    example_urls_content = """# YouTube URLs for Indian Classical Music Dataset
# Format: URL|RAAG|ARTIST|INSTRUMENT|START_TIME|END_TIME|SPLIT
# Example: https://www.youtube.com/watch?v=example|Bihag|Artist Name|Sitar|01:30|05:30|train
# Use time segments to get the characteristic portions of the raga
https://www.youtube.com/watch?v=example1|Bihag|Artist Name|Sitar|02:00|07:00|train
https://www.youtube.com/watch?v=example2|Darbari|Artist Name|Vocal|03:15|08:45|train
https://www.youtube.com/watch?v=example3|Desh|Artist Name|Flute|01:20|06:30|train
https://www.youtube.com/watch?v=example4|Gaud_Malhar|Artist Name|Tabla|04:10|09:20|train
https://www.youtube.com/watch?v=example5|Yaman|Artist Name|Guitar|02:45|07:55|train
https://www.youtube.com/watch?v=example6|Bihag|Artist Name|Sitar|10:15|15:25|test
"""
    
    with open("/home/developer/FinalYear/example_urls.txt", "w") as f:
        f.write(example_urls_content)
    
    print("Created example URL file at: /home/developer/FinalYear/example_urls.txt")
    print("\nTo use this script:")
    print("1. Install requirements: pip install yt-dlp pydub")
    print("2. Edit the example_urls.txt file with real YouTube URLs")
    print("3. Run: python create_dataset_from_youtube.py --dataset_dir /path/to/dataset --urls_file example_urls.txt")
    print("\nIMPORTANT: Make sure you comply with YouTube's ToS and copyright laws when downloading content.")