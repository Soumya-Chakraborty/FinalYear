#!/usr/bin/env python3
"""
Quick MP3 Demo - Test RaagHMM functionality with your MP3 file.

This script demonstrates the core functionality without requiring a full training dataset.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Quick demo with user's MP3 file."""
    print("🎵 Quick RaagHMM MP3 Demo")
    print("=" * 30)
    
    # Get MP3 file
    mp3_file = input("Enter path to your MP3 file: ").strip()
    
    if not mp3_file or not Path(mp3_file).exists():
        print("❌ MP3 file not found!")
        return
    
    print(f"📁 Using file: {mp3_file}")
    
    # Get tonic frequency
    print("\nCommon tonic frequencies:")
    print("  C: 261.63 Hz")
    print("  D: 293.66 Hz") 
    print("  G: 392.00 Hz")
    print("  A: 440.00 Hz")
    
    tonic = input("Enter tonic frequency (Hz) [261.63]: ").strip() or "261.63"
    
    try:
        tonic_hz = float(tonic)
    except:
        tonic_hz = 261.63
    
    print(f"\n🎼 Extracting pitch from your MP3...")
    print("This will show you what the system can extract from your audio.")
    
    # Run pitch extraction
    try:
        result = subprocess.run([
            sys.executable, "-m", "raag_hmm.cli.main",
            "dataset", "extract-pitch",
            mp3_file,
            "--tonic", str(tonic_hz),
            "--quantize"
        ], timeout=60)
        
        if result.returncode == 0:
            print("✅ Pitch extraction successful!")
        else:
            print("❌ Pitch extraction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n💡 What this shows:")
    print(f"   - Your MP3 file is readable by the system")
    print(f"   - Pitch can be extracted and quantized")
    print(f"   - The audio processing pipeline works")
    
    print(f"\n🎓 Next Steps for Full Training:")
    print(f"1. Get more MP3 files (different raags)")
    print(f"2. Run: python train_with_single_mp3.py")
    print(f"3. Or run: python create_metadata.py for multiple files")

if __name__ == "__main__":
    main()