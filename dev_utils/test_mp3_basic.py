#!/usr/bin/env python3
"""
Basic MP3 test - Check if your MP3 file can be processed by RaagHMM.
"""

import numpy as np
from pathlib import Path

def test_mp3_file(mp3_path):
    """Test basic MP3 processing."""
    print(f"🎵 Testing MP3 file: {mp3_path}")
    
    try:
        # Test 1: Load audio
        print("1. Loading audio...")
        from raag_hmm.io.audio import load_audio
        audio_data = load_audio(mp3_path)
        print(f"   ✅ Audio loaded: {len(audio_data)} samples")
        
        # Test 2: Extract pitch
        print("2. Extracting pitch...")
        from raag_hmm.pitch.extractor import extract_pitch_praat
        f0_hz, voicing_prob = extract_pitch_praat(audio_data, sr=22050)
        print(f"   ✅ Pitch extracted: {len(f0_hz)} frames")
        
        # Test 3: Smooth pitch
        print("3. Smoothing pitch...")
        from raag_hmm.pitch.smoother import smooth_pitch
        f0_smooth = smooth_pitch(f0_hz, voicing_prob)
        print(f"   f0_smooth type: {type(f0_smooth)}")
        
        # Handle if smooth_pitch returns a tuple
        if isinstance(f0_smooth, tuple):
            f0_smooth = f0_smooth[0]  # Take first element
        
        print(f"   ✅ Pitch smoothed")
        
        # Test 4: Quantize
        print("4. Quantizing pitch...")
        print(f"   f0_smooth shape: {f0_smooth.shape}")
        print(f"   f0_smooth type: {type(f0_smooth)}")
        
        from raag_hmm.quantize.sequence import quantize_sequence
        tonic_hz = 261.63  # C4
        
        # Ensure 1D array
        if f0_smooth.ndim > 1:
            f0_smooth = f0_smooth.flatten()
        
        quantized = quantize_sequence(f0_smooth, tonic_hz)
        print(f"   ✅ Pitch quantized: {len(quantized)} frames")
        
        # Show results
        valid_frames = np.sum(~np.isnan(f0_smooth))
        print(f"\n📊 Results:")
        print(f"   Total frames: {len(f0_hz)}")
        print(f"   Valid pitch frames: {valid_frames}")
        print(f"   Quantized range: {quantized.min()} - {quantized.max()}")
        
        print(f"\n✅ Your MP3 file works perfectly with RaagHMM!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🎼 RaagHMM MP3 Basic Test")
    print("=" * 30)
    
    # Test the user's file
    mp3_file = "/home/developer/Downloads/01RagaBhairavi-Madhya-layGatinTeental.mp3"
    
    if not Path(mp3_file).exists():
        print(f"❌ File not found: {mp3_file}")
        return
    
    success = test_mp3_file(mp3_file)
    
    if success:
        print(f"\n🎯 Next Steps:")
        print(f"1. Your MP3 file is compatible with RaagHMM")
        print(f"2. To train a model, you need more MP3 files of different raags")
        print(f"3. Create metadata files using: python create_metadata.py")
        print(f"4. Train models using: python train_with_my_data.py")
    else:
        print(f"\n❌ There were issues processing your MP3 file")
        print(f"   Try converting to WAV format or check audio quality")

if __name__ == "__main__":
    main()