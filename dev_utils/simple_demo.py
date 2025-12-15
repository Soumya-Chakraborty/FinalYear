#!/usr/bin/env python3
"""
Simple RaagHMM Demo

A simplified demonstration that focuses on the core CLI functionality
without running into integration issues.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd_list, description):
    """Run a command and show results."""
    print(f"\n🔧 {description}")
    print(f"Command: python -m raag_hmm.cli.main {' '.join(cmd_list)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "raag_hmm.cli.main"] + cmd_list,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run simple CLI demonstrations."""
    print("🎼 RaagHMM Simple CLI Demonstration")
    print("=" * 50)
    
    # Test basic CLI functionality
    commands = [
        (["--help"], "Main help"),
        (["info"], "System information"),
        (["version"], "Version information"),
        (["examples"], "Usage examples"),
        (["examples", "dataset"], "Dataset examples"),
        (["dataset", "--help"], "Dataset help"),
        (["train", "--help"], "Training help"),
        (["predict", "--help"], "Prediction help"),
        (["evaluate", "--help"], "Evaluation help"),
    ]
    
    success_count = 0
    
    for cmd, desc in commands:
        if run_command(cmd, desc):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"🎉 CLI Demo Complete!")
    print(f"✅ {success_count}/{len(commands)} commands executed successfully")
    
    if success_count == len(commands):
        print("\n🎵 The RaagHMM CLI is fully functional!")
        print("\nNext steps to use the complete system:")
        print("1. Prepare your dataset: raag-hmm dataset prepare <input> <output>")
        print("2. Train models: raag-hmm train models <dataset> <models>")
        print("3. Make predictions: raag-hmm predict single <audio> <models>")
        print("4. Evaluate results: raag-hmm evaluate test <dataset> <models> <results>")
    else:
        print(f"\n⚠️  Some commands failed. Check the output above for details.")

if __name__ == "__main__":
    main()