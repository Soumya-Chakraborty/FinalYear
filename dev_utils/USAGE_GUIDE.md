# RaagHMM Usage Guide

## Quick Start

The RaagHMM system is now fully operational! Here's how to use it:

### 1. Check System Status

```bash
# Check system information and requirements
python -m raag_hmm.cli.main info

# Show version
python -m raag_hmm.cli.main version

# Get usage examples
python -m raag_hmm.cli.main examples
```

### 2. Dataset Preparation

```bash
# Prepare dataset with resampling and validation
python -m raag_hmm.cli.main dataset prepare /path/to/raw/dataset /path/to/processed/dataset

# Validate dataset structure
python -m raag_hmm.cli.main dataset validate /path/to/dataset

# Extract pitch from single file
python -m raag_hmm.cli.main dataset extract-pitch audio.wav --metadata metadata.json --quantize
```

### 3. Model Training

```bash
# Train all raag models
python -m raag_hmm.cli.main train models /path/to/dataset /path/to/models

# Train with custom parameters
python -m raag_hmm.cli.main train models dataset/ models/ --max-iter 300 --tolerance 0.05

# Train single raag model
python -m raag_hmm.cli.main train single bihag /path/to/dataset bihag_model.pkl
```

### 4. Making Predictions

```bash
# Classify single audio file
python -m raag_hmm.cli.main predict single audio.wav models/ --tonic 261.63

# Batch classification
python -m raag_hmm.cli.main predict batch audio_dir/ models/ results.json

# With metadata for tonic
python -m raag_hmm.cli.main predict single audio.wav models/ --metadata metadata.json
```

### 5. Model Evaluation

```bash
# Evaluate on test set
python -m raag_hmm.cli.main evaluate test dataset/ models/ results/

# Compare multiple model sets
python -m raag_hmm.cli.main evaluate compare dataset/ models_v1/ models_v2/ comparison.json

# Custom evaluation parameters
python -m raag_hmm.cli.main evaluate test dataset/ models/ results/ --top-k 1 --top-k 5
```

## Command Reference

### Global Options

- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress output except errors
- `--debug`: Enable debug mode with detailed traces
- `--config, -c`: Path to configuration file

### Dataset Commands

- `dataset prepare`: Prepare dataset with resampling and validation
- `dataset extract-pitch`: Extract pitch from single audio file
- `dataset validate`: Validate dataset structure and integrity

### Training Commands

- `train models`: Train HMM models for all raag classes
- `train single`: Train single raag model

### Prediction Commands

- `predict single`: Classify single audio file
- `predict batch`: Batch classification of multiple files

### Evaluation Commands

- `evaluate test`: Evaluate models on test dataset
- `evaluate compare`: Compare multiple model sets

### Utility Commands

- `info`: Display system information and requirements
- `examples`: Show usage examples
- `version`: Show version information

## Dataset Format

The system expects datasets in the following structure:

```
dataset/
├── train/
│   ├── audio1.wav
│   ├── audio1.json
│   ├── audio2.wav
│   ├── audio2.json
│   └── ...
└── test/
    ├── audio3.wav
    ├── audio3.json
    └── ...
```

### Metadata Format

Each audio file should have a corresponding JSON metadata file:

```json
{
  "recording_id": "unique_id",
  "raag": "Bihag",
  "tonic_hz": 261.63,
  "artist": "Artist Name",
  "instrument": "sitar",
  "split": "train"
}
```

Supported raag names: `Bihag`, `Darbari`, `Desh`, `Gaud_Malhar`, `Yaman`

## Audio Formats

Supported audio formats:

- WAV (.wav)
- FLAC (.flac)
- MP3 (.mp3)
- M4A (.m4a)
- AAC (.aac)

All audio is automatically resampled to 22050 Hz mono.

## Error Handling

The system provides comprehensive error handling with helpful suggestions:

- File not found errors include suggestions for similar files
- Dataset validation errors show specific issues and fixes
- Model training errors provide troubleshooting guidance
- Audio processing errors suggest format conversions

## Getting Help

```bash
# Main help
python -m raag_hmm.cli.main --help

# Command-specific help
python -m raag_hmm.cli.main dataset --help
python -m raag_hmm.cli.main train models --help
python -m raag_hmm.cli.main predict single --help

# Usage examples
python -m raag_hmm.cli.main examples
python -m raag_hmm.cli.main examples dataset
```

## System Requirements

- Python 3.8+
- Required packages: numpy, scipy, librosa, soundfile, joblib
- Optional: praat-parselmouth (for Praat pitch extraction)
- Optional: scikit-learn (for additional metrics)

Check requirements with:

```bash
python -m raag_hmm.cli.main info
```

## Complete Workflow Example

```bash
# 1. Check system
python -m raag_hmm.cli.main info

# 2. Prepare dataset
python -m raag_hmm.cli.main dataset prepare raw_data/ processed_data/

# 3. Validate dataset
python -m raag_hmm.cli.main dataset validate processed_data/

# 4. Train models
python -m raag_hmm.cli.main train models processed_data/ models/

# 5. Make predictions
python -m raag_hmm.cli.main predict single test_audio.wav models/ --tonic 261.63

# 6. Evaluate performance
python -m raag_hmm.cli.main evaluate test processed_data/ models/ evaluation_results/
```

The RaagHMM system is now ready for production use!
