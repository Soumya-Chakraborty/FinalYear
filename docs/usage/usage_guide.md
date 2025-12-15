# RaagHMM Usage Guide

This guide provides comprehensive information on how to use the RaagHMM system for raag classification in Indian classical music.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Usage](#cli-usage)
4. [Programmatic API](#programmatic-api)
5. [Dataset Format](#dataset-format)
6. [Model Training](#model-training)
7. [Prediction and Classification](#prediction-and-classification)
8. [Evaluation](#evaluation)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Development Installation
```bash
git clone https://github.com/raaghmm/raag-hmm.git
cd raag-hmm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Production Installation
```bash
pip install raag-hmm
```

## Quick Start

### Using the Command Line Interface

The RaagHMM package provides a rich command-line interface for all operations:

```bash
# View help
raag-hmm --help

# View specific command help
raag-hmm dataset --help
raag-hmm train --help
raag-hmm predict --help
raag-hmm evaluate --help
```

### Basic Workflow Example

```bash
# 1. Prepare dataset (resample audio, validate metadata)
raag-hmm dataset prepare /path/to/raw/dataset /path/to/processed/dataset

# 2. Train models for all raag classes
raag-hmm train models /path/to/processed/dataset /path/to/models

# 3. Classify a single audio file
raag-hmm predict single /path/to/audio.wav /path/to/models --tonic 261.63

# 4. Evaluate models on test set
raag-hmm evaluate test /path/to/dataset /path/to/models /path/to/results
```

## CLI Usage

### Dataset Commands

#### Prepare Dataset
```bash
raag-hmm dataset prepare [OPTIONS] INPUT_DIR OUTPUT_DIR

# Options:
# --sample-rate, -sr: Target sample rate (default: 22050)
# --force, -f: Overwrite existing output directory
# --validate-only: Only validate dataset structure
```

#### Extract Pitch from Audio
```bash
raag-hmm dataset extract-pitch [OPTIONS] AUDIO_FILE

# Options:
# --output, -o: Output file for pitch data
# --metadata, -m: Metadata file with tonic information
# --tonic, -t: Tonic frequency in Hz
# --method: Pitch extraction method (praat, librosa, auto)
# --smooth/--no-smooth: Apply smoothing (default: True)
# --quantize: Quantize pitch to chromatic bins
```

#### Validate Dataset
```bash
raag-hmm dataset validate [OPTIONS] DATASET_DIR

# Options:
# --split: Specific split to validate (train/test/val)
# --check-audio/--no-check-audio: Validate audio files
```

### Training Commands

#### Train All Models
```bash
raag-hmm train models [OPTIONS] DATASET_DIR OUTPUT_DIR

# Options:
# --max-iter, -i: Maximum training iterations (default: 200)
# --tolerance, -t: Convergence tolerance (default: 0.1)
# --regularization, -r: Regularization parameter (default: 0.01)
# --states, -s: Number of HMM states (default: 36)
# --force, -f: Overwrite existing models
# --raag: Filter to specific raag (can be used multiple times)
```

#### Train Single Model
```bash
raag-hmm train single [OPTIONS] RAAG_NAME DATASET_DIR OUTPUT_FILE

# Options:
# --max-iter, -i: Maximum training iterations
# --tolerance, -t: Convergence tolerance
# --states, -s: Number of HMM states
```

### Prediction Commands

#### Single Audio Prediction
```bash
raag-hmm predict single [OPTIONS] AUDIO_FILE MODELS_DIR

# Options:
# --metadata, -m: Metadata file with tonic
# --tonic, -t: Tonic frequency in Hz
# --output, -o: Output file for results
# --top-k: Number of top predictions (default: 3)
# --threshold: Confidence threshold
```

#### Batch Prediction
```bash
raag-hmm predict batch [OPTIONS] INPUT_DIR MODELS_DIR OUTPUT_FILE

# Options:
# --pattern, -p: File pattern (default: *.wav)
# --require-metadata/--no-require-metadata: Require metadata files
# --default-tonic: Default tonic when metadata missing
# --top-k: Number of top predictions (default: 3)
```

### Evaluation Commands

#### Test Set Evaluation
```bash
raag-hmm evaluate test [OPTIONS] DATASET_DIR MODELS_DIR OUTPUT_DIR

# Options:
# --split: Dataset split (default: test)
# --top-k: Top-K values to compute (default: 1,3,5)
# --format: Export formats (json, csv) (default: both)
# --detailed-errors/--no-detailed-errors: Include error analysis
```

#### Model Comparison
```bash
raag-hmm evaluate compare [OPTIONS] DATASET_DIR MODELS_DIR1 MODELS_DIR2... OUTPUT_FILE

# Options:
# --split: Dataset split (default: test)
# --name: Names for model sets (for display)
```

## Programmatic API

### Training Models

```python
from raag_hmm.train import RaagTrainer, ModelPersistence

# Initialize trainer
trainer = RaagTrainer(
    n_states=36,
    n_observations=36,
    max_iterations=200,
    convergence_tolerance=0.1,
    regularization_alpha=0.01
)

# Train all models
trained_models = trainer.train_all_raag_models(
    dataset_root="path/to/dataset",
    split="train",
    verbose=True
)

# Save models
persistence = ModelPersistence("models/")
persistence.save_all_models(trained_models)
```

### Loading and Using Models

```python
from raag_hmm.infer import RaagClassifier

# Initialize classifier with model directory
classifier = RaagClassifier("models/")

# Make predictions
sequence = [0, 2, 4, 5, 7, 9, 11, 0]  # Quantized pitch sequence
predicted_raag = classifier.predict_raag(sequence)

# Get predictions with confidence
result = classifier.predict_with_confidence(sequence)
print(f"Predicted: {result['predicted_raag']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"All scores: {result['scores']}")
```

### Processing Audio

```python
from raag_hmm.io import load_audio
from raag_hmm.pitch import extract_pitch_with_fallback, smooth_pitch
from raag_hmm.quantize import quantize_sequence

# Load audio
audio_data = load_audio("path/to/audio.wav")

# Extract pitch
f0_hz, voicing_prob = extract_pitch_with_fallback(audio_data, sr=22050)

# Smooth pitch
f0_smoothed = smooth_pitch(f0_hz, voicing_prob)

# Quantize sequence (requires tonic frequency)
tonic_hz = 261.63  # C4
quantized_sequence = quantize_sequence(f0_smoothed, tonic_hz)
```

## Dataset Format

### Directory Structure

```
dataset/
├── train/
│   ├── audio1.wav
│   ├── audio1.json
│   ├── audio2.wav
│   └── audio2.json
├── test/
│   ├── test1.wav
│   └── test1.json
└── val/
    ├── val1.wav
    └── val1.json
```

### Metadata Format (JSON)

```json
{
  "recording_id": "unique_identifier",
  "raag": "Bihag",  // One of: Bihag, Darbari, Desh, Gaud_Malhar, Yaman
  "tonic_hz": 261.63,  // Tonic frequency in Hz
  "artist": "Artist Name",  // Optional
  "instrument": "sitar",  // Optional
  "split": "train",  // train, test, or val
  "notes": "Additional notes",  // Optional
  "duration_sec": 300.5  // Optional
}
```

### Supported Audio Formats
- WAV
- FLAC  
- MP3
- M4A
- OGG

All audio is automatically resampled to 22050 Hz during processing.

## Model Training

### HMM Configuration

The system uses Discrete Hidden Markov Models with:
- **States**: 36 (representing chromatic pitch classes)
- **Observations**: 36 (chromatic quantization bins)
- **Topology**: Fully connected (ergodic)

### Training Parameters

Key training parameters that can be configured:
- `n_states`: Number of hidden states (default: 36)
- `max_iterations`: Maximum EM iterations (default: 200)
- `convergence_tolerance`: Log-likelihood improvement threshold (default: 0.1)
- `regularization_alpha`: Dirichlet regularization (default: 0.01)
- `probability_floor`: Minimum probability (default: 1e-8)

### Training Process

1. **Data Loading**: Load audio files and metadata from dataset
2. **Feature Extraction**: Extract and smooth pitch contours
3. **Quantization**: Convert pitch to chromatic bins with tonic normalization
4. **Model Training**: Train separate HMM for each raag class using Baum-Welch
5. **Model Persistence**: Save trained models with metadata

## Prediction and Classification

### Classification Process

1. **Audio Processing**: Load and process audio file
2. **Pitch Extraction**: Extract fundamental frequency contour
3. **Smoothing**: Apply median filtering, Gaussian smoothing, and gap filling
4. **Normalization**: Normalize by tonic frequency
5. **Quantization**: Convert to chromatic bins
6. **Scoring**: Compute log-likelihood for each trained model
7. **Decision**: Select raag with highest likelihood

### Confidence Scoring

The system provides confidence scores using:
- **Log-likelihood**: Raw likelihood scores from HMMs
- **Normalized probabilities**: Softmax-transformed scores
- **Top-K predictions**: Ranked list of most likely raags
- **Score margins**: Difference between top predictions

## Evaluation

### Metrics Computed

- **Overall accuracy**: Proportion of correct predictions
- **Per-class accuracy**: Accuracy for each individual raag
- **Balanced accuracy**: Average of per-class accuracies
- **Top-K accuracy**: Accuracy considering top K predictions
- **Confusion matrix**: Detailed classification breakdown
- **Per-class metrics**: Precision, recall, F1-score per raag
- **Confidence statistics**: Distribution of prediction confidence

### Export Formats

Evaluation results can be exported in:
- **JSON**: Complete structured results
- **CSV**: Tabular format for spreadsheets
- **Reports**: Formatted analysis with visualizations

## Configuration

### Global Configuration

The system uses a hierarchical configuration system:

1. **Default values**: Built-in defaults in `raag_hmm/config.py`
2. **Environment variables**: Override with `RAAG_HMM_*` prefixes
3. **Configuration file**: Custom file (TBD)

### Configuration Options

```python
from raag_hmm.config import get_config, set_config

# Get specific configuration value
sample_rate = get_config('audio', 'sample_rate')

# Set configuration value
set_config('hmm', 'max_iterations', 300)
```

## Troubleshooting

### Common Issues

#### Missing Dependencies
- **Praat (parselmouth)**: Required for primary pitch extraction
- **Librosa**: Required as fallback pitch extraction
- **Soundfile**: Required for audio I/O

Install missing dependencies:
```bash
pip install praat-parselmouth librosa soundfile
```

#### Audio Processing Failures
- Check audio file format and integrity
- Verify sample rate compatibility
- Ensure sufficient audio length (> 1 second recommended)

#### Training Issues
- **Low convergence**: Try increasing max_iterations or adjusting tolerance
- **Numerical instability**: Verify quantization ranges and normalization
- **Poor accuracy**: Check dataset quality and class balance

#### Memory Issues
- Process large datasets in smaller batches
- Use appropriate HMM dimensions for your use case
- Consider using swap space for large models

### Debugging

Enable verbose logging:
```bash
# CLI
raag-hmm --verbose train models dataset/ models/

# Programmatic
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Support

For additional support:
1. Check the [GitHub Issues](https://github.com/raaghmm/raag-hmm/issues)
2. Create a detailed issue with:
   - Python version
   - Package version
   - Error message
   - Steps to reproduce
   - Expected vs actual behavior