# RaagHMM: Hidden Markov Model-based Raag Detection

A Python library for automatic raag (raga) detection in Indian classical music using Hidden Markov Models with discrete emissions.

## Overview

RaagHMM implements an end-to-end system for identifying raag patterns in Indian classical music recordings. The system extracts pitch contours from audio, quantizes them to chromatic bins normalized by tonic, and uses trained HMM models to classify musical pieces into one of five raag classes: Bihag, Darbari, Desh, Gaud_Malhar, and Yaman.

## Features

- **Multi-method pitch extraction**: Primary Praat-based extraction with librosa fallback
- **Chromatic quantization**: 36-bin chromatic scale with tonic normalization  
- **Discrete HMM training**: Baum-Welch algorithm with numerical stability
- **Comprehensive evaluation**: Accuracy metrics, confusion matrices, and detailed analysis
- **Command-line interface**: Easy-to-use CLI for all operations
- **Extensible architecture**: Modular design for adding new raag classes and features

## Installation

```bash
# Clone the repository
git clone https://github.com/raaghmm/raag-hmm.git
cd raag-hmm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```bash
# Train models on dataset
raag-hmm train --dataset-path /path/to/dataset --output-dir models/

# Classify a single audio file
raag-hmm predict --audio-path recording.wav --models-dir models/ --tonic-hz 261.63

# Evaluate on test set
raag-hmm evaluate --dataset-path /path/to/dataset --models-dir models/ --split test
```

## Project Structure

```
raag_hmm/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── logger.py            # Logging infrastructure
├── exceptions.py        # Exception hierarchy
├── io/                  # Audio I/O and dataset management
├── pitch/               # Pitch extraction
├── quantize/            # Chromatic quantization
├── hmm/                 # HMM implementation
├── train/               # Model training
├── infer/               # Classification and inference
├── evaluate/            # Performance evaluation
└── cli/                 # Command-line interface
```

## Requirements

- Python 3.8+
- NumPy, SciPy for numerical computing
- librosa, soundfile for audio processing
- praat-parselmouth for pitch extraction
- scikit-learn, joblib for machine learning
- typer, rich for CLI interface

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use RaagHMM in your research, please cite:

```bibtex
@software{raaghmm2024,
  title={RaagHMM: Hidden Markov Model-based Raag Detection},
  author={RaagHMM Development Team},
  year={2024},
  url={https://github.com/raaghmm/raag-hmm}
}
```