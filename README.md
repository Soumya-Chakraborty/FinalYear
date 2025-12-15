# RaagHMM: Hidden Markov Model-based Raag Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/github/license/raaghmm/raag-hmm)](LICENSE)
[![Build Status](https://github.com/raaghmm/raag-hmm/actions/workflows/test.yml/badge.svg)](https://github.com/raaghmm/raag-hmm/actions)

A Python library for automatic raag (raga) detection in Indian classical music using Hidden Markov Models with discrete emissions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [CLI Reference](#cli-reference)
- [Development](#development)
- [Testing](#testing)
- [License](#license)
- [Citation](#citation)

## Features

- **Multi-method pitch extraction**: Primary Praat-based extraction with librosa fallback
- **Robust pitch smoothing**: Median filtering, Gaussian smoothing, gap filling, octave error correction
- **Chromatic quantization**: 36-bin chromatic scale with tonic normalization
- **Discrete HMM training**: Baum-Welch algorithm with numerical stability
- **Comprehensive evaluation**: Accuracy metrics, confusion matrices, top-k analysis
- **Rich CLI interface**: Easy-to-use command line tools with rich formatting
- **Extensible architecture**: Modular design for adding new raag classes and features

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install from Source

```bash
# Clone the repository
git clone https://github.com/raaghmm/raag-hmm.git
cd raag-hmm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Install in Development Mode

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### CLI Usage

```bash
# Prepare dataset
raag-hmm dataset prepare /path/to/raw/dataset /path/to/processed/dataset

# Train models
raag-hmm train models /path/to/dataset /path/to/models

# Classify single audio file
raag-hmm predict single audio.wav /path/to/models --tonic 261.63

# Evaluate models
raag-hmm evaluate test /path/to/dataset /path/to/models /path/to/results
```

### Programmatic Usage

```python
from raag_hmm.train import RaagTrainer
from raag_hmm.infer import RaagClassifier

# Train models
trainer = RaagTrainer()
models = trainer.train_all_raag_models('path/to/dataset')

# Classify audio
classifier = RaagClassifier('path/to/models')
result = classifier.predict_with_confidence(quantized_sequence)
```

## Architecture

The RaagHMM system follows a modular architecture:

```
src/
└── raag_hmm/                 # Main package
    ├── __init__.py          # Package initialization
    ├── config.py            # Configuration management
    ├── logger.py            # Logging infrastructure
    ├── exceptions.py        # Exception hierarchy
    ├── cli/                 # Command-line interface
    ├── evaluate/            # Performance evaluation
    ├── hmm/                 # HMM implementation
    ├── infer/               # Classification and inference
    ├── io/                  # Audio I/O and dataset management
    ├── pitch/               # Pitch extraction
    ├── quantize/            # Chromatic quantization
    └── train/               # Model training
```

## CLI Reference

### Dataset Commands
```bash
raag-hmm dataset prepare <input> <output>      # Prepare dataset
raag-hmm dataset extract-pitch <audio>         # Extract pitch from audio
raag-hmm dataset validate <dataset>            # Validate dataset
```

### Training Commands
```bash
raag-hmm train models <dataset> <models>       # Train all models
raag-hmm train single <raag> <dataset> <model> # Train single model
```

### Prediction Commands
```bash
raag-hmm predict single <audio> <models>       # Classify single audio
raag-hmm predict batch <dir> <models> <out>    # Batch classification
```

### Evaluation Commands
```bash
raag-hmm evaluate test <dataset> <models> <out> # Evaluate test set
raag-hmm evaluate compare <dataset> <models...> <out> # Compare models
```

## Development

### Setting up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/raag-hmm.git
cd raag-hmm

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Python code follows PEP 8 standards
- Code is formatted with Black
- Type hints are required for public functions
- Docstrings follow Google style

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/raag_hmm

# Run specific test module
pytest tests/test_module.py

# Run integration tests only
pytest -m integration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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