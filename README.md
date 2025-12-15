# 🎵 RaagHMM: Hidden Markov Model-based Raag Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/github/license/raaghmm/raag-hmm)](LICENSE)
[![Build Status](https://github.com/raaghmm/raag-hmm/actions/workflows/test.yml/badge.svg)](https://github.com/raaghmm/raag-hmm/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/ambv/black)

> 🎶 **Indian Classical Music Meets Machine Learning**  
> Advanced raag recognition using Hidden Markov Models for Hindustani music analysis

---

## 🌟 Overview

**RaagHMM** is a state-of-the-art Python library designed for **automatic raag detection** in Indian classical music using **Hidden Markov Models** with discrete emissions. This system provides comprehensive tools for analyzing, training, and classifying musical pieces into the five fundamental ragas of Hindustani classical music.

### 🎭 Featured Ragas
- 🎼 **Bihag** - Late night serenity
- 🎼 **Darbari** - Midnight depth 
- 🎼 **Desh** - Evening romance
- 🎼 **Gaud Malhar** - Rainy season melody
- 🎼 **Yaman** - Evening tranquility

---

## ✨ Key Features

### 🎚️ Multi-Method Pitch Extraction
- **Primary**: Praat-based extraction for accuracy
- **Fallback**: Librosa implementation for robustness
- Adaptive voicing threshold with configurable parameters

### 🎯 Advanced Signal Processing
- **Median Filtering** for noise reduction
- **Gaussian Smoothing** for continuity
- **Gap Filling** for missing data
- **Octave Error Correction** for accuracy

### 📊 Discrete HMM Framework
- **36-state** models with full connectivity
- **Baum-Welch** training with numerical stability
- **Regularization** and probability floor for robustness
- **Forward-Backward** algorithm with scaling

### 🧠 Comprehensive Evaluation
- **Top-k** accuracy metrics (1, 3, 5)
- **Confusion matrices** with detailed analysis
- **Per-class** accuracy statistics
- **Confidence** scoring with probabilistic outputs

---

## 🚀 Quick Start

### Installation
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

### Development Installation
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## 🛠️ CLI Commands

### Dataset Management
```bash
# Prepare dataset with validation
raag-hmm dataset prepare /path/to/raw/dataset /path/to/processed

# Extract pitch from audio
raag-hmm dataset extract-pitch path/to/audio.wav

# Validate dataset integrity
raag-hmm dataset validate /path/to/dataset
```

### Model Training
```bash
# Train all raga models
raag-hmm train models /path/to/dataset /path/to/models

# Train single model
raag-hmm train single Bihag /path/to/dataset bihag_model.pkl
```

### Prediction & Inference
```bash
# Single audio classification
raag-hmm predict single audio.wav /path/to/models --tonic 261.63

# Batch classification
raag-hmm predict batch /path/to/audio/dir /path/to/models results.json
```

### Evaluation
```bash
# Test set evaluation
raag-hmm evaluate test /path/to/dataset /path/to/models /path/to/results

# Compare multiple model sets
raag-hmm evaluate compare /path/to/dataset model1/ model2/ results/
```

---

## 🏗️ Architecture

```
src/
└── raag_hmm/                 # Main package
    ├── 📁 cli/              # Command-line interface
    ├── 📁 hmm/              # HMM implementation  
    ├── 📁 train/            # Model training
    ├── 📁 infer/            # Classification & inference
    ├── 📁 io/               # Audio I/O & dataset management
    ├── 📁 pitch/            # Pitch extraction & processing
    ├── 📁 quantize/         # Chromatic quantization
    ├── 📁 evaluate/         # Performance evaluation
    ├── 📁 config.py         # Configuration system
    └── 📁 logger.py         # Logging infrastructure
```

---

## 🧪 Testing & Validation

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/raag_hmm

# Run specific test suites
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest -m integration        # Filter by marker
```

---

## 🔧 Configuration

The system provides flexible configuration through:

### Default Configuration
```python
config = {
    "audio": {
        "sample_rate": 22050,
        "channels": 1,
        "supported_formats": ["wav", "flac", "mp3"]
    },
    "pitch": {
        "frame_sec": 0.0464,
        "hop_sec": 0.01,
        "voicing_threshold": 0.5
    },
    "quantization": {
        "n_bins": 36,
        "base_midi": 48,  # C3
        "reference_tonic": 261.63  # C4 in Hz
    },
    "hmm": {
        "n_states": 36,
        "n_observations": 36,
        "max_iterations": 200,
        "convergence_tolerance": 0.1
    }
}
```

---

## 🎓 Academic & Research Applications

### Use Cases
- **Music Information Retrieval** research
- **Cultural Heritage** preservation projects
- **Automated Music** analysis systems
- **Educational** tools for Indian classical music
- **Ethnomusicology** studies

### Research Extensions
- Multi-artist raag classification
- Tempo-invariant recognition
- Cross-tradition (Carnatic/Hindustani) analysis
- Real-time performance monitoring

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use RaagHMM in your research, please cite:

```bibtex
@software{raaghmm2024,
  title={RaagHMM: Hidden Markov Model-based Raag Detection},
  author={RaagHMM Development Team},
  year={2024},
  url={https://github.com/raaghmm/raag-hmm}
}
```

---

## 🆘 Support & Contact

- 🐛 **Issues**: [GitHub Issues](https://github.com/raaghmm/raag-hmm/issues)
- 📧 **Development Team**: dev@raaghmm.org
- 📖 **Documentation**: [Read the Docs](https://raag-hmm.readthedocs.io/)

---

<p align="center">
  <b>Made with ❤️ for Indian Classical Music</b>  
</p>

<p align="center">
  <sub>The RaagHMM project is committed to preserving and advancing the understanding of Indian classical music through technology.</sub>
</p>