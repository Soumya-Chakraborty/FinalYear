# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CLI with dataset, train, predict, and evaluate commands
- HMM-based raag classification system
- Pitch extraction with Praat and librosa fallback
- Tonic normalization and chromatic quantization
- Model training and persistence system
- Comprehensive evaluation metrics with confusion matrices
- Audio I/O and dataset management utilities
- Configuration management system
- Logging infrastructure

### Changed
- Moved source code to `src/` directory for better packaging
- Updated pyproject.toml for modern Python packaging
- Improved documentation structure

### Fixed
- Fixed various bugs in pitch extraction and smoothing
- Fixed model serialization issues

### Removed
- Removed legacy setup files in favor of pyproject.toml