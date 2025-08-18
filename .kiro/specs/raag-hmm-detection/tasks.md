# Implementation Plan

- [x] 1. Set up project structure and core configuration

  - Create Python package structure with proper **init**.py files
  - Implement configuration management system with default settings
  - Set up logging infrastructure for debugging and monitoring
  - Create requirements.txt with all necessary dependencies
  - _Requirements: 7.1, 7.2, 8.5_

- [x] 2. Implement audio I/O and dataset management

  - [x] 2.1 Create audio loading functionality with format support

    - Implement load_audio() function supporting wav, flac, mp3 formats
    - Add automatic resampling to 22050 Hz with proper error handling
    - Write unit tests for various audio formats and edge cases
    - _Requirements: 1.1, 1.5, 8.1_

  - [x] 2.2 Implement metadata parsing and validation

    - Create MetadataParser class with JSON schema validation
    - Implement load_metadata() function with proper error messages
    - Add validation for required fields (recording_id, raag, tonic_hz)
    - Write unit tests for metadata validation edge cases
    - _Requirements: 1.2, 8.2_

  - [x] 2.3 Create dataset iteration functionality
    - Implement iter_dataset() function for train/test split traversal
    - Add proper file pairing between audio and metadata files
    - Create DatasetIterator class with error handling for missing files
    - Write integration tests for dataset structure validation
    - _Requirements: 1.3, 8.2_

- [-] 3. Implement pitch extraction with multi-method support

  - [x] 3.1 Create Praat-based pitch extraction

    - Implement extract_pitch_praat() using parselmouth library
    - Configure frame size (0.0464s) and hop size (0.01s) parameters
    - Add voicing probability extraction and thresholding
    - Write unit tests for pitch extraction accuracy and edge cases
    - _Requirements: 2.1, 2.6, 8.3_

  - [x] 3.2 Implement librosa fallback methods

    - Create extract_pitch_librosa() with pyin and yin algorithms
    - Add automatic fallback mechanism when Praat fails
    - Implement consistent output format across all methods
    - Write unit tests for fallback behavior and method consistency
    - _Requirements: 2.2, 8.3_

  - [x] 3.3 Create pitch smoothing and post-processing
    - Implement smooth_pitch() with median and Gaussian filtering
    - Add gap filling for short unvoiced segments using interpolation
    - Create octave error correction using pitch continuity constraints
    - Write unit tests for smoothing effectiveness and parameter sensitivity
    - _Requirements: 2.3, 2.4_

- [x] 4. Implement chromatic quantization and tonic normalization

  - [x] 4.1 Create frequency-to-MIDI conversion utilities

    - Implement hz_to_midi() function with proper logarithmic mapping
    - Add nearest_chromatic_bin() for 36-bin quantization (C3-B5)
    - Create boundary handling for frequencies outside valid range
    - Write unit tests for quantization accuracy and edge cases
    - _Requirements: 3.1, 3.3, 3.5_

  - [x] 4.2 Implement tonic normalization system
    - Create normalize_by_tonic() function shifting tonic to C4 (261.63 Hz)
    - Implement quantize_sequence() combining normalization and binning
    - Add validation for tonic frequency ranges and edge cases
    - Write unit tests for normalization accuracy and consistency
    - _Requirements: 3.2, 3.4_

- [x] 5. Implement discrete HMM with Baum-Welch training

  - [x] 5.1 Create core HMM data structures and initialization

    - Implement DiscreteHMM class with 36 states and 36 observations
    - Add random initialization for transition (A) and emission (B) matrices
    - Create uniform initialization for initial state probabilities (π)
    - Write unit tests for proper stochastic matrix properties
    - _Requirements: 4.2, 4.3_

  - [x] 5.2 Implement forward-backward algorithm with scaling

    - Create forward_backward_scaled() function preventing numerical underflow
    - Implement scaling coefficients computation and log-likelihood calculation
    - Add proper handling of zero probabilities and edge cases
    - Write unit tests for numerical stability and correctness
    - _Requirements: 4.3, 4.4, 8.3_

  - [x] 5.3 Implement Baum-Welch parameter updates

    - Create update_parameters() function for EM M-step
    - Implement efficient computation of sufficient statistics
    - Add Dirichlet regularization (α=0.01) and probability floors (1e-8)
    - Write unit tests for parameter update correctness and convergence
    - _Requirements: 4.3, 4.5_

  - [x] 5.4 Create training loop with convergence monitoring
    - Implement train_hmm_discrete() with iteration limit (200) and tolerance (0.1)
    - Add convergence detection based on log-likelihood improvement
    - Create training progress logging and early stopping
    - Write unit tests for convergence behavior and edge cases
    - _Requirements: 4.4, 4.5_

- [x] 6. Implement model training pipeline for multiple raag classes

  - [x] 6.1 Create per-raag model training system

    - Implement train_all_raag_models() function for batch training
    - Add sequence grouping by raag class from dataset
    - Create separate HMM training for each of 5 raag classes
    - Write integration tests for multi-class training pipeline
    - _Requirements: 4.1, 4.6_

  - [x] 6.2 Implement model persistence and metadata storage
    - Create model serialization using joblib for HMM parameters
    - Add JSON metadata storage with training statistics and hyperparameters
    - Implement proper file organization in models/ directory
    - Write unit tests for serialization/deserialization consistency
    - _Requirements: 4.6_

- [x] 7. Implement classification and inference system

  - [x] 7.1 Create model loading and validation

    - Implement model deserialization with error handling for corrupted files
    - Add metadata validation and version compatibility checking
    - Create ModelLoader class with caching for performance
    - Write unit tests for model loading edge cases and validation
    - _Requirements: 5.5, 8.4_

  - [x] 7.2 Implement forward algorithm scoring for classification
    - Create score_sequence() function using forward algorithm
    - Implement predict_raag() with argmax over all model scores
    - Add confidence score computation and ranking for all classes
    - Write unit tests for scoring accuracy and consistency
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 8. Implement comprehensive evaluation system

  - [x] 8.1 Create accuracy metrics computation

    - Implement overall accuracy calculation across test set
    - Add per-class accuracy computation for individual raag analysis
    - Create top-k accuracy metrics (especially top-3 as specified)
    - Write unit tests for metrics calculation correctness
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 8.2 Implement confusion matrix and detailed analysis
    - Create confusion matrix generation with proper class ordering
    - Add statistical analysis of classification patterns and errors
    - Implement results export in structured formats (JSON/CSV)
    - Write integration tests for evaluation pipeline completeness
    - _Requirements: 6.4, 6.5_

- [x] 9. Create command-line interface for all operations

  - [x] 9.1 Implement dataset preparation commands

    - Create prepare-dataset CLI command with resampling options
    - Add extract-pitch command for single file processing
    - Implement proper argument parsing and validation using typer
    - Write integration tests for CLI command functionality
    - _Requirements: 7.1, 7.2_

  - [x] 9.2 Implement training and inference commands

    - Create train command for batch model training from dataset
    - Add predict command for single audio file classification
    - Implement evaluate command for comprehensive test set analysis
    - Write end-to-end tests for complete CLI workflow
    - _Requirements: 7.3, 7.4, 7.5_

  - [x] 9.3 Add comprehensive error handling and help system
    - Implement clear error messages for all failure modes
    - Add usage examples and help text for each command
    - Create proper exit codes and logging for automation
    - Write tests for error handling and user experience
    - _Requirements: 7.6, 8.1, 8.2, 8.3, 8.4_

- [ ] 10. Create comprehensive test suite and validation

  - [ ] 10.1 Implement unit tests for all core components

    - Create tests for quantization mapping accuracy and boundary conditions
    - Add tests for HMM numerical stability with extreme values
    - Implement tests for audio processing and format compatibility
    - Write tests for configuration validation and parameter bounds
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 10.2 Create integration tests with synthetic data
    - Generate synthetic pitch sequences with known patterns
    - Implement end-to-end training and prediction on synthetic data
    - Add reproducibility tests with fixed random seeds
    - Create performance benchmarks for speed and memory usage
    - _Requirements: All requirements validation_

- [ ] 11. Finalize project with documentation and examples

  - [ ] 11.1 Create comprehensive documentation and usage examples

    - Write README with installation instructions and quick start guide
    - Add API documentation for all public functions and classes
    - Create example scripts demonstrating typical usage patterns
    - Document configuration options and customization possibilities

  - [ ] 11.2 Add example dataset and demonstration
    - Create minimal example dataset with proper structure
    - Add demonstration script showing complete workflow
    - Implement validation script for installation verification
    - Create troubleshooting guide for common issues
