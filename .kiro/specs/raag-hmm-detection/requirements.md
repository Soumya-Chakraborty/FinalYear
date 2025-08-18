# Requirements Document

## Introduction

This feature implements an end-to-end Python system for raag (raga) detection in Indian classical music using Hidden Markov Models with discrete emissions. The system extracts pitch contours from audio recordings, quantizes them to chromatic pitches, normalizes by tonic, and trains per-raag HMM models to classify musical pieces. The implementation is based on research showing 81.25% accuracy using HMMs trained on chromatic-quantized pitch contours that learn characteristic note transitions.

## Requirements

### Requirement 1

**User Story:** As a music researcher, I want to load and preprocess audio files with their metadata, so that I can extract standardized pitch information for raag analysis.

#### Acceptance Criteria

1. WHEN an audio file (wav, flac, mp3) is provided THEN the system SHALL load it at 22050 Hz sample rate in mono
2. WHEN metadata JSON is provided with tonic_hz THEN the system SHALL parse and validate the tonic frequency
3. WHEN dataset directory is provided THEN the system SHALL iterate through train/test splits with proper file pairing
4. IF audio format is unsupported THEN the system SHALL raise a clear error message
5. WHEN sample rate differs from 22050 Hz THEN the system SHALL resample automatically

### Requirement 2

**User Story:** As a music researcher, I want to extract accurate pitch contours from audio recordings, so that I can analyze the melodic content for raag identification.

#### Acceptance Criteria

1. WHEN audio is processed THEN the system SHALL extract pitch using Praat parselmouth as primary method
2. IF Praat fails THEN the system SHALL fall back to librosa pyin or yin methods
3. WHEN pitch extraction completes THEN the system SHALL apply voicing probability thresholding
4. WHEN raw pitch is extracted THEN the system SHALL apply median/gaussian smoothing to reduce octave errors
5. WHEN short unvoiced segments exist THEN the system SHALL fill gaps using interpolation
6. WHEN frame size is 0.0464 seconds and hop size is 0.01 seconds THEN the system SHALL maintain these timing parameters

### Requirement 3

**User Story:** As a music researcher, I want to quantize pitch contours to chromatic bins normalized by tonic, so that I can create standardized discrete sequences for HMM training.

#### Acceptance Criteria

1. WHEN pitch frequencies are provided THEN the system SHALL map them to 36 chromatic bins spanning C3 to B5
2. WHEN tonic frequency is known THEN the system SHALL normalize pitch by shifting tonic to C4 (261.63 Hz)
3. WHEN quantization is performed THEN the system SHALL use nearest chromatic pitch mapping in 12-TET
4. WHEN output is generated THEN the system SHALL produce integer sequences in range [0, 35]
5. IF pitch is outside C3-B5 range THEN the system SHALL clamp to nearest valid bin

### Requirement 4

**User Story:** As a music researcher, I want to train discrete HMM models for each raag class, so that I can learn characteristic note transition patterns.

#### Acceptance Criteria

1. WHEN training data is provided THEN the system SHALL create separate HMM models for each raag (Bihag, Darbari, Desh, Gaud_Malhar, Yaman)
2. WHEN HMM is initialized THEN the system SHALL use 36 hidden states and 36 observation symbols
3. WHEN training begins THEN the system SHALL use Baum-Welch algorithm with scaling for numerical stability
4. WHEN convergence is checked THEN the system SHALL stop after 200 iterations or 0.1 tolerance improvement
5. WHEN regularization is applied THEN the system SHALL use Dirichlet alpha 0.01 and probability floor 1e-08
6. WHEN training completes THEN the system SHALL save models in joblib format with JSON metadata

### Requirement 5

**User Story:** As a music researcher, I want to classify unknown audio recordings by raag, so that I can automatically identify the musical mode being performed.

#### Acceptance Criteria

1. WHEN test audio is provided THEN the system SHALL extract and quantize pitch using same preprocessing pipeline
2. WHEN classification is performed THEN the system SHALL compute log-likelihood for each trained raag model
3. WHEN scoring completes THEN the system SHALL predict raag as argmax over all model scores
4. WHEN prediction is made THEN the system SHALL return confidence scores for all raag classes
5. IF models are missing THEN the system SHALL raise appropriate error with clear message

### Requirement 6

**User Story:** As a music researcher, I want to evaluate model performance with comprehensive metrics, so that I can assess the accuracy and reliability of raag detection.

#### Acceptance Criteria

1. WHEN evaluation is performed THEN the system SHALL compute overall accuracy across test set
2. WHEN per-class analysis is needed THEN the system SHALL report accuracy for each raag individually
3. WHEN detailed analysis is required THEN the system SHALL generate confusion matrix showing classification patterns
4. WHEN top-k evaluation is performed THEN the system SHALL compute top-3 accuracy metrics
5. WHEN evaluation completes THEN the system SHALL save results in structured format (JSON/CSV)

### Requirement 7

**User Story:** As a developer, I want a command-line interface for all system operations, so that I can easily integrate the tool into workflows and scripts.

#### Acceptance Criteria

1. WHEN dataset preparation is needed THEN the system SHALL provide prepare-dataset command with resampling
2. WHEN pitch extraction is needed THEN the system SHALL provide extract-pitch command for single files
3. WHEN model training is required THEN the system SHALL provide train command for batch processing
4. WHEN prediction is needed THEN the system SHALL provide predict command for single audio files
5. WHEN evaluation is required THEN the system SHALL provide evaluate command for test set analysis
6. WHEN any command fails THEN the system SHALL provide clear error messages and usage help

### Requirement 8

**User Story:** As a developer, I want robust error handling and validation, so that the system fails gracefully with informative messages.

#### Acceptance Criteria

1. WHEN invalid audio files are provided THEN the system SHALL raise specific audio format errors
2. WHEN missing metadata is encountered THEN the system SHALL raise clear validation errors
3. WHEN numerical instability occurs THEN the system SHALL handle scaling issues gracefully
4. WHEN file I/O fails THEN the system SHALL provide clear path and permission error messages
5. WHEN configuration is invalid THEN the system SHALL validate parameters and suggest corrections