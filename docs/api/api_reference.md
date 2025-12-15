# RaagHMM API Reference

This document provides a comprehensive reference for the RaagHMM public API.

## Table of Contents

1. [Training Module](#training-module)
2. [Inference Module](#inference-module)
3. [HMM Module](#hmm-module)
4. [Pitch Module](#pitch-module)
5. [Quantization Module](#quantization-module)
6. [IO Module](#io-module)
7. [Evaluation Module](#evaluation-module)

## Training Module

### RaagTrainer

The main class for training HMM models for multiple raag classes.

```python
class RaagTrainer(
    n_states: int = 36,
    n_observations: int = 36,
    max_iterations: int = 200,
    convergence_tolerance: float = 0.1,
    regularization_alpha: float = 0.01,
    probability_floor: float = 1e-8,
    random_state: Optional[int] = None
)
```

**Parameters:**
- `n_states`: Number of HMM hidden states
- `n_observations`: Number of observation symbols
- `max_iterations`: Maximum Baum-Welch iterations
- `convergence_tolerance`: Log-likelihood improvement threshold
- `regularization_alpha`: Dirichlet regularization parameter
- `probability_floor`: Minimum probability value
- `random_state`: Random seed for reproducible initialization

**Methods:**

#### `train_all_raag_models`
```python
def train_all_raag_models(
    self, 
    dataset_root: str, 
    split: str = "train", 
    verbose: bool = False
) -> Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]]
```
Train HMM models for all raag classes in the dataset.

**Parameters:**
- `dataset_root`: Root directory of the dataset
- `split`: Dataset split to use for training
- `verbose`: Whether to print detailed training progress

**Returns:** Dictionary mapping raag names to (model, metadata) tuples

#### `extract_and_quantize_sequence`
```python
def extract_and_quantize_sequence(
    self,
    audio_path: str,
    tonic_hz: float,
    frame_sec: float = 0.0464,
    hop_sec: float = 0.01
) -> np.ndarray
```
Extract pitch from audio and quantize to chromatic sequence.

**Parameters:**
- `audio_path`: Path to audio file
- `tonic_hz`: Tonic frequency for normalization
- `frame_sec`: Frame size in seconds
- `hop_sec`: Hop size in seconds

**Returns:** Quantized chromatic sequence as integer array

---

### ModelPersistence

Handles model serialization and deserialization.

```python
class ModelPersistence(models_dir: str = "models")
```

**Parameters:**
- `models_dir`: Directory to store models and metadata

**Methods:**

#### `save_model`
```python
def save_model(
    self,
    raag_name: str,
    model: DiscreteHMM,
    metadata: Dict[str, Any],
    overwrite: bool = False
) -> Tuple[str, str]
```
Save HMM model and metadata to disk.

**Returns:** Tuple of (model_path, metadata_path)

#### `load_model`
```python
def load_model(self, raag_name: str) -> Tuple[DiscreteHMM, Dict[str, Any]]
```
Load HMM model and metadata from disk.

**Returns:** Tuple of (model, metadata)

#### `save_all_models`
```python
def save_all_models(
    self,
    trained_models: Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]],
    overwrite: bool = False
) -> Dict[str, Tuple[str, str]]
```
Save all trained models and their metadata.

#### `load_all_models`
```python
def load_all_models(self) -> Dict[str, Tuple[DiscreteHMM, Dict[str, Any]]]
```
Load all available models from the models directory.

---

## Inference Module

### RaagClassifier

Multi-model inference engine for raag classification.

```python
class RaagClassifier(models_dir: str = "models", use_cache: bool = True)
```

**Parameters:**
- `models_dir`: Directory containing trained models
- `use_cache`: Whether to use model caching

**Methods:**

#### `predict_raag`
```python
def predict_raag(
    self, 
    sequence: np.ndarray, 
    return_all_scores: bool = False
) -> Union[str, Tuple[str, Dict[str, float]]]
```
Predict raag class using argmax over all model scores.

**Parameters:**
- `sequence`: Quantized pitch sequence
- `return_all_scores`: Whether to return scores for all classes

#### `predict_with_confidence`
```python
def predict_with_confidence(
    self, 
    sequence: np.ndarray, 
    normalize_scores: bool = True
) -> Dict[str, Any]
```
Predict raag with confidence scores and ranking for all classes.

**Returns:** Dictionary with prediction results and confidence metrics.

#### `score_sequence`
```python
def score_sequence(self, sequence: np.ndarray, raag_name: str) -> float
```
Compute log-likelihood score for a sequence using specific raag model.

---

## HMM Module

### DiscreteHMM

Discrete Hidden Markov Model implementation.

```python
class DiscreteHMM(
    n_states: int = 36, 
    n_observations: int = 36, 
    random_state: Optional[int] = None
)
```

**Parameters:**
- `n_states`: Number of hidden states
- `n_observations`: Number of observation symbols
- `random_state`: Random seed for initialization

**Methods:**

#### `train`
```python
def train(
    self,
    observations_list: list,
    max_iterations: int = 200,
    convergence_tolerance: float = 0.1,
    regularization_alpha: float = 0.01,
    probability_floor: float = 1e-8,
    verbose: bool = False
) -> dict
```
Train HMM using Baum-Welch algorithm with convergence monitoring.

**Returns:** Dictionary with training statistics.

#### `score`
```python
def score(self, observations: np.ndarray) -> float
```
Compute log-likelihood of observation sequence using forward algorithm.

#### `forward_backward_scaled`
```python
def forward_backward_scaled(
    self, 
    observations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]
```
Compute forward-backward algorithm with scaling to prevent numerical underflow.

---

## Pitch Module

### Pitch Extraction Functions

#### `extract_pitch_with_fallback`
```python
def extract_pitch_with_fallback(
    y: np.ndarray,
    sr: int,
    frame_sec: float = 0.0464,
    hop_sec: float = 0.01,
    voicing_threshold: float = 0.5,
    primary_method: str = 'praat',
    fallback_methods: list = None
) -> Tuple[np.ndarray, np.ndarray]
```
Extract pitch with automatic fallback between methods.

**Returns:** Tuple of (f0_hz, voicing_prob) arrays

### Pitch Smoothing

#### `smooth_pitch`
```python
def smooth_pitch(
    f0_hz: np.ndarray,
    voicing_prob: Optional[np.ndarray] = None,
    hop_sec: float = 0.01,
    median_window: int = 5,
    gaussian_sigma: float = 1.0,
    gap_fill_threshold_ms: float = 100.0,
    octave_tolerance: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]
```
Convenience function for pitch smoothing with default parameters.

---

## Quantization Module

### Chromatic Quantization

#### `hz_to_midi`
```python
def hz_to_midi(f_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]
```
Convert frequency in Hz to MIDI note number.

#### `quantize_sequence`
```python
def quantize_sequence(
    f0_hz: np.ndarray,
    tonic_hz: float,
    n_bins: int = 36,
    base_midi: float = 48
) -> np.ndarray
```
Quantize a pitch sequence by normalizing tonic and binning to chromatic scale.

**Returns:** Quantized sequence as integer array [0, n_bins-1]

---

## IO Module

### Audio Loading

#### `load_audio`
```python
def load_audio(path: str, sr: int = 22050) -> np.ndarray
```
Load audio file with automatic resampling to target sample rate.

### Dataset Iteration

#### `iter_dataset`
```python
def iter_dataset(root: str, split: str = "train") -> Iterator[Tuple[str, Dict[str, Any]]]
```
Iterate through dataset files for specified split.

### Metadata Loading

#### `load_metadata`
```python
def load_metadata(path: str) -> Dict[str, Any]
```
Load and validate metadata from JSON file.

---

## Evaluation Module

### Metrics Functions

#### `compute_comprehensive_metrics`
```python
def compute_comprehensive_metrics(
    y_true: List[str], 
    y_pred: List[str],
    y_scores: Optional[List[Dict[str, float]]] = None,
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, Any]
```
Compute comprehensive accuracy metrics in a single function call.

#### `compute_confusion_matrix`
```python
def compute_confusion_matrix(
    y_true: List[str], 
    y_pred: List[str],
    class_order: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]
```
Create confusion matrix with proper class ordering.

#### `analyze_confusion_matrix`
```python
def analyze_confusion_matrix(
    confusion_matrix: np.ndarray, 
    class_names: List[str]
) -> Dict[str, Any]
```
Add statistical analysis of classification patterns and errors.

#### `export_classification_report`
```python
def export_classification_report(
    y_true: List[str], 
    y_pred: List[str],
    output_path: str, 
    format: str = 'json',
    class_order: Optional[List[str]] = None
) -> None
```
Export comprehensive classification report in structured format.