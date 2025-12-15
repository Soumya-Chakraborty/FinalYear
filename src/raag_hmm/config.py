"""
Configuration management system for RaagHMM.

Provides default settings and configuration override capabilities.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


# Default configuration based on design document
DEFAULT_CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'channels': 1,
        'supported_formats': ['wav', 'flac', 'mp3']
    },
    'pitch': {
        'frame_sec': 0.0464,
        'hop_sec': 0.01,
        'voicing_threshold': 0.5,
        'smoothing_window': 5,
        'gaussian_sigma': 1.0,
        'gap_fill_threshold_ms': 100
    },
    'quantization': {
        'n_bins': 36,
        'base_midi': 48,  # C3
        'reference_tonic': 261.63  # C4 in Hz
    },
    'hmm': {
        'n_states': 36,
        'n_observations': 36,
        'max_iterations': 200,
        'convergence_tolerance': 0.1,
        'regularization_alpha': 0.01,
        'probability_floor': 1e-8
    },
    'training': {
        'random_seed': 42,
        'validation_split': 0.2,
        'early_stopping': True
    },
    'evaluation': {
        'top_k_values': [1, 3, 5],
        'confidence_threshold': 0.5
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_logging': False,
        'log_file': 'raag_hmm.log'
    }
}


class ConfigManager:
    """Manages configuration settings with override capabilities."""
    
    def __init__(self):
        self._config = DEFAULT_CONFIG.copy()
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        # Check for config file path in environment
        config_file = os.getenv('RAAG_HMM_CONFIG')
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Override specific settings from environment variables
        env_overrides = {
            'RAAG_HMM_SAMPLE_RATE': ('audio', 'sample_rate', int),
            'RAAG_HMM_MAX_ITERATIONS': ('hmm', 'max_iterations', int),
            'RAAG_HMM_LOG_LEVEL': ('logging', 'level', str),
            'RAAG_HMM_RANDOM_SEED': ('training', 'random_seed', int)
        }
        
        for env_var, (section, key, type_func) in env_overrides.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    self._config[section][key] = type_func(value)
                except (ValueError, KeyError):
                    pass  # Ignore invalid environment values
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Get configuration value(s)."""
        if key is None:
            return self._config.get(section, {})
        return self._config.get(section, {}).get(key)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        for section, values in config_dict.items():
            if section not in self._config:
                self._config[section] = {}
            if isinstance(values, dict):
                self._config[section].update(values)
            else:
                self._config[section] = values
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            self.update(file_config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._config.copy()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = DEFAULT_CONFIG.copy()
        self._load_environment_overrides()


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config(section: str, key: Optional[str] = None) -> Any:
    """Get configuration value(s) from global config manager."""
    return _config_manager.get(section, key)


def set_config(section: str, key: str, value: Any) -> None:
    """Set configuration value in global config manager."""
    _config_manager.set(section, key, value)


def update_config(config_dict: Dict[str, Any]) -> None:
    """Update global configuration with dictionary."""
    _config_manager.update(config_dict)


def load_config_file(config_path: str) -> None:
    """Load configuration from file into global config manager."""
    _config_manager.load_from_file(config_path)


def save_config_file(config_path: str) -> None:
    """Save global configuration to file."""
    _config_manager.save_to_file(config_path)


def get_all_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return _config_manager.get_all()


def reset_config() -> None:
    """Reset global configuration to defaults."""
    _config_manager.reset_to_defaults()