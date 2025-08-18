"""
Tests for configuration management system.
"""

import pytest
import json
from pathlib import Path

from raag_hmm.config import (
    get_config, set_config, update_config, 
    load_config_file, save_config_file, 
    get_all_config, reset_config
)


def test_default_config():
    """Test that default configuration is loaded correctly."""
    # Test audio config
    assert get_config('audio', 'sample_rate') == 22050
    assert get_config('audio', 'channels') == 1
    assert 'wav' in get_config('audio', 'supported_formats')
    
    # Test HMM config
    assert get_config('hmm', 'n_states') == 36
    assert get_config('hmm', 'max_iterations') == 200
    assert get_config('hmm', 'convergence_tolerance') == 0.1


def test_get_config_section():
    """Test getting entire configuration sections."""
    audio_config = get_config('audio')
    assert isinstance(audio_config, dict)
    assert 'sample_rate' in audio_config
    assert 'channels' in audio_config


def test_set_config():
    """Test setting individual configuration values."""
    # Set a new value
    set_config('audio', 'sample_rate', 44100)
    assert get_config('audio', 'sample_rate') == 44100
    
    # Set value in new section
    set_config('test_section', 'test_key', 'test_value')
    assert get_config('test_section', 'test_key') == 'test_value'


def test_update_config():
    """Test updating configuration with dictionary."""
    update_dict = {
        'audio': {
            'sample_rate': 48000,
            'new_setting': True
        },
        'new_section': {
            'key1': 'value1',
            'key2': 42
        }
    }
    
    update_config(update_dict)
    
    # Check updated values
    assert get_config('audio', 'sample_rate') == 48000
    assert get_config('audio', 'new_setting') is True
    assert get_config('new_section', 'key1') == 'value1'
    assert get_config('new_section', 'key2') == 42
    
    # Check that other values are preserved
    assert get_config('audio', 'channels') == 1


def test_config_file_operations(temp_dir):
    """Test saving and loading configuration files."""
    config_file = temp_dir / "test_config.json"
    
    # Modify configuration
    set_config('audio', 'sample_rate', 48000)
    set_config('test', 'value', 123)
    
    # Save to file
    save_config_file(str(config_file))
    assert config_file.exists()
    
    # Reset and verify it's back to defaults
    reset_config()
    assert get_config('audio', 'sample_rate') == 22050
    assert get_config('test', 'value') is None
    
    # Load from file
    load_config_file(str(config_file))
    assert get_config('audio', 'sample_rate') == 48000
    assert get_config('test', 'value') == 123


def test_get_all_config():
    """Test getting complete configuration dictionary."""
    all_config = get_all_config()
    
    assert isinstance(all_config, dict)
    assert 'audio' in all_config
    assert 'hmm' in all_config
    assert 'logging' in all_config
    
    # Verify it's a copy (modifications don't affect original)
    all_config['audio']['sample_rate'] = 99999
    assert get_config('audio', 'sample_rate') != 99999


def test_reset_config():
    """Test resetting configuration to defaults."""
    # Modify configuration
    set_config('audio', 'sample_rate', 48000)
    set_config('custom', 'key', 'value')
    
    # Reset
    reset_config()
    
    # Verify back to defaults
    assert get_config('audio', 'sample_rate') == 22050
    assert get_config('custom', 'key') is None


def test_invalid_config_file(temp_dir):
    """Test handling of invalid configuration files."""
    # Test non-existent file
    with pytest.raises(ValueError):
        load_config_file(str(temp_dir / "nonexistent.json"))
    
    # Test invalid JSON
    invalid_file = temp_dir / "invalid.json"
    invalid_file.write_text("{ invalid json }")
    
    with pytest.raises(ValueError):
        load_config_file(str(invalid_file))