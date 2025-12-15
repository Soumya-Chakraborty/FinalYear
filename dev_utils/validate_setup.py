#!/usr/bin/env python3
"""
Validation script for RaagHMM project setup.

This script validates that the core project structure and configuration
system are working correctly.
"""

import sys
from pathlib import Path


def test_package_import():
    """Test that the main package can be imported."""
    try:
        import raag_hmm
        print(f"✓ Package import successful (version: {raag_hmm.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Package import failed: {e}")
        return False


def test_configuration_system():
    """Test that the configuration system works."""
    try:
        from raag_hmm import get_config, set_config
        
        # Test getting default values
        sample_rate = get_config('audio', 'sample_rate')
        n_states = get_config('hmm', 'n_states')
        
        if sample_rate != 22050:
            print(f"✗ Unexpected default sample rate: {sample_rate}")
            return False
            
        if n_states != 36:
            print(f"✗ Unexpected default HMM states: {n_states}")
            return False
        
        # Test setting values
        set_config('audio', 'sample_rate', 44100)
        new_rate = get_config('audio', 'sample_rate')
        
        if new_rate != 44100:
            print(f"✗ Config setting failed: expected 44100, got {new_rate}")
            return False
        
        print("✓ Configuration system working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Configuration system failed: {e}")
        return False


def test_logging_system():
    """Test that the logging system works."""
    try:
        from raag_hmm import get_logger
        
        logger = get_logger('validation')
        logger.info("Logging system test")
        
        print("✓ Logging system working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Logging system failed: {e}")
        return False


def test_project_structure():
    """Test that all required directories and files exist."""
    required_files = [
        'raag_hmm/__init__.py',
        'raag_hmm/config.py',
        'raag_hmm/logger.py',
        'raag_hmm/exceptions.py',
        'requirements.txt',
        'setup.py',
        'pyproject.toml',
        'README.md'
    ]
    
    required_dirs = [
        'raag_hmm/io',
        'raag_hmm/pitch', 
        'raag_hmm/quantize',
        'raag_hmm/hmm',
        'raag_hmm/train',
        'raag_hmm/infer',
        'raag_hmm/evaluate',
        'raag_hmm/cli',
        'tests'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
        if missing_dirs:
            print(f"✗ Missing directories: {missing_dirs}")
        return False
    
    print("✓ Project structure complete")
    return True


def test_module_init_files():
    """Test that all module directories have __init__.py files."""
    module_dirs = [
        'raag_hmm/io',
        'raag_hmm/pitch',
        'raag_hmm/quantize', 
        'raag_hmm/hmm',
        'raag_hmm/train',
        'raag_hmm/infer',
        'raag_hmm/evaluate',
        'raag_hmm/cli'
    ]
    
    missing_init = []
    
    for module_dir in module_dirs:
        init_file = Path(module_dir) / '__init__.py'
        if not init_file.exists():
            missing_init.append(str(init_file))
    
    if missing_init:
        print(f"✗ Missing __init__.py files: {missing_init}")
        return False
    
    print("✓ All module __init__.py files present")
    return True


def main():
    """Run all validation tests."""
    print("RaagHMM Project Setup Validation")
    print("=" * 40)
    
    tests = [
        test_project_structure,
        test_module_init_files,
        test_package_import,
        test_configuration_system,
        test_logging_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All validation tests passed! Project setup is complete.")
        return 0
    else:
        print("✗ Some validation tests failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())