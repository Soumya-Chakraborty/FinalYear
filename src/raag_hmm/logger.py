"""
Logging infrastructure for RaagHMM system.

Provides centralized logging configuration with file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import get_config


class RaagHMMLogger:
    """Centralized logger for RaagHMM system."""
    
    def __init__(self):
        self._loggers = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Configure root logger with settings from config."""
        # Get logging configuration
        log_level = get_config('logging', 'level') or 'INFO'
        log_format = get_config('logging', 'format')
        file_logging = get_config('logging', 'file_logging') or False
        log_file = get_config('logging', 'log_file') or 'raag_hmm.log'
        
        # Configure root logger
        root_logger = logging.getLogger('raag_hmm')
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if file_logging:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate messages
        root_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name."""
        full_name = f'raag_hmm.{name}' if not name.startswith('raag_hmm') else name
        
        if full_name not in self._loggers:
            logger = logging.getLogger(full_name)
            self._loggers[full_name] = logger
        
        return self._loggers[full_name]
    
    def set_level(self, level: str):
        """Set logging level for all loggers."""
        log_level = getattr(logging, level.upper())
        
        # Update root logger
        root_logger = logging.getLogger('raag_hmm')
        root_logger.setLevel(log_level)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
    
    def enable_file_logging(self, log_file: Optional[str] = None):
        """Enable file logging with optional custom log file path."""
        if log_file is None:
            log_file = get_config('logging', 'log_file') or 'raag_hmm.log'
        
        root_logger = logging.getLogger('raag_hmm')
        
        # Check if file handler already exists
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return  # File logging already enabled
        
        # Create file handler
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(root_logger.level)
        
        # Use same formatter as console handler
        if root_logger.handlers:
            formatter = root_logger.handlers[0].formatter
            file_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)
    
    def disable_file_logging(self):
        """Disable file logging by removing file handlers."""
        root_logger = logging.getLogger('raag_hmm')
        
        # Remove file handlers
        handlers_to_remove = [
            h for h in root_logger.handlers 
            if isinstance(h, logging.FileHandler)
        ]
        
        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)
            handler.close()


# Global logger manager instance
_logger_manager = RaagHMMLogger()


def get_logger(name: str = 'main') -> logging.Logger:
    """Get a logger instance for the specified module/component."""
    return _logger_manager.get_logger(name)


def set_log_level(level: str):
    """Set global logging level."""
    _logger_manager.set_level(level)


def enable_file_logging(log_file: Optional[str] = None):
    """Enable file logging globally."""
    _logger_manager.enable_file_logging(log_file)


def disable_file_logging():
    """Disable file logging globally."""
    _logger_manager.disable_file_logging()


# Create convenience loggers for common components
def get_audio_logger() -> logging.Logger:
    """Get logger for audio processing components."""
    return get_logger('audio')


def get_pitch_logger() -> logging.Logger:
    """Get logger for pitch extraction components."""
    return get_logger('pitch')


def get_hmm_logger() -> logging.Logger:
    """Get logger for HMM components."""
    return get_logger('hmm')


def get_training_logger() -> logging.Logger:
    """Get logger for training components."""
    return get_logger('training')


def get_evaluation_logger() -> logging.Logger:
    """Get logger for evaluation components."""
    return get_logger('evaluation')