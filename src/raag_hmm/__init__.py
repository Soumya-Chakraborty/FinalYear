"""
RaagHMM: Hidden Markov Model-based Raag Detection System

A Python library for automatic raag (raga) detection in Indian classical music
using Hidden Markov Models with discrete emissions.
"""

__version__ = "0.1.0"
__author__ = "RaagHMM Development Team"

from .config import get_config, set_config
from .logger import get_logger

__all__ = [
    "get_config",
    "set_config", 
    "get_logger",
    "__version__"
]