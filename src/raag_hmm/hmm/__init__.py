"""
Hidden Markov Model module.

Discrete HMM implementation with Baum-Welch training and forward algorithm inference.
"""

from .model import DiscreteHMM

__all__ = [
    "DiscreteHMM"
]