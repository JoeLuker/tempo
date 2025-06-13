"""Domain interfaces for the TEMPO system.

This package contains the interface definitions (protocols) that must be
implemented by the infrastructure layer.
"""

from .token_generation import TokenGeneratorInterface
from .tokenizer import TokenizerInterface
from .attention_cache import AttentionCacheInterface
from .model import ModelInterface
from .cache_manager import CacheManagerInterface
from .performance_tracker import PerformanceTrackerInterface

__all__ = [
    "TokenGeneratorInterface",
    "TokenizerInterface",
    "AttentionCacheInterface",
    "ModelInterface",
    "CacheManagerInterface",
    "PerformanceTrackerInterface"
]