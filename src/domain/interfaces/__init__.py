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
from .attention_manager import AttentionManagerInterface
from .data_capture import DataCaptureInterface
from .retroactive_remover import RetroactiveRemoverInterface

__all__ = [
    "TokenGeneratorInterface",
    "TokenizerInterface",
    "AttentionCacheInterface",
    "ModelInterface",
    "CacheManagerInterface",
    "PerformanceTrackerInterface",
    "AttentionManagerInterface",
    "DataCaptureInterface",
    "RetroactiveRemoverInterface"
]