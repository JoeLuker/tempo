"""TEMPO Extension System - Ultra-simple approach.

Extensions are just functions: state -> state

No registry. No decorators. No magic. Just a list of functions.
"""

from .ultra_simple import (
    GenState,
    Extension,
    run_extensions,
    # Built-in extensions
    confidence_surf,
    track_genealogy,
    watch_entropy,
    collect_pruned,
    # Configuration factories
    make_confidence_surf,
)

__all__ = [
    # Core
    'GenState',
    'Extension',
    'run_extensions',
    # Built-in extensions
    'confidence_surf',
    'track_genealogy',
    'watch_entropy',
    'collect_pruned',
    # Factories
    'make_confidence_surf',
]
