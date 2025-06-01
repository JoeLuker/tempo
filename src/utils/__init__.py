"""
Utilities module for TEMPO project.

This package provides common utilities used throughout the TEMPO project.
"""

# Re-export the configuration manager for easy imports
from src.utils.config_manager import config, TempoConfig, get_debug_mode

__all__ = ["config", "TempoConfig", "get_debug_mode"]
