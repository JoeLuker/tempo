"""
Configuration module for TEMPO project.

This module provides centralized configuration settings for all modules
in the project, including debug and logging settings.
"""

import os

# Default debug mode setting - can be overridden by environment variable
DEFAULT_DEBUG_MODE = os.environ.get("TEMPO_DEBUG", "true").lower() in [
    "true",
    "1",
    "yes",
]

# File logging settings
ENABLE_FILE_LOGGING = True
LOG_DIR = os.path.join(os.getcwd(), "logs")

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Per-module debug settings - these allow fine-grained control
# Defaults to DEFAULT_DEBUG_MODE if not specified
MODULE_DEBUG_SETTINGS = {
    # Core modules
    "token_generator": DEFAULT_DEBUG_MODE,
    "attention_manager": DEFAULT_DEBUG_MODE,
    "rope_modifier": DEFAULT_DEBUG_MODE,
    "model_wrapper": DEFAULT_DEBUG_MODE,
    "token_selector": DEFAULT_DEBUG_MODE,
    "parallel_generator": DEFAULT_DEBUG_MODE,
    # Other modules
    "experiment_runner": DEFAULT_DEBUG_MODE,
    "retroactive_pruner": DEFAULT_DEBUG_MODE,
    "mcts_generator": DEFAULT_DEBUG_MODE,
}


def get_debug_mode(module_name: str) -> bool:
    """
    Get the debug mode setting for a specific module.

    Args:
        module_name: Name of the module

    Returns:
        bool: Whether debug mode is enabled for this module
    """
    # First check if there's a specific environment variable for this module
    env_var = f"TEMPO_DEBUG_{module_name.upper()}"
    if env_var in os.environ:
        return os.environ[env_var].lower() in ["true", "1", "yes"]

    # Then check if there's a setting in MODULE_DEBUG_SETTINGS
    if module_name in MODULE_DEBUG_SETTINGS:
        return MODULE_DEBUG_SETTINGS[module_name]

    # Fall back to default debug mode
    return DEFAULT_DEBUG_MODE
