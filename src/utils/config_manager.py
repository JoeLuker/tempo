"""
Configuration Manager for TEMPO project.

This module provides a comprehensive, centralized configuration system for the entire
TEMPO project. It supports loading settings from environment variables, files, and
default values, with proper validation and documentation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List, Set, Literal
from pathlib import Path
from dataclasses import dataclass, field, asdict


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    pass


@dataclass
class LoggingConfig:
    """Configuration settings for logging."""

    enable_file_logging: bool = True
    log_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "logs"))
    log_level: str = "INFO"
    console_logging: bool = True

    def __post_init__(self):
        """Create log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.log_level}. Must be one of {valid_levels}"
            )

        self.log_level = self.log_level.upper()


@dataclass
class ModelConfig:
    """Configuration settings for model loading and inference."""

    model_id: str = "deepcogito/cogito-v1-preview-llama-3B"
    device: Optional[str] = None  # None means auto-detect
    quantization: Optional[str] = None  # None, "4bit", "8bit"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    revision: Optional[str] = None
    low_cpu_mem_usage: bool = True
    torch_dtype: Optional[str] = None  # None, "float16", "bfloat16", "float32"

    def __post_init__(self):
        """Validate model configuration."""
        # Validate quantization
        valid_quantization = [None, "4bit", "8bit"]
        if self.quantization not in valid_quantization:
            raise ConfigurationError(
                f"Invalid quantization: {self.quantization}. Must be one of {valid_quantization}"
            )

        # Validate torch_dtype
        valid_dtypes = [None, "float16", "bfloat16", "float32"]
        if self.torch_dtype not in valid_dtypes:
            raise ConfigurationError(
                f"Invalid torch_dtype: {self.torch_dtype}. Must be one of {valid_dtypes}"
            )


@dataclass
class GenerationConfig:
    """Configuration settings for text generation."""

    max_length: int = 200
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    beam_width: int = 1  # 1 means greedy decoding
    use_dynamic_thresholding: bool = True
    use_retroactive_pruning: bool = True
    use_parallel_generation: bool = True
    max_parallel_tokens: int = 5

    def __post_init__(self):
        """Validate generation configuration."""
        if self.top_k < 1:
            raise ConfigurationError(f"top_k must be >= 1, got {self.top_k}")

        if not 0 < self.top_p <= 1:
            raise ConfigurationError(f"top_p must be in (0, 1], got {self.top_p}")

        if self.temperature <= 0:
            raise ConfigurationError(f"temperature must be > 0, got {self.temperature}")

        if self.max_parallel_tokens < 1:
            raise ConfigurationError(
                f"max_parallel_tokens must be >= 1, got {self.max_parallel_tokens}"
            )


@dataclass
class ApiConfig:
    """Configuration settings for API."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    debug: bool = False
    enable_docs: bool = True
    api_version: str = "v2"
    max_concurrent_requests: int = 5

    def __post_init__(self):
        """Validate API configuration."""
        if self.port < 1 or self.port > 65535:
            raise ConfigurationError(
                f"port must be between 1 and 65535, got {self.port}"
            )


@dataclass
class DebugConfig:
    """Configuration settings for debugging."""

    global_debug: bool = False
    module_debug: Dict[str, bool] = field(default_factory=dict)

    def is_debug_enabled(self, module_name: str) -> bool:
        """
        Check if debug is enabled for a module.

        Args:
            module_name: Name of the module

        Returns:
            bool: Whether debug is enabled for the module
        """
        # First check environment variable
        env_var = f"TEMPO_DEBUG_{module_name.upper()}"
        if env_var in os.environ:
            return os.environ[env_var].lower() in ["true", "1", "yes"]

        # Then check module_debug dict
        if module_name in self.module_debug:
            return self.module_debug[module_name]

        # Fall back to global debug setting
        return self.global_debug


@dataclass
class TempoConfig:
    """
    Central configuration class for TEMPO project.

    This class holds all configuration settings for the TEMPO project in a structured way.
    """

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_env(cls) -> "TempoConfig":
        """
        Create a configuration instance from environment variables.

        Returns:
            TempoConfig: Configuration instance with values from environment variables
        """
        config = cls()

        # Process environment variables
        for env_name, env_value in os.environ.items():
            if not env_name.startswith("TEMPO_"):
                continue

            # Skip debug variables as they're handled specially
            if env_name.startswith("TEMPO_DEBUG_"):
                continue

            # Handle global debug setting
            if env_name == "TEMPO_DEBUG":
                config.debug.global_debug = env_value.lower() in ["true", "1", "yes"]
                continue

            # Extract configuration section and key
            parts = env_name.replace("TEMPO_", "", 1).lower().split("_", 1)
            if len(parts) != 2:
                continue

            section, key = parts

            # Update configuration based on section
            if hasattr(config, section) and hasattr(getattr(config, section), key):
                section_obj = getattr(config, section)

                # Convert value to appropriate type based on the field's current type
                current_value = getattr(section_obj, key)
                if isinstance(current_value, bool):
                    new_value = env_value.lower() in ["true", "1", "yes"]
                elif isinstance(current_value, int):
                    new_value = int(env_value)
                elif isinstance(current_value, float):
                    new_value = float(env_value)
                elif isinstance(current_value, list):
                    new_value = env_value.split(",")
                else:
                    new_value = env_value

                setattr(section_obj, key, new_value)

        # Handle module-specific debug settings from environment
        for env_name, env_value in os.environ.items():
            if env_name.startswith("TEMPO_DEBUG_"):
                module_name = env_name.replace("TEMPO_DEBUG_", "", 1).lower()
                debug_value = env_value.lower() in ["true", "1", "yes"]
                config.debug.module_debug[module_name] = debug_value

        return config

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "TempoConfig":
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file

        Returns:
            TempoConfig: Configuration instance with values from the file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with file_path.open("r") as f:
                data = json.load(f)

            config = cls()

            # Update each section
            for section_name, section_data in data.items():
                if not hasattr(config, section_name):
                    continue

                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

            return config
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dict[str, Any]: Configuration as a nested dictionary
        """
        return {
            "logging": asdict(self.logging),
            "model": asdict(self.model),
            "generation": asdict(self.generation),
            "api": asdict(self.api),
            "debug": {
                "global_debug": self.debug.global_debug,
                "module_debug": self.debug.module_debug,
            },
        }

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.

        Args:
            file_path: Path to save the configuration file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_debug_mode(self, module_name: str) -> bool:
        """
        Get debug mode for a specific module.

        Args:
            module_name: Name of the module

        Returns:
            bool: Whether debug is enabled for the module
        """
        return self.debug.is_debug_enabled(module_name)


# Global configuration instance
# This will be initialized with defaults and environment variables
config = TempoConfig.from_env()


# Backwards compatibility functions
def get_debug_mode(module_name: str) -> bool:
    """
    Get debug mode for a specific module.
    Compatibility function for the old config module.

    Args:
        module_name: Name of the module

    Returns:
        bool: Whether debug is enabled for the module
    """
    return config.get_debug_mode(module_name)


# Backwards compatibility constants
DEFAULT_DEBUG_MODE = config.debug.global_debug
ENABLE_FILE_LOGGING = config.logging.enable_file_logging
LOG_DIR = config.logging.log_dir
