#!/usr/bin/env python3
"""
Configuration system demo.

This script demonstrates how to use the TEMPO configuration system.
"""

import os
import json
from pathlib import Path
from src.utils import config, TempoConfig


def main():
    """
    Demonstrate the usage of the configuration system.
    """
    # Print the default configuration loaded from environment variables
    print("Current configuration:")
    print(json.dumps(config.to_dict(), indent=2))

    print("\n" + "=" * 50 + "\n")

    # Demonstrate how to access configuration values
    print(f"Model ID: {config.model.model_id}")
    print(f"Debug mode for token_generator: {config.get_debug_mode('token_generator')}")
    print(f"API port: {config.api.port}")

    print("\n" + "=" * 50 + "\n")

    # Demonstrate how to modify configuration
    print("Modifying configuration...")
    config.model.model_id = "meta-llama/Llama-2-7b-chat-hf"
    config.generation.max_length = 500
    config.debug.module_debug["token_generator"] = True

    print("Updated configuration:")
    print(f"Model ID: {config.model.model_id}")
    print(f"Max length: {config.generation.max_length}")
    print(f"Debug mode for token_generator: {config.get_debug_mode('token_generator')}")

    print("\n" + "=" * 50 + "\n")

    # Demonstrate saving to file
    config_path = Path("./temp_config.json")
    print(f"Saving configuration to {config_path}...")
    config.save_to_file(config_path)
    print("Configuration saved.")

    print("\n" + "=" * 50 + "\n")

    # Demonstrate loading from file
    print(f"Loading configuration from {config_path}...")
    loaded_config = TempoConfig.from_file(config_path)
    print(f"Loaded model ID: {loaded_config.model.model_id}")

    # Clean up
    config_path.unlink()
    print("Temporary configuration file deleted.")

    print("\n" + "=" * 50 + "\n")

    # Demonstrate environment variable overrides
    print("Setting environment variables and reloading config...")
    os.environ["TEMPO_MODEL_MODEL_ID"] = "huggyllama/llama-7b"
    os.environ["TEMPO_GENERATION_MAX_LENGTH"] = "300"
    os.environ["TEMPO_DEBUG"] = "true"

    # Create a new config instance from environment
    env_config = TempoConfig.from_env()
    print(f"Environment-loaded model ID: {env_config.model.model_id}")
    print(f"Environment-loaded max length: {env_config.generation.max_length}")
    print(f"Environment-loaded global debug: {env_config.debug.global_debug}")


if __name__ == "__main__":
    main()
