#!/usr/bin/env python3
"""
Run TEMPO experiments with data capture.

This script loads experiment configurations from YAML files and runs
them with full data capture for mechanistic interpretability analysis.
"""

import argparse
import yaml
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.experiments.experiment_runner import ExperimentRunner
from src.modeling.model_wrapper import TEMPOModelWrapper
from src.utils.model_utils import get_best_device, get_device_dtype


def load_experiment_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with experiment configuration
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run TEMPO experiments with data capture"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepcogito/cogito-v1-preview-llama-3B",
        help="Model to use for experiments"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed logging"
    )

    args = parser.parse_args()

    # Load experiment configuration
    config_path = Path(args.config)
    print(f"Loading experiment configuration from: {config_path}")
    experiment_config = load_experiment_config(config_path)

    experiment_name = experiment_config.get('experiment_name', 'unnamed')
    print(f"Running experiment: {experiment_name}")
    print(f"Prompt: {experiment_config.get('prompt', 'N/A')}")
    print(f"Max tokens: {experiment_config.get('max_tokens', 'N/A')}")

    # Determine device
    device_str = get_best_device()
    dtype = get_device_dtype(device_str)
    print(f"Using device: {device_str} with dtype: {dtype}")

    # Load model and tokenizer
    model_id = args.model
    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_str,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Use eager for stability
    )

    # Wrap model
    model_wrapper = TEMPOModelWrapper(
        model=model,
        device=device_str
    )

    # Set debug mode if requested
    if args.debug:
        model_wrapper.set_debug_mode(True)

    print(f"Model loaded on device: {model_wrapper.device}")

    # Create experiment runner
    runner = ExperimentRunner(
        model=model_wrapper,
        tokenizer=tokenizer,
        device=device_str
    )
    runner.debug_mode = args.debug

    # Check if we need to run both isolation modes
    run_both_modes = experiment_config.get('run_both_modes', False)

    if run_both_modes:
        print("\n=== Running ISOLATED mode ===")
        isolated_config = experiment_config.copy()
        isolated_config['allow_intraset_token_visibility'] = False
        isolated_config['experiment_name'] = f"{experiment_name}_isolated"
        isolated_config['output_dir'] = f"./experiments/output/{experiment_name}_isolated"

        print(f"Output directory: {isolated_config['output_dir']}")
        result_isolated = runner.run_experiment(isolated_config)

        print(f"\nIsolated mode completed:")
        print(f"  Generation time: {result_isolated.get('generation_time', 0):.4f}s")
        print(f"  Raw text: {result_isolated.get('raw_generated_text', '')[:100]}...")

        print("\n=== Running VISIBLE mode ===")
        visible_config = experiment_config.copy()
        visible_config['allow_intraset_token_visibility'] = True
        visible_config['experiment_name'] = f"{experiment_name}_visible"
        visible_config['output_dir'] = f"./experiments/output/{experiment_name}_visible"

        print(f"Output directory: {visible_config['output_dir']}")
        result_visible = runner.run_experiment(visible_config)

        print(f"\nVisible mode completed:")
        print(f"  Generation time: {result_visible.get('generation_time', 0):.4f}s")
        print(f"  Raw text: {result_visible.get('raw_generated_text', '')[:100]}...")

        print("\n=== Comparison ===")
        print(f"Time difference: {abs(result_isolated.get('generation_time', 0) - result_visible.get('generation_time', 0)):.4f}s")

        # Check if outputs are identical
        isolated_text = result_isolated.get('raw_generated_text', '')
        visible_text = result_visible.get('raw_generated_text', '')
        if isolated_text == visible_text:
            print("✓ Outputs are IDENTICAL")
        else:
            print("✗ Outputs DIFFER")
            print(f"  Isolated: {isolated_text[:50]}...")
            print(f"  Visible:  {visible_text[:50]}...")

    else:
        # Single mode run
        print(f"\nRunning experiment with settings from config...")
        isolation_mode = "isolated" if not experiment_config.get('allow_intraset_token_visibility', False) else "visible"
        print(f"Isolation mode: {isolation_mode}")

        # Set output directory if not specified
        if 'output_dir' not in experiment_config:
            experiment_config['output_dir'] = f"./experiments/output/{experiment_name}"

        print(f"Output directory: {experiment_config['output_dir']}")

        result = runner.run_experiment(experiment_config)

        print(f"\nExperiment completed:")
        print(f"  Generation time: {result.get('generation_time', 0):.4f}s")
        print(f"  Raw text: {result.get('raw_generated_text', '')[:100]}...")

    print("\n✓ Experiment finished successfully")


if __name__ == "__main__":
    main()
