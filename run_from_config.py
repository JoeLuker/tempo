#!/usr/bin/env python3
"""Run TEMPO from YAML configuration file.

Usage:
    python3 run_from_config.py configs/examples/simple.yaml
    python3 run_from_config.py configs/examples/two-phase.yaml
    python3 run_from_config.py my_config.yaml --output results.json
"""

import sys
import argparse
import torch
import random
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from src.config.schema import TEMPOConfig
from src.experiments import ExperimentRunner
from src.utils.model_utils import get_best_device, get_device_dtype, load_tempo_components


def main():
    """Run TEMPO from YAML config."""
    parser = argparse.ArgumentParser(
        description="Run TEMPO generation from YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file
  python3 run_from_config.py configs/examples/simple.yaml

  # Override output file
  python3 run_from_config.py configs/examples/two-phase.yaml --output my_results.json

  # Enable debug mode
  python3 run_from_config.py configs/examples/multi-phase.yaml --debug

  # Save config after modifications
  python3 run_from_config.py config.yaml --save-config modified.yaml
        """
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Override JSON output file path'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    parser.add_argument(
        '--save-config',
        type=str,
        help='Save the loaded config to a new YAML file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load config and show settings without running'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    try:
        config = TEMPOConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Override settings from command line
    if args.output:
        config.output.json_file = args.output

    if args.debug:
        config.debug.debug_mode = True
        config.debug.verbose = True

    # Generate output filename if not specified
    if config.output.json_output and not config.output.json_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output.json_file = f"output/{config.name}_{timestamp}.json"

    # Ensure output directory exists
    if config.output.json_file:
        Path(config.output.json_file).parent.mkdir(parents=True, exist_ok=True)

    # Save config if requested
    if args.save_config:
        print(f"Saving config to: {args.save_config}")
        config.to_yaml(args.save_config)

    # Print configuration
    print("\n" + "="*60)
    print(f"TEMPO Configuration: {config.name}")
    print("="*60)
    print(f"Description: {config.description}")
    if config.tags:
        print(f"Tags: {', '.join(config.tags)}")
    print()
    print(f"Model: {config.model.name}")
    print(f"Prompt: {config.generation.prompt[:60]}...")
    print(f"Max tokens: {config.generation.max_tokens}")
    print(f"Selection threshold: {config.generation.selection_threshold}")
    print()

    if config.extensions.two_phase:
        if config.extensions.dynamic_phase:
            print(f"Mode: Dynamic two-phase (max_positions={config.extensions.max_positions})")
        else:
            print(f"Mode: Fixed two-phase (phase1_steps={config.extensions.phase1_steps})")
        print(f"Phase 2 threshold: {config.extensions.phase2_threshold}")

    if config.multi_phase.enabled:
        print(f"Mode: Multi-phase ({len(config.multi_phase.phases)} phases)")
        for i, phase in enumerate(config.multi_phase.phases, 1):
            print(f"  Phase {i}: {phase.name} (max_pos={phase.max_positions}, threshold={phase.threshold})")

    if config.pruning.enabled:
        print(f"Pruning: Enabled (threshold={config.pruning.attention_threshold})")

    extensions = []
    if config.extensions.confidence_surfing:
        extensions.append("confidence_surfing")
    if config.extensions.genealogy_tracking:
        extensions.append("genealogy_tracking")
    if config.extensions.entropy_watching:
        extensions.append("entropy_watching")
    if extensions:
        print(f"Extensions: {', '.join(extensions)}")

    print()
    if config.output.json_file:
        print(f"Output: {config.output.json_file}")
    print("="*60)
    print()

    # Dry run exit
    if args.dry_run:
        print("Dry run - exiting without generation")
        return

    # Set random seeds
    seed = config.generation.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine device
    device_str = get_best_device()
    device = torch.device(device_str)
    dtype = get_device_dtype(device_str)
    print(f"Using device: {device} with dtype: {dtype}")
    print()

    # Load model and components
    print("Loading TEMPO components...")
    load_start = time.time()

    components = load_tempo_components(
        model_id=config.model.name,
        device=device_str,
        load_model_wrapper=True,
        load_token_generator=False,
        load_parallel_generator=False,
        debug_mode=config.debug.debug_mode,
        use_fast_tokenizer=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    model = components["model"]
    tokenizer = components["tokenizer"]
    model_wrapper = components["model_wrapper"]

    print(f"Model loaded in {time.time() - load_start:.2f}s")
    print()

    # Create experiment runner
    runner = ExperimentRunner(
        model=model_wrapper,
        tokenizer=tokenizer,
        device=device_str,
    )
    runner.debug_mode = config.debug.debug_mode

    # Convert to args dict
    args_dict = config.to_args_dict()

    # Run experiment
    print("Starting generation...")
    print()

    try:
        gen_start = time.time()
        results = runner.run_experiment(args_dict)
        gen_time = time.time() - gen_start

        print()
        print("="*60)
        print("Generation completed successfully!")
        print(f"Generation time: {gen_time:.2f}s")
        if config.output.json_file:
            print(f"Results saved to: {config.output.json_file}")
        print("="*60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
