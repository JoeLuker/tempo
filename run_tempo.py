#!/usr/bin/env python3
"""
TEMPO: Threshold-based Exploration with Multipath Parallel Output

This script runs text generation using the TEMPO parallel generation approach.
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import os
import sys
import traceback

# Use the standardized ArgumentParser from src
from src.experiments import ArgumentParser, ExperimentRunner
from src.modeling.model_wrapper import (
    TEMPOModelWrapper,
)  # Ensure TEMPOModelWrapper is imported

# Add imports for cProfile (as the chosen profiling method)
import cProfile
import pstats
from pstats import SortKey

# --- Helper Functions ---

# Import centralized utilities
from src.utils.model_utils import (
    get_best_device,
    get_device_dtype,
    load_model,
    load_tempo_components,
)

# --- Main Execution ---


def main():
    """Main entry point for the TEMPO generator."""
    try:
        # 1. Parse Arguments using the consolidated parser
        args_dict = ArgumentParser.parse_args()

        # Extract profiling flags early
        enable_profiling = args_dict.pop("profile", False)
        use_cprofile = args_dict.pop("use_cprofile", False)  # Keep this flag
        profile_output = args_dict.pop("profile_output", "tempo_profile.prof")
        debug_mode = args_dict.get("debug_mode", False)  # Get debug mode

        # 2. Set Random Seeds
        seed = args_dict.get("seed", 42)  # Use .get with default
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Using random seed: {seed}")

        # 3. Determine Device and DType using centralized utilities
        device_str = get_best_device()
        device = torch.device(device_str)  # Use torch.device object
        dtype = get_device_dtype(device_str)
        print(f"Using device: {device} with dtype: {dtype}")

        # 4. Load Model, Tokenizer and TEMPO components
        # User wants this hardcoded for now, but ideally use args_dict.get("model", DEFAULT_MODEL)
        model_name = "deepcogito/cogito-v1-preview-llama-3B"
        print(f"Loading model: {model_name}")

        load_start_time = time.time()

        print("Loading TEMPO components...")
        # Load model and TEMPO components using the centralized function
        components = load_tempo_components(
            model_id=model_name,
            device=device_str,
            load_model_wrapper=True,
            load_token_generator=False,  # We'll create this later with the ExperimentRunner
            load_parallel_generator=False,  # We'll create this later with the ExperimentRunner
            debug_mode=debug_mode,
            use_fast_tokenizer=True,
            attn_implementation="eager",  # Stick to eager for now for stability
            low_cpu_mem_usage=True,
        )

        # Extract components
        model = components["model"]
        tokenizer = components["tokenizer"]
        model_wrapper = components["model_wrapper"]

        print(f"Model and components loaded in {time.time() - load_start_time:.2f}s")
        print(f"Model loaded on device: {model_wrapper.device}")

        # Ensure debug mode is set
        model_wrapper.set_debug_mode(debug_mode)

        # 6. Create Experiment Runner (passing components and debug_mode)
        runner = ExperimentRunner(
            model=model_wrapper,  # Pass the wrapped model
            tokenizer=tokenizer,
            device=device_str,  # Pass device string
            # Note: ExperimentRunner does NOT wrap again if skip_wrapping=True
            # We are passing the already wrapped model, so no skip_wrapping needed
        )
        # Set debug mode on the runner itself, which should propagate it
        runner.debug_mode = debug_mode  # Pass debug mode

        # 7. Run Experiment (with cProfile if requested)
        profiler = None
        if enable_profiling and use_cprofile:
            print("cProfile profiling enabled.")
            profiler = cProfile.Profile()
            profiler.enable()

        generation_start_time = time.time()
        print(
            f"\nStarting generation for prompt: '{args_dict.get('prompt','')[:50]}...'"
        )

        # Call run_experiment with the full dictionary of arguments
        results = runner.run_experiment(args_dict)

        generation_time = time.time() - generation_start_time
        print(f"\nGeneration completed in {generation_time:.2f} seconds")

        # 8. Handle Profiling Results
        if profiler:
            profiler.disable()
            print(f"Saving cProfile stats to {profile_output}")
            # Sort by cumulative time and save/print
            stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
            stats.dump_stats(profile_output)
            print("\nTop 20 functions by cumulative time:")
            stats.print_stats(20)

        # Optionally print basic timing from results if available
        gen_time = results.get("generation_time", generation_time)
        prune_time = results.get("pruning_time", 0)
        print(f"Reported Generation Time: {gen_time:.4f}s")
        print(f"Reported Pruning Time: {prune_time:.4f}s")

        print("\nExperiment finished successfully.")

    except Exception as e:
        print(f"\n--- An Error Occurred ---", file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Details: {e}", file=sys.stderr)
        print("\n--- Traceback ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("-------------------", file=sys.stderr)
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    # Ensure we're in the project root directory (useful if script is called from elsewhere)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check if current dir is different from script dir before changing
    if os.getcwd() != script_dir:
        print(f"Changing working directory to: {script_dir}")
        os.chdir(script_dir)

    main()
