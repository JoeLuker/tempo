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
from src.modeling.model_wrapper import TEMPOModelWrapper # Ensure TEMPOModelWrapper is imported

# Add imports for cProfile (as the chosen profiling method)
import cProfile
import pstats
from pstats import SortKey

# --- Helper Functions ---

def get_device():
    """
    Determine the best available device for model execution.
    Returns 'cuda' if an NVIDIA GPU is available, 'mps' for Apple Silicon,
    or 'cpu' as a fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_dtype(device_str=None):
    """
    Determine the appropriate dtype based on the device.

    Args:
        device_str: Device string ('cuda', 'mps', 'cpu')

    Returns:
        torch.dtype: The appropriate dtype for the device
    """
    if device_str is None:
        device_str = get_device()

    # Use bfloat16 for CUDA Ampere+, float16 for older CUDA/MPS, float32 for CPU
    if device_str == "cuda":
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
             return torch.bfloat16 # Ampere and later GPUs
        else:
             return torch.float16 # Older GPUs
    elif device_str == "mps":
         # MPS performance with float16 can be inconsistent, float32 is safer
         return torch.float32
    else:  # "cpu"
        return torch.float32

# --- Main Execution ---

def main():
    """Main entry point for the TEMPO generator."""
    try:
        # 1. Parse Arguments using the consolidated parser
        args_dict = ArgumentParser.parse_args()

        # Extract profiling flags early
        enable_profiling = args_dict.pop("profile", False)
        use_cprofile = args_dict.pop("use_cprofile", False) # Keep this flag
        profile_output = args_dict.pop("profile_output", "tempo_profile.prof")
        debug_mode = args_dict.get("debug_mode", False) # Get debug mode

        # 2. Set Random Seeds
        seed = args_dict.get("seed", 42) # Use .get with default
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Using random seed: {seed}")

        # 3. Determine Device and DType
        device_str = get_device()
        device = torch.device(device_str) # Use torch.device object
        dtype = get_device_dtype(device_str)
        print(f"Using device: {device} with dtype: {dtype}")

        # 4. Load Model and Tokenizer
        # User wants this hardcoded for now, but ideally use args_dict.get("model", DEFAULT_MODEL)
        model_name = "deepcogito/cogito-v1-preview-llama-3B"
        print(f"Loading model: {model_name}")

        load_start_time = time.time()

        print("Loading tokenizer...", end="", flush=True)
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f" Done! ({time.time() - tokenizer_start:.2f}s)")

        print("Loading model configuration...", end="", flush=True)
        config_start = time.time()
        config = AutoConfig.from_pretrained(model_name)
        # Explicitly check for Qwen model type for specific logic
        if 'qwen' in config.model_type.lower():
             if hasattr(config, "sliding_window") and config.sliding_window is not None:
                  print(f"\nDisabling Qwen sliding window (was {config.sliding_window})", end="")
                  config.sliding_window = None
        # Optionally enable attention output if needed by pruning strategies
        # config.output_attentions = True # Enable if retroactive pruning needs it
        print(f" Done! ({time.time() - config_start:.2f}s)")


        print("Loading model weights...", end="", flush=True)
        model_load_start = time.time()
        # Recommended loading approach
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            # Use device_map for multi-GPU or very large models on CUDA
            device_map="auto" if device_str == "cuda" else None,
            # Use 'auto' or specific implementation like 'flash_attention_2' if available
            attn_implementation="eager", # Stick to eager for now for stability
            low_cpu_mem_usage=True,
        )
        # Manually move to device if not using device_map
        if device_str != "cuda":
             model = model.to(device)

        print(f" Done! ({time.time() - model_load_start:.2f}s)")
        print(f"Total model loading time: {time.time() - load_start_time:.2f}s")

        # Ensure model is in evaluation mode
        model.eval()

        # 5. Create Model Wrapper (passing debug_mode)
        # TEMPOModelWrapper only needs the raw model now
        model_wrapper = TEMPOModelWrapper(model)
        model_wrapper.set_debug_mode(debug_mode) # Pass debug mode

        # 6. Create Experiment Runner (passing components and debug_mode)
        runner = ExperimentRunner(
            model=model_wrapper, # Pass the wrapped model
            tokenizer=tokenizer,
            device=device_str, # Pass device string
            # Note: ExperimentRunner does NOT wrap again if skip_wrapping=True
            # We are passing the already wrapped model, so no skip_wrapping needed
        )
        # Set debug mode on the runner itself, which should propagate it
        runner.debug_mode = debug_mode # Pass debug mode

        # 7. Run Experiment (with cProfile if requested)
        profiler = None
        if enable_profiling and use_cprofile:
            print("cProfile profiling enabled.")
            profiler = cProfile.Profile()
            profiler.enable()

        generation_start_time = time.time()
        print(f"\nStarting generation for prompt: '{args_dict.get('prompt','')[:50]}...'")

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
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    # Ensure we're in the project root directory (useful if script is called from elsewhere)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check if current dir is different from script dir before changing
    if os.getcwd() != script_dir:
         print(f"Changing working directory to: {script_dir}")
         os.chdir(script_dir)

    main()