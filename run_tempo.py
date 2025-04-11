#!/usr/bin/env python3
"""
TEMPO: Threshold-based Exploration with Multipath Parallel Output

This script runs text generation using the TEMPO parallel generation approach.
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.experiments import ExperimentRunner, ArgumentParser
from src.modeling.model_wrapper import TEMPOModelWrapper

def main():
    """Main entry point for the TEMPO generator."""
    # Parse command line arguments
    args = ArgumentParser.parse_args()
    
    # Set random seeds for reproducibility
    random_seed = args.pop("seed")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Determine device and precision
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if device != "cpu" else torch.float32
    print(f"Using device: {device} with {dtype}")
    
    # Load model and tokenizer with optimized settings
    model_name = "deepcogito/cogito-v1-preview-qwen-14B"
    print(f"Loading model: {model_name}")
    
    # Load tokenizer only once with caching
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # First load the config to modify it
    config = AutoConfig.from_pretrained(model_name)
    
    # Disable sliding window attention for Qwen models to fix compatibility issues
    if hasattr(config, "sliding_window") and config.sliding_window is not None:
        print(f"Disabling sliding window attention (was set to {config.sliding_window})")
        config.sliding_window = None
    
    # Load model with optimized settings and modified config
    print("Loading model with optimized settings...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Use eager attention for better compatibility
    )
    
    # Optimize model for inference
    if hasattr(model, "eval"):
        model.eval()
    
    # Wrap the model to capture intermediate values
    wrapped_model = TEMPOModelWrapper(model)
    print("Model wrapped with TEMPO wrapper for intermediate value extraction")
    
    # Create experiment runner with wrapped model
    runner = ExperimentRunner(wrapped_model, tokenizer, device)
    
    # Run experiment
    prompt = args.get('prompt', '')
    print(f"Running experiment with threshold: {args.get('threshold', 0.1)}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    # Run with performance monitoring
    with torch.inference_mode():
        results = runner.run_experiment(args)
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main() 