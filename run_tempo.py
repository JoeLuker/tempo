#!/usr/bin/env python3
"""
TEMPO: Threshold-based Exploration with Multipath Parallel Output

This script runs text generation using the TEMPO parallel generation approach.
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.experiments import ExperimentRunner, ArgumentParser

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
    model_name = "mistralai/Mistral-7B-v0.3"
    print(f"Loading model: {model_name}")
    
    # Load tokenizer only once with caching
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimized settings
    print("Loading model with optimized settings...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "eager",
        use_cache=True,
        low_cpu_mem_usage=True
    )
    
    # Optimize model for inference
    if hasattr(model, "eval"):
        model.eval()
    
    # Enable memory efficient attention if available
    if hasattr(model.config, "use_memory_efficient_attention") and device == "cuda":
        model.config.use_memory_efficient_attention = True
    
    # Create experiment runner
    runner = ExperimentRunner(model, tokenizer, device)
    
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