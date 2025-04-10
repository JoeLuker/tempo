#!/usr/bin/env python3
"""
Profiling script for TEMPO to identify performance bottlenecks.
"""

import time
import cProfile
import pstats
from pstats import SortKey
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.experiments import ExperimentRunner, ArgumentParser
import os
import argparse
import src.generation.token_generator
import src.generation.token_selector
import src.pruning.pruner
import src.generation.attention_manager
import src.generation.text_formatter
import src

# Dictionary to store timing information
timings = {
    "model_loading": 0,
    "tokenization": 0,
    "token_generation": [],
    "token_selection": [],
    "pruning": [],
    "attention_update": [],
    "text_formatting": 0,
    "full_generation": 0
}

class PerformanceTracker:
    """
    Context manager to track performance of code blocks.
    """
    def __init__(self, name, detailed=False):
        self.name = name
        self.detailed = detailed
        self.start_time = 0
        
    def __enter__(self):
        # Record current CUDA events if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure cuda operations are complete before recording end time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        
        # Store timing info
        if self.name in ["token_generation", "token_selection", "pruning", "attention_update"]:
            timings[self.name].append(elapsed)
        else:
            timings[self.name] = elapsed
        
        if self.detailed:
            print(f"{self.name}: {elapsed:.4f} seconds")

def patch_classes():
    """
    Monkey-patch key methods to add timing instrumentation.
    Only patch high-level TEMPO methods, not model internals.
    """
    # Store original methods
    orig_get_next_token_logits = src.generation.token_generator.TokenGenerator.get_next_token_logits_cached
    orig_select_tokens = src.generation.token_selector.TokenSelector.select_tokens_above_threshold
    orig_prune_tokens = src.pruning.pruner.Pruner.prune_parallel_tokens
    orig_update_input = src.generation.attention_manager.AttentionManager.update_input_efficiently
    orig_format_text = src.generation.text_formatter.TextFormatter.format_generated_text
    
    # Patch methods with timing wrappers
    def timed_get_next_token_logits(self, *args, **kwargs):
        with PerformanceTracker("token_generation", detailed=False):
            return orig_get_next_token_logits(self, *args, **kwargs)
            
    def timed_select_tokens(self, *args, **kwargs):
        with PerformanceTracker("token_selection"):
            return orig_select_tokens(self, *args, **kwargs)
            
    def timed_prune_tokens(self, *args, **kwargs):
        with PerformanceTracker("pruning"):
            return orig_prune_tokens(self, *args, **kwargs)
            
    def timed_update_input(self, *args, **kwargs):
        with PerformanceTracker("attention_update"):
            return orig_update_input(self, *args, **kwargs)
            
    def timed_format_text(self, *args, **kwargs):
        with PerformanceTracker("text_formatting"):
            return orig_format_text(self, *args, **kwargs)
    
    # Apply patches ONLY to TEMPO-specific methods, not model internals
    src.generation.token_generator.TokenGenerator.get_next_token_logits_cached = timed_get_next_token_logits
    src.generation.token_selector.TokenSelector.select_tokens_above_threshold = timed_select_tokens  
    src.pruning.pruner.Pruner.prune_parallel_tokens = timed_prune_tokens
    src.generation.attention_manager.AttentionManager.update_input_efficiently = timed_update_input
    src.generation.text_formatter.TextFormatter.format_generated_text = timed_format_text
    
    # DO NOT patch torch.nn.Module.__call__ at all - this is what causes the issues

def print_performance_report():
    """Print a detailed performance report."""
    print("\n" + "="*50)
    print("TEMPO PERFORMANCE REPORT")
    print("="*50)
    
    print(f"\nModel Loading: {timings['model_loading']:.2f} seconds")
    print(f"Tokenization: {timings['tokenization']:.4f} seconds")
    
    # Calculate statistics for iterative operations
    for key in ["token_generation", "token_selection", "pruning", "attention_update"]:
        if timings[key]:
            avg = sum(timings[key]) / len(timings[key])
            total = sum(timings[key])
            maximum = max(timings[key])
            print(f"\n{key.replace('_', ' ').title()}:")
            print(f"  Total: {total:.4f} seconds")
            print(f"  Avg per step: {avg:.4f} seconds")
            print(f"  Max: {maximum:.4f} seconds")
            print(f"  Steps: {len(timings[key])}")
    
    print(f"\nText Formatting: {timings['text_formatting']:.4f} seconds")
    print(f"\nTotal Generation Time: {timings['full_generation']:.2f} seconds")
    
    # Calculate percentage breakdown
    print("\nPercentage Breakdown:")
    total_time = timings['full_generation']
    if total_time > 0:
        for key in ["token_generation", "token_selection", "pruning", "attention_update"]:
            if timings[key]:
                pct = sum(timings[key]) / total_time * 100
                print(f"  {key.replace('_', ' ').title()}: {pct:.1f}%")
        
        text_pct = timings['text_formatting'] / total_time * 100
        print(f"  Text Formatting: {text_pct:.1f}%")
    
    print("="*50)

def profile_memory_usage():
    """Profile memory usage."""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    import psutil
    process = psutil.Process(os.getpid())
    print(f"\nCPU Memory Usage: {process.memory_info().rss / 1024**2:.1f} MB")

def main():
    """Main profiling function."""
    # Parse profiling arguments
    parser = argparse.ArgumentParser(description="Profile TEMPO performance")
    parser.add_argument("--prompt", type=str, default="Write a story about a robot who discovers emotions:", 
                       help="Prompt to use for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--threshold", type=float, default=0.1, help="Token selection threshold")
    parser.add_argument("--use-profiler", action="store_true", help="Use cProfile for detailed profiling")
    parser.add_argument("--output-file", type=str, default="tempo_profile.prof", help="Output file for cProfile results")
    
    profile_args = parser.parse_args()
    
    # Apply timing instrumentation
    patch_classes()
    
    # Begin profiling if requested
    profiler = None
    if profile_args.use_profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    
    # Start TEMPO with timing
    with PerformanceTracker("full_generation", detailed=True):
        # Parse command line arguments from ArgumentParser
        args_list = [
            "--prompt", profile_args.prompt,
            "--max-tokens", str(profile_args.max_tokens),
            "--threshold", str(profile_args.threshold),
            "--use-pruning"  # Add pruning flag for a more realistic test
        ]
        # Use the correct way to parse arguments for this specific ArgumentParser implementation
        import sys
        old_argv = sys.argv
        sys.argv = ["run_tempo.py"] + args_list
        args = ArgumentParser.parse_args()
        sys.argv = old_argv
        
        # Add device info to args manually since it's handled in run_tempo.py differently
        args["device"] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Set random seeds for reproducibility
        random_seed = args.pop("seed", 42)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        # Determine device and precision
        device = args.get("device", "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if device != "cpu" else torch.float32
        print(f"Using device: {device} with {dtype}")
        
        # Load model and tokenizer
        with PerformanceTracker("model_loading", detailed=True):
            # Load model and tokenizer with optimized settings
            model_name = "mistralai/Mistral-7B-v0.3"
            print(f"Loading model: {model_name}")
            
            # Load tokenizer only once with caching
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with optimized settings
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
    
    # Stop and save cProfile results if used
    if profiler:
        profiler.disable()
        # Sort by cumulative time
        profiler.dump_stats(profile_args.output_file)
        
        # Print top functions by time
        print("\nTop functions by cumulative time:")
        stats = pstats.Stats(profile_args.output_file).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(20)  # Print top 20 functions
    
    # Print performance report
    print_performance_report()
    
    # Print memory usage
    profile_memory_usage()

if __name__ == "__main__":
    main() 