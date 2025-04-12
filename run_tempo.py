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
import time
from tqdm import tqdm
import os

# Add imports for profiling
import cProfile
import pstats
from pstats import SortKey
import src.generation.token_generator
import src.generation.token_selector
import src.pruning.pruner
import src.generation.attention_manager
import src.generation.text_formatter

# Dictionary to store timing information when profiling
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

# Track performance by sequence length
sequence_length_stats = {}

class PerformanceTracker:
    """
    Context manager to track performance of code blocks.
    """
    def __init__(self, name, detailed=False, seq_len=None):
        self.name = name
        self.detailed = detailed
        self.start_time = 0
        self.seq_len = seq_len
        
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
            
            # Track by sequence length if provided
            if self.seq_len is not None:
                if self.seq_len not in sequence_length_stats:
                    sequence_length_stats[self.seq_len] = {
                        "token_generation": [],
                        "token_selection": [],
                        "pruning": [],
                        "attention_update": []
                    }
                sequence_length_stats[self.seq_len][self.name].append(elapsed)
                
                # Debug print for sequence length tracking (only for token generation)
                if self.name == "token_generation":
                    len_list = len(sequence_length_stats[self.seq_len][self.name])
                    current_time = elapsed * 1000  # Convert to ms
                    print(f"Seq len: {self.seq_len:2d}, Token #{len_list:2d}, Time: {current_time:.1f}ms")
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
    def timed_get_next_token_logits(self, input_ids, attention_mask, past_key_values=None, custom_attention_mask=None):
        # Get current sequence length from input_ids and past_key_values
        seq_len = None
        
        # First, try to get sequence length directly from input_ids
        if hasattr(input_ids, 'shape') and len(input_ids.shape) > 1:
            if past_key_values is None:
                # For the first token, use full input_ids length
                seq_len = input_ids.shape[1]
            else:
                # For subsequent tokens, need to add kv_cache length + 1 (current token)
                # Find past sequence length from KV cache
                past_seq_len = 0
                if isinstance(past_key_values, list) and len(past_key_values) > 0:
                    for layer in past_key_values:
                        if isinstance(layer, tuple) and len(layer) >= 2:
                            if hasattr(layer[0], 'size') and layer[0].dim() >= 3:
                                past_seq_len = layer[0].size(2)
                                break
                
                # Total sequence length is past length + current token
                seq_len = past_seq_len + 1
        
        # Log the sequence length for debugging
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"Processing sequence length: {seq_len}")
        
        with PerformanceTracker("token_generation", detailed=False, seq_len=seq_len):
            return orig_get_next_token_logits(self, input_ids, attention_mask, past_key_values, custom_attention_mask)
            
    def timed_select_tokens(self, *args, **kwargs):
        # Get current sequence length
        seq_len = None
        if len(args) > 1 and hasattr(args[1], 'shape') and len(args[1].shape) > 1:
            seq_len = args[1].shape[1]  # Second arg is usually logits with shape [batch, seq_len, vocab]
        
        with PerformanceTracker("token_selection", seq_len=seq_len):
            return orig_select_tokens(self, *args, **kwargs)
            
    def timed_prune_tokens(self, *args, **kwargs):
        # Get current sequence length
        seq_len = None
        if len(args) > 0 and isinstance(args[0], list) and len(args[0]) > 0:
            # First arg is token_set_list, each with an 'input_ids' field
            if hasattr(args[0][0], 'input_ids') and hasattr(args[0][0].input_ids, 'shape'):
                seq_len = args[0][0].input_ids.shape[1]
        
        with PerformanceTracker("pruning", seq_len=seq_len):
            return orig_prune_tokens(self, *args, **kwargs)
            
    def timed_update_input(self, *args, **kwargs):
        # Get current sequence length
        seq_len = None
        if len(args) > 0 and isinstance(args[0], list) and len(args[0]) > 0:
            # First arg is token_set_list, each with an 'input_ids' field
            if hasattr(args[0][0], 'input_ids') and hasattr(args[0][0].input_ids, 'shape'):
                seq_len = args[0][0].input_ids.shape[1]
        
        with PerformanceTracker("attention_update", seq_len=seq_len):
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

def print_sequence_length_analysis():
    """Print a detailed analysis of how performance scales with sequence length."""
    if not sequence_length_stats:
        print("\nNo sequence length data available.")
        return
        
    print("\n" + "="*50)
    print("SEQUENCE LENGTH PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Sort sequence lengths
    seq_lengths = sorted(sequence_length_stats.keys())
    
    print("\nToken Generation Time by Sequence Length:")
    print(f"{'Seq Length':<10} {'Avg Time (ms)':<15} {'Ratio to Initial':<20} {'Visual'}")
    print("-" * 70)
    
    # Get the initial sequence length for reference
    initial_seq_len = seq_lengths[0]
    initial_gen_time = sum(sequence_length_stats[initial_seq_len]["token_generation"]) / len(sequence_length_stats[initial_seq_len]["token_generation"]) * 1000
    
    # Find the maximum time for scaling the visualization
    max_time_ms = 0
    for seq_len in seq_lengths:
        if "token_generation" in sequence_length_stats[seq_len] and sequence_length_stats[seq_len]["token_generation"]:
            avg_time = sum(sequence_length_stats[seq_len]["token_generation"]) / len(sequence_length_stats[seq_len]["token_generation"]) * 1000
            if avg_time > max_time_ms:
                max_time_ms = avg_time
    
    # Scale factor for ASCII visualization (max 40 characters)
    scale_factor = 40.0 / max_time_ms if max_time_ms > 0 else 1.0
    
    for seq_len in seq_lengths:
        if "token_generation" in sequence_length_stats[seq_len] and sequence_length_stats[seq_len]["token_generation"]:
            avg_time = sum(sequence_length_stats[seq_len]["token_generation"]) / len(sequence_length_stats[seq_len]["token_generation"]) * 1000  # ms
            ratio = avg_time / initial_gen_time if initial_gen_time > 0 else 0
            
            # Create visual bar
            bar_length = int(avg_time * scale_factor)
            visual_bar = "▇" * bar_length
            
            print(f"{seq_len:<10} {avg_time:<15.2f} {ratio:<20.2f} {visual_bar}")
    
    # Analyze scaling behavior
    if len(seq_lengths) > 2:
        # Check if scaling appears to be quadratic (O(n²)) or linear (O(n))
        first_len = seq_lengths[0]
        last_len = seq_lengths[-1]
        first_time = sum(sequence_length_stats[first_len]["token_generation"]) / len(sequence_length_stats[first_len]["token_generation"])
        last_time = sum(sequence_length_stats[last_len]["token_generation"]) / len(sequence_length_stats[last_len]["token_generation"])
        
        # Calculate observed ratio
        observed_ratio = last_time / first_time if first_time > 0 else 0
        
        # Calculate theoretical ratios
        linear_ratio = last_len / first_len if first_len > 0 else 0
        quadratic_ratio = (last_len / first_len) ** 2 if first_len > 0 else 0
        
        print("\nScaling Analysis:")
        print(f"Sequence length increased by factor: {last_len / first_len:.2f}x")
        print(f"Time increased by factor: {observed_ratio:.2f}x")
        print(f"Expected for linear scaling (O(n)): {linear_ratio:.2f}x")
        print(f"Expected for quadratic scaling (O(n²)): {quadratic_ratio:.2f}x")
        
        # Determine which scaling is closer
        linear_diff = abs(observed_ratio - linear_ratio)
        quadratic_diff = abs(observed_ratio - quadratic_ratio)
        
        if linear_diff < quadratic_diff:
            print(f"Scaling appears closer to LINEAR (O(n)) with deviation of {linear_diff:.2f}")
        else:
            print(f"Scaling appears closer to QUADRATIC (O(n²)) with deviation of {quadratic_diff:.2f}")
            print("This suggests attention computation is the bottleneck, which is expected in Transformers.")
            print("\nPotential optimizations:")
            print("1. Use a model with sliding window attention for longer contexts")
            print("2. Enable CUDA Flash Attention if using NVIDIA GPUs")
            print("3. Consider using sequence pruning more aggressively as length increases")

    # Analyze pruning as well
    if any("pruning" in sequence_length_stats[seq_len] and sequence_length_stats[seq_len]["pruning"] for seq_len in seq_lengths):
        print("\nPruning Time by Sequence Length:")
        print(f"{'Seq Length':<10} {'Avg Time (ms)':<15} {'Calls':<10}")
        print("-" * 35)
        
        for seq_len in seq_lengths:
            if "pruning" in sequence_length_stats[seq_len] and sequence_length_stats[seq_len]["pruning"]:
                avg_time = sum(sequence_length_stats[seq_len]["pruning"]) / len(sequence_length_stats[seq_len]["pruning"]) * 1000  # ms
                calls = len(sequence_length_stats[seq_len]["pruning"])
                print(f"{seq_len:<10} {avg_time:<15.2f} {calls:<10}")
    
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
    """Main entry point for the TEMPO generator."""
    # Parse command line arguments
    args = ArgumentParser.parse_args()
    
    # Get profiling flags
    enable_profiling = args.pop("profile", False)
    use_cprofile = args.pop("use_cprofile", False)
    profile_output = args.pop("profile_output", "tempo_profile.prof")
    
    # Setup profiling if enabled
    profiler = None
    if enable_profiling:
        print("Profiling enabled - performance details will be reported at the end")
        # Apply timing instrumentation
        patch_classes()
        
        # Setup cProfile if requested
        if use_cprofile:
            profiler = cProfile.Profile()
            profiler.enable()
    
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
    
    # Track model loading time if profiling
    model_load_tracker = PerformanceTracker("model_loading", detailed=enable_profiling) if enable_profiling else None
    if model_load_tracker:
        model_load_tracker.__enter__()
    
    # Load model and tokenizer with optimized settings
    model_name = "deepcogito/cogito-v1-preview-qwen-14B"
    print(f"Loading model: {model_name}")
    
    # Show progress while loading model components
    print("Loading tokenizer...", end="", flush=True)
    start_time = time.time()
    
    # Track tokenization time if profiling
    tokenize_tracker = PerformanceTracker("tokenization") if enable_profiling else None
    if tokenize_tracker:
        tokenize_tracker.__enter__()
    
    # Load tokenizer only once with caching
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenize_tracker:
        tokenize_tracker.__exit__(None, None, None)
    
    print(f" Done! ({time.time() - start_time:.2f}s)")
    
    # First load the config to modify it
    print("Loading model configuration...", end="", flush=True)
    start_time = time.time()
    config = AutoConfig.from_pretrained(model_name)
    
    # Disable sliding window attention for Qwen models to fix compatibility issues
    if hasattr(config, "sliding_window") and config.sliding_window is not None:
        print(f"\nDisabling sliding window attention (was set to {config.sliding_window})")
        config.sliding_window = None
    else:
        print(f" Done! ({time.time() - start_time:.2f}s)")
    
    # Load model with optimized settings and modified config
    print("Loading model with optimized settings...")
    start_time = time.time()
    with tqdm(total=100, desc="Loading model", unit="%") as pbar:
        # Update progress at key loading stages
        pbar.update(10)  # Starting model load
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else device,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Use eager attention for better compatibility
        )
        
        pbar.update(80)  # Major loading complete
        
        # Optimize model for inference
        if hasattr(model, "eval"):
            model.eval()
            
        pbar.update(10)  # Finalization complete
    
    print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    
    # Close model loading tracker if profiling
    if model_load_tracker:
        model_load_tracker.__exit__(None, None, None)
    
    # Wrap the model to capture intermediate values
    print("Wrapping model with TEMPO wrapper...", end="", flush=True)
    start_time = time.time()
    wrapped_model = TEMPOModelWrapper(model)
    print(f" Done! ({time.time() - start_time:.2f}s)")
    print("Model wrapped with TEMPO wrapper for intermediate value extraction")
    
    # Create experiment runner with wrapped model
    runner = ExperimentRunner(wrapped_model, tokenizer, device)
    
    # Run experiment
    prompt = args.get('prompt', '')
    threshold = args.get('threshold', 0.1)
    max_tokens = args.get('max_tokens', 100)
    print(f"Running experiment with threshold: {threshold}, max tokens: {max_tokens}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    # Run with performance monitoring
    generation_tracker = PerformanceTracker("full_generation", detailed=enable_profiling) if enable_profiling else None
    if generation_tracker:
        generation_tracker.__enter__()
    
    start_time = time.time()
    with torch.inference_mode():
        results = runner.run_experiment(args)
    generation_time = time.time() - start_time
    
    if generation_tracker:
        generation_tracker.__exit__(None, None, None)
    
    # Print generation statistics
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    print(f"Average tokens/second: {max_tokens/generation_time:.2f}")
    print("\nExperiment completed successfully!")
    
    # Print profiling reports if enabled
    if enable_profiling:
        # Stop and save cProfile results if used
        if profiler:
            profiler.disable()
            # Sort by cumulative time
            profiler.dump_stats(profile_output)
            
            # Print top functions by time
            print("\nTop functions by cumulative time:")
            stats = pstats.Stats(profile_output).sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(20)  # Print top 20 functions
        
        # Print performance report
        print_performance_report()
        
        # Print sequence length analysis
        print_sequence_length_analysis()
        
        # Print memory usage
        profile_memory_usage()

if __name__ == "__main__":
    main() 