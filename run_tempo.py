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
    "tokenization": 0,
    "model_loading": 0,
    "token_generation": [],
    "token_selection": [],
    "pruning": [],
    "retroactive_pruning": [],
    "attention_update": [],
    "full_generation": 0,
    "text_formatting": 0.0,
}

# Track performance by sequence length
sequence_length_stats = {}


# Create a dedicated class for sequence length tracking
class SequenceLengthTracker:
    """
    Class for tracking sequence length and step count across generation.
    Provides a clean interface for performance monitoring.
    """

    def __init__(self, debug=False):
        """Initialize the sequence length tracker."""
        self.current_length = 0
        self.max_length = 0
        self.initial_length = 0
        self.debug = debug
        self.step_count = 0
        self.length_history = []
        # Initialize performance stats dictionary indexed by step
        self.step_stats = {}

    def update_length(self, new_length):
        """Update sequence length if the new length is larger than current."""
        if new_length > self.current_length:
            prev_length = self.current_length
            self.current_length = new_length

            # Record in history
            self.length_history.append(new_length)

            # Update max length
            if new_length > self.max_length:
                self.max_length = new_length

            # Set initial length if this is the first update
            if self.initial_length == 0:
                self.initial_length = new_length

            # Debug output
            if self.debug:
                print(f"New sequence length encountered: {self.current_length}")
                print(
                    f"🔄 Global sequence length updated: {prev_length} → {self.current_length}"
                )
                print(
                    f"   Step count: {self.step_count}, Growth rate: +{self.current_length - prev_length} tokens"
                )

                # Every 10 updates, print a summary of sequence length progression
                if self.step_count % 10 == 0 and self.step_count > 0:
                    self.print_progress_summary()

            return True
        return False

    def increment_step(self, sequence_length=None):
        """Increment the step count and update sequence length if provided."""
        self.step_count += 1

        # Initialize stats for this step
        self.step_stats[self.step_count] = {
            "token_generation": [],
            "token_selection": [],
            "pruning": [],
            "retroactive_pruning": [],
            "attention_update": [],
            "sequence_length": sequence_length or self.current_length,
        }

        # Update sequence length if provided
        if sequence_length is not None:
            self.update_length(sequence_length)

        if self.debug:
            print(f"Step {self.step_count}: Sequence length = {self.current_length}")

        return self.step_count

    def record_timing(self, operation, elapsed_time, step=None):
        """Record timing information for an operation at a specific step."""
        # If step not provided, use current step
        step = step or self.step_count

        if operation in [
            "token_generation",
            "token_selection",
            "pruning",
            "retroactive_pruning",
            "attention_update",
        ]:
            # Add to global timings
            timings[operation].append(elapsed_time)

            # Add to step-specific timings
            if step in self.step_stats:
                if operation in self.step_stats[step]:
                    self.step_stats[step][operation].append(elapsed_time)

                    # Debug print for timing
                    if self.debug and operation == "token_generation":
                        current_time = elapsed_time * 1000  # Convert to ms
                        print(f"🔍 Step: {step:2d}, Time: {current_time:.1f}ms")

    def get_length(self):
        """Get the current sequence length."""
        return self.current_length

    def get_max_length(self):
        """Get the maximum sequence length observed."""
        return self.max_length

    def get_initial_length(self):
        """Get the initial sequence length."""
        return self.initial_length

    def get_growth_rate(self):
        """Calculate the average growth rate of sequence length."""
        if self.step_count > 1:
            return (self.current_length - self.initial_length) / (self.step_count - 1)
        return 0

    def print_progress_summary(self):
        """Print a summary of sequence length progression."""
        if len(self.length_history) > 1:
            print("\n--- Sequence Length Progression Summary ---")
            print(f"Initial length: {self.initial_length}")
            print(f"Current length: {self.current_length}")
            print(f"Maximum length: {self.max_length}")
            print(f"Total steps: {self.step_count}")
            print(f"Average growth rate: {self.get_growth_rate():.2f} tokens/step")
            print("----------------------------------------\n")

    def get_step_stats(self):
        """Get the step-based statistics for analysis."""
        return self.step_stats


class PerformanceTracker:
    """
    Context manager to track performance of code blocks.
    """

    def __init__(self, name, detailed=False, seq_tracker=None, step=None):
        self.name = name
        self.detailed = detailed
        self.start_time = 0
        self.seq_tracker = seq_tracker
        self.step = (
            step if step is not None else (seq_tracker.step_count if seq_tracker else 0)
        )

        # Debug output
        if (
            seq_tracker
            and seq_tracker.debug
            and self.name
            in [
                "token_generation",
                "token_selection",
                "pruning",
                "retroactive_pruning",
                "attention_update",
            ]
        ):
            print(
                f"Performance tracker for {self.name} using step: {self.step}, seq len: {seq_tracker.get_length()}"
            )

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

        # Store timing info by operation type
        if self.name in [
            "token_generation",
            "token_selection",
            "pruning",
            "retroactive_pruning",
            "attention_update",
        ]:
            # Store in global timings
            timings[self.name].append(elapsed)

            # Store in sequence tracker if available
            if self.seq_tracker:
                self.seq_tracker.record_timing(self.name, elapsed, self.step)

        elif self.name == "text_formatting":
            # Text formatting is only called once
            timings[self.name] = elapsed


def patch_classes(seq_tracker=None):
    """
    Patch performance-sensitive methods with timing wrappers.
    This allows us to profile the performance of each component.

    Args:
        seq_tracker: Optional SequenceLengthTracker instance for performance monitoring
    """
    # Store original methods
    orig_get_next_token_logits = (
        src.generation.token_generator.TokenGenerator.get_next_token_logits_cached
    )
    orig_select_tokens = (
        src.generation.token_selector.TokenSelector.select_tokens_above_threshold
    )
    orig_prune_tokens = src.pruning.pruner.Pruner.prune_parallel_tokens
    orig_update_input = (
        src.generation.attention_manager.AttentionManager.update_input_efficiently
    )
    orig_format_text = src.generation.text_formatter.TextFormatter.format_generated_text
    orig_format_text_with_pruning = (
        src.generation.text_formatter.TextFormatter.format_generated_text_with_pruning
    )
    orig_retroactive_prune = (
        src.pruning.retroactive_pruner.RetroactivePruner.retroactively_prune
    )

    # Patch methods with timing wrappers
    def timed_get_next_token_logits(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        custom_attention_mask=None,
    ):
        # For the first token (without past_key_values), use the input length as initial length
        if (
            past_key_values is None
            and seq_tracker
            and seq_tracker.get_initial_length() == 0
        ):
            if hasattr(input_ids, "shape") and len(input_ids.shape) > 1:
                initial_length = input_ids.shape[1]
                if seq_tracker.debug:
                    print(f"Initial sequence - using input length: {initial_length}")
                seq_tracker.initial_length = initial_length
                seq_tracker.update_length(initial_length)

        # Use the performance tracker with the current sequence length
        with PerformanceTracker("token_generation", seq_tracker=seq_tracker):
            return orig_get_next_token_logits(
                self, input_ids, attention_mask, past_key_values, custom_attention_mask
            )

    def timed_select_tokens(self, *args, **kwargs):
        # Use performance tracker
        with PerformanceTracker("token_selection", seq_tracker=seq_tracker):
            return orig_select_tokens(self, *args, **kwargs)

    def timed_prune_tokens(self, *args, **kwargs):
        # Use performance tracker
        with PerformanceTracker("pruning", seq_tracker=seq_tracker):
            return orig_prune_tokens(self, *args, **kwargs)

    def timed_update_input(self, *args, **kwargs):
        # Use performance tracker
        with PerformanceTracker("attention_update", seq_tracker=seq_tracker):
            return orig_update_input(self, *args, **kwargs)

    def timed_format_text(self, *args, **kwargs):
        with PerformanceTracker("text_formatting", seq_tracker=seq_tracker):
            return orig_format_text(self, *args, **kwargs)

    def timed_format_text_with_pruning(self, *args, **kwargs):
        with PerformanceTracker("text_formatting", seq_tracker=seq_tracker):
            return orig_format_text_with_pruning(self, *args, **kwargs)

    def timed_retroactive_prune(self, *args, **kwargs):
        # Use performance tracker
        with PerformanceTracker("retroactive_pruning", seq_tracker=seq_tracker):
            return orig_retroactive_prune(self, *args, **kwargs)

    # Apply patches ONLY to TEMPO-specific methods, not model internals
    src.generation.token_generator.TokenGenerator.get_next_token_logits_cached = (
        timed_get_next_token_logits
    )
    src.generation.token_selector.TokenSelector.select_tokens_above_threshold = (
        timed_select_tokens
    )
    src.pruning.pruner.Pruner.prune_parallel_tokens = timed_prune_tokens
    src.generation.attention_manager.AttentionManager.update_input_efficiently = (
        timed_update_input
    )
    src.generation.text_formatter.TextFormatter.format_generated_text = (
        timed_format_text
    )
    src.generation.text_formatter.TextFormatter.format_generated_text_with_pruning = (
        timed_format_text_with_pruning
    )
    src.pruning.retroactive_pruner.RetroactivePruner.retroactively_prune = (
        timed_retroactive_prune
    )


def print_performance_report(seq_tracker=None):
    """
    Print a detailed performance report.

    Args:
        seq_tracker: Optional SequenceLengthTracker with step-based timings
    """
    print("\n" + "=" * 50)
    print("TEMPO PERFORMANCE REPORT")
    print("=" * 50)

    print(f"\nModel Loading: {timings['model_loading']:.2f} seconds")
    print(f"Tokenization: {timings['tokenization']:.4f} seconds")

    # Calculate statistics for iterative operations
    for key in [
        "token_generation",
        "token_selection",
        "pruning",
        "retroactive_pruning",
        "attention_update",
    ]:
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

    # Add more detail if text formatting time is 0
    if timings["text_formatting"] <= 0.0001:
        print("  Warning: Text formatting time is unexpectedly low or zero.")
        print(
            "  This could indicate the text formatter is not being properly called or timed."
        )

    print(f"\nTotal Generation Time: {timings['full_generation']:.2f} seconds")

    # Calculate percentage breakdown
    total_time = timings["full_generation"]
    token_gen_time = (
        sum(timings["token_generation"]) if timings["token_generation"] else 0
    )
    token_select_time = (
        sum(timings["token_selection"]) if timings["token_selection"] else 0
    )
    pruning_time = sum(timings["pruning"]) if timings["pruning"] else 0
    retroactive_pruning_time = (
        sum(timings["retroactive_pruning"]) if timings["retroactive_pruning"] else 0
    )
    attn_update_time = (
        sum(timings["attention_update"]) if timings["attention_update"] else 0
    )
    text_format_time = timings["text_formatting"]

    token_pct = token_gen_time / total_time * 100
    select_pct = token_select_time / total_time * 100
    prune_pct = pruning_time / total_time * 100
    retroactive_prune_pct = retroactive_pruning_time / total_time * 100
    attn_pct = attn_update_time / total_time * 100
    text_pct = text_format_time / total_time * 100

    # Calculate other time not accounted for
    tracked_time = (
        token_gen_time
        + token_select_time
        + pruning_time
        + retroactive_pruning_time
        + attn_update_time
        + text_format_time
    )
    other_time = total_time - tracked_time
    other_pct = other_time / total_time * 100

    print("\nPercentage Breakdown:")
    print(f"  Token Generation: {token_pct:.1f}%")
    print(f"  Token Selection: {select_pct:.1f}%")
    print(f"  Pruning: {prune_pct:.1f}%")
    print(f"  Retroactive Pruning: {retroactive_prune_pct:.1f}%")
    print(f"  Attention Update: {attn_pct:.1f}%")
    print(f"  Text Formatting: {text_pct:.1f}%")
    print(f"  Other Operations: {other_pct:.1f}%")

    # If sequence tracker is available, report step statistics
    if seq_tracker and seq_tracker.step_count > 0:
        print(f"\nStep Information:")
        print(f"  Total Steps: {seq_tracker.step_count}")
        print(f"  Sequence Length: {seq_tracker.get_length()}")
        print(
            f"  Average Tokens/Step: {(seq_tracker.get_length() - seq_tracker.get_initial_length()) / seq_tracker.step_count:.2f}"
        )

        # Print operation timing by step if detailed timings are available
        if seq_tracker.get_step_stats():
            step_timing = seq_tracker.get_step_stats()
            max_step = max(step_timing.keys())
            print(f"  Detailed step timings: Available for {len(step_timing)} steps")

            # Additional stats if enough steps are available
            if max_step > 5:
                # Get first and last steps for comparison
                first_step = min(step_timing.keys())

                # Compare token generation time growth
                if (
                    "token_generation" in step_timing[first_step]
                    and "token_generation" in step_timing[max_step]
                ):
                    first_time = sum(step_timing[first_step]["token_generation"]) / len(
                        step_timing[first_step]["token_generation"]
                    )
                    last_time = sum(step_timing[max_step]["token_generation"]) / len(
                        step_timing[max_step]["token_generation"]
                    )
                    growth = last_time / first_time if first_time > 0 else 0
                    print(
                        f"  Token generation time growth (step {first_step} to {max_step}): {growth:.2f}x"
                    )


def print_step_performance_analysis(seq_tracker):
    """
    Print a detailed analysis of how performance scales with step count and sequence length.

    Args:
        seq_tracker: SequenceLengthTracker instance with performance data
    """
    step_stats = seq_tracker.get_step_stats()
    if not step_stats:
        print("\nNo step performance data available.")
        return

    print("\n" + "=" * 50)
    print("STEP-BASED PERFORMANCE ANALYSIS")
    print("=" * 50)

    # Get data from the sequence tracker
    initial_length = seq_tracker.get_initial_length()
    max_length = seq_tracker.get_length()
    step_count = seq_tracker.step_count
    length_history = seq_tracker.length_history

    # Sort steps (which are now the keys in sequence_length_stats)
    steps = sorted(step_stats.keys())

    # Print detailed summary of step tracking
    n_steps = len(steps)
    min_step = min(steps) if steps else 0
    max_step = max(steps) if steps else 0

    print(f"\nGeneration Summary:")
    print(f"  Total steps tracked: {step_count}")
    print(f"  Initial sequence length: {initial_length}")
    print(f"  Final sequence length: {max_length}")
    print(f"  Sequence growth: +{max_length - initial_length} tokens")

    # Calculate average tokens per step
    if step_count > 0:
        avg_tokens_per_step = (max_length - initial_length) / step_count
        print(f"  Average tokens per step: {avg_tokens_per_step:.2f}")

    print(f"\nPerformance Tracking:")
    print(f"  Steps tracked: {n_steps}")

    # Extract sequence length at each step to show how it grows
    if n_steps > 0:
        seq_lengths = [step_stats[step].get("sequence_length", 0) for step in steps]
        min_seq_len = min(seq_lengths) if seq_lengths else 0
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        print(f"  Sequence length range: {min_seq_len} to {max_seq_len}")

    # Show step-to-sequence mapping if available
    if n_steps > 1 and n_steps <= 20:
        print("\nStep-to-Sequence Mapping:")
        for step in steps:
            seq_len = step_stats[step].get("sequence_length", "unknown")
            print(f"  Step {step}: Sequence length = {seq_len}")
    elif n_steps > 20:
        # For many steps, show a sampling
        print("\nStep-to-Sequence Mapping (sample):")
        sample_indices = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
        sample_indices = sorted(
            set([max(0, min(i, n_steps - 1)) for i in sample_indices])
        )
        for i in sample_indices:
            step = steps[i]
            seq_len = step_stats[step].get("sequence_length", "unknown")
            print(f"  Step {step}: Sequence length = {seq_len}")

    # For each operation type that supports step tracking
    for operation in [
        "token_generation",
        "token_selection",
        "pruning",
        "retroactive_pruning",
        "attention_update",
    ]:
        # Check if any step has data for this operation
        if any(
            operation in step_stats[step] and step_stats[step][operation]
            for step in steps
        ):

            # Count total timing points for this operation
            total_timing_points = sum(
                len(step_stats[step][operation])
                for step in steps
                if operation in step_stats[step] and step_stats[step][operation]
            )

            print(f"\n{operation.replace('_', ' ').title()} Time by Step:")
            print(f"Total timing points: {total_timing_points}")
            print(
                f"{'Step':<10} {'Seq Len':<10} {'Count':<10} {'Avg Time (ms)':<15} {'Ratio to Initial':<20} {'Visual'}"
            )
            print("-" * 80)

            # Get the initial step for reference
            initial_step = steps[0]
            # Use the first step that has data
            for step in steps:
                if operation in step_stats[step] and step_stats[step][operation]:
                    initial_step = step
                    break

            initial_time = 0
            if (
                operation in step_stats[initial_step]
                and step_stats[initial_step][operation]
            ):
                initial_time = (
                    sum(step_stats[initial_step][operation])
                    / len(step_stats[initial_step][operation])
                    * 1000  # Convert to ms
                )

            # Find the maximum time for scaling the visualization
            max_time_ms = 0
            for step in steps:
                if operation in step_stats[step] and step_stats[step][operation]:
                    avg_time = (
                        sum(step_stats[step][operation])
                        / len(step_stats[step][operation])
                        * 1000
                    )
                    if avg_time > max_time_ms:
                        max_time_ms = avg_time

            # Scale factor for ASCII visualization (max 40 characters)
            scale_factor = 40.0 / max_time_ms if max_time_ms > 0 else 1.0

            # Collect data for trend analysis
            x_data = []  # Steps
            y_data = []  # Times
            z_data = []  # Sequence lengths

            # Only print data for steps that have this operation
            for step in steps:
                if operation in step_stats[step] and step_stats[step][operation]:
                    count = len(step_stats[step][operation])
                    avg_time = sum(step_stats[step][operation]) / count * 1000  # ms
                    ratio = avg_time / initial_time if initial_time > 0 else 0
                    seq_len = step_stats[step].get("sequence_length", "?")

                    # Save for trend analysis
                    x_data.append(step)
                    y_data.append(avg_time)
                    if isinstance(seq_len, (int, float)):
                        z_data.append(seq_len)

                    # Create visual bar
                    bar_length = int(avg_time * scale_factor)
                    visual_bar = "▇" * bar_length

                    print(
                        f"{step:<10} {seq_len:<10} {count:<10} {avg_time:<15.2f} {ratio:<20.2f} {visual_bar}"
                    )

            # If we have enough data points, try to fit a trend line
            if len(x_data) >= 3:
                try:
                    import numpy as np
                    from scipy import stats

                    # Perform linear regression with steps
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        x_data, y_data
                    )

                    print("\nTrend Analysis by Step:")
                    print(f"  Time = {slope:.4f} * step + {intercept:.4f}")
                    print(f"  R² = {r_value**2:.4f} (higher is better fit)")

                    # Project future performance
                    projected_step = 2 * max_step
                    projected_time = slope * projected_step + intercept
                    print(
                        f"  Projected time at step {projected_step}: {projected_time:.2f} ms"
                    )

                    # If we have sequence length data, try correlation with sequence length
                    if len(z_data) >= 3:
                        # Check correlation between time and sequence length
                        corr_time_seq, _ = stats.pearsonr(y_data, z_data)
                        print(
                            f"\nCorrelation between time and sequence length: {corr_time_seq:.4f}"
                        )

                        # Perform linear regression with sequence length if correlation is strong
                        if abs(corr_time_seq) > 0.5:
                            slope_seq, intercept_seq, r_value_seq, _, _ = (
                                stats.linregress(z_data, y_data)
                            )
                            print("\nTrend Analysis by Sequence Length:")
                            print(
                                f"  Time = {slope_seq:.4f} * sequence_length + {intercept_seq:.4f}"
                            )
                            print(f"  R² = {r_value_seq**2:.4f} (higher is better fit)")

                            # Compare which is better predictor: step or sequence length
                            if r_value_seq**2 > r_value**2:
                                print(
                                    "  Sequence length is a better predictor of time than step count"
                                )
                            else:
                                print(
                                    "  Step count is a better predictor of time than sequence length"
                                )

                except ImportError:
                    # Skip trend analysis if scipy is not available
                    print("\nTrend analysis skipped: scipy not available")
                except Exception as e:
                    print(f"Error in trend analysis: {e}")

    # Analyze scaling behavior if we have sequence length data
    if len(steps) > 2 and all(
        "sequence_length" in step_stats[step] for step in [steps[0], steps[-1]]
    ):
        first_step = steps[0]
        last_step = steps[-1]
        first_seq_len = step_stats[first_step].get("sequence_length", 0)
        last_seq_len = step_stats[last_step].get("sequence_length", 0)

        # Only proceed if we have valid sequence length data
        if first_seq_len > 0 and last_seq_len > 0:
            # Find an operation with data for both first and last step
            for operation in [
                "token_generation",
                "token_selection",
                "attention_update",
            ]:
                if (
                    operation in step_stats[first_step]
                    and step_stats[first_step][operation]
                    and operation in step_stats[last_step]
                    and step_stats[last_step][operation]
                ):

                    first_time = sum(step_stats[first_step][operation]) / len(
                        step_stats[first_step][operation]
                    )
                    last_time = sum(step_stats[last_step][operation]) / len(
                        step_stats[last_step][operation]
                    )

                    # Calculate observed ratio
                    observed_ratio = last_time / first_time if first_time > 0 else 0

                    # Calculate theoretical ratios
                    step_ratio = last_step / first_step if first_step > 0 else 0
                    seq_len_ratio = (
                        last_seq_len / first_seq_len if first_seq_len > 0 else 0
                    )
                    quadratic_seq_ratio = (
                        (last_seq_len / first_seq_len) ** 2 if first_seq_len > 0 else 0
                    )

                    print(f"\nScaling Analysis (based on {operation}):")
                    print(
                        f"  Steps increased by factor: {step_ratio:.2f}x (from {first_step} to {last_step})"
                    )
                    print(
                        f"  Sequence length increased by factor: {seq_len_ratio:.2f}x (from {first_seq_len} to {last_seq_len})"
                    )
                    print(f"  Time increased by factor: {observed_ratio:.2f}x")
                    print(
                        f"  Expected for linear-in-sequence scaling (O(n)): {seq_len_ratio:.2f}x"
                    )
                    print(
                        f"  Expected for quadratic-in-sequence scaling (O(n²)): {quadratic_seq_ratio:.2f}x"
                    )

                    # Determine which scaling is closer
                    linear_diff = abs(observed_ratio - seq_len_ratio)
                    quadratic_diff = abs(observed_ratio - quadratic_seq_ratio)

                    if linear_diff < quadratic_diff:
                        print(
                            f"  Scaling appears closer to LINEAR in sequence length (O(n)) with deviation of {linear_diff:.2f}"
                        )
                    else:
                        print(
                            f"  Scaling appears closer to QUADRATIC in sequence length (O(n²)) with deviation of {quadratic_diff:.2f}"
                        )
                        print(
                            "  This suggests attention computation is the bottleneck, which is expected in Transformers."
                        )

                    # Only analyze one operation
                    break

    # Give optimization recommendations based on analysis
    print("\nOptimization Recommendations:")
    print("1. Use a model with sliding window attention for longer contexts")
    print("2. Enable CUDA Flash Attention if using NVIDIA GPUs")
    print("3. Consider using sequence pruning more aggressively as length increases")
    print(
        "4. For retroactive pruning, investigate optimizing the normalization and sigmoid calculations"
    )

    print("=" * 50)


def profile_memory_usage():
    """Profile memory usage."""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    import psutil

    process = psutil.Process(os.getpid())
    print(f"\nCPU Memory Usage: {process.memory_info().rss / 1024**2:.1f} MB")


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


def get_device_dtype(device=None):
    """
    Determine the appropriate dtype based on the device.

    Args:
        device: Device to get dtype for (None uses auto-detected device)

    Returns:
        torch.dtype: The appropriate dtype for the device
    """
    if device is None:
        device = get_device()

    # Use bfloat16 for CUDA, float16 for MPS, and float32 for CPU
    if device == "cuda":
        return torch.bfloat16
    elif device == "mps":
        return torch.float16
    else:  # "cpu"
        return torch.float32


def load_model(model_name_or_path, device=None):
    """
    Load a model from a local path or Hugging Face Hub.

    Args:
        model_name_or_path: Path to local model or name on HF Hub
        device: Device to load model on (None for auto-detect)

    Returns:
        tuple: (wrapped_model, tokenizer)
    """
    if device is None:
        device = get_device()

    print(f"Loading model {model_name_or_path} on {device}...")

    # Auto-detect config to get tokenizer padding token if possible
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Add padding token if missing
    if tokenizer.pad_token is None:
        if hasattr(config, "pad_token_id") and config.pad_token_id is not None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(config.pad_token_id)
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate dtype
    dtype = get_device_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=dtype
    ).to(device)

    # Wrap the model
    wrapped_model = TEMPOModelWrapper(model)

    return wrapped_model, tokenizer


def main():
    """Main entry point for the TEMPO generator."""
    # Parse command line arguments
    args = ArgumentParser.parse_args()

    # Get profiling flags
    enable_profiling = args.pop("profile", False)
    use_cprofile = args.pop("use_cprofile", False)
    profile_output = args.pop("profile_output", "tempo_profile.prof")

    # Get default mode flag
    default_mode = args.pop("default_mode", False)

    # Get parallel token visibility flag (false means tokens are isolated)
    allow_intraset_token_visibility = args.pop("allow_intraset_token_visibility", False)

    # Get preserving isolated tokens flag (true means don't preserve by default)
    no_preserve_isolated_tokens = args.pop("no_preserve_isolated_tokens", False)

    # Create sequence length tracker for performance monitoring
    debug_mode = args.pop("debug_mode", False)
    # Explicitly print debug status at startup
    if debug_mode:
        print(
            "Debug mode is ENABLED - detailed logging will be written to logs/ directory"
        )
    seq_tracker = SequenceLengthTracker(debug=debug_mode or enable_profiling)

    # Make debug_mode globally accessible for imports
    global sequence_tracker
    sequence_tracker = seq_tracker
    sequence_tracker.debug = debug_mode

    # Pass sequence tracker to the token generator module
    try:
        from src.generation.token_generator import set_sequence_tracker
        set_sequence_tracker(seq_tracker)
        print("Successfully set sequence tracker in token_generator module")
    except ImportError as e:
        print(f"Error setting sequence tracker: {e}")

    # Setup profiling if enabled
    profiler = None
    if enable_profiling:
        print("Profiling enabled - performance details will be reported at the end")
        # Apply timing instrumentation with sequence tracker
        patch_classes(seq_tracker)

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
    device = get_device()
    dtype = get_device_dtype()
    print(f"Using device: {device} with {dtype}")

    # Track model loading time if profiling
    model_load_tracker = (
        PerformanceTracker("model_loading", detailed=enable_profiling)
        if enable_profiling
        else None
    )
    if model_load_tracker:
        model_load_tracker.__enter__()

    # Load model and tokenizer with optimized settings
    # model_name = "deepcogito/cogito-v1-preview-qwen-14B"
    model_name = "deepcogito/cogito-v1-preview-llama-3B"
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
        print(
            f"\nDisabling sliding window attention (was set to {config.sliding_window})"
        )
        config.sliding_window = None
    else:
        print(f" Done! ({time.time() - start_time:.2f}s)")
        
    # Explicitly enable attention output in config
    config.output_attentions = True
    print("Explicitly enabled output_attentions in model config")

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
            attn_implementation="eager",  # Use eager attention for better compatibility
        )

    # Close model loading tracker if profiling
    if model_load_tracker:
        model_load_tracker.__exit__(None, None, None)

    # Create the experiment runner
    if not default_mode:
        runner = ExperimentRunner(
            model, tokenizer, device, skip_wrapping=True
        )  # Model already wrapped
        
        # Set debug mode on the experiment runner itself
        if hasattr(runner, 'set_debug_mode'):
            runner.set_debug_mode(debug_mode)
            print(f"ExperimentRunner debug mode {'ENABLED' if debug_mode else 'DISABLED'}")
            
        # Note: Components like token_generator are created within ParallelGenerator
        # which is instantiated in run_experiment() with debug_mode from args
        if debug_mode:
            print("Debug mode will be passed to all generator components through args dictionary")

    # Define a callback function to update the sequence tracker
    def sequence_length_callback(new_length, step_count, prompt_length):
        """Callback to update the sequence tracker with generator step count."""
        # Calculate total sequence length (prompt + generated)
        total_length = prompt_length + new_length

        # Update the sequence tracker with the step and length information
        seq_tracker.increment_step()
        seq_tracker.update_length(total_length)

        # Debug output
        if seq_tracker.debug:
            print(
                f"Step {seq_tracker.step_count}: Sequence length = {total_length} (prompt: {prompt_length}, generated: {new_length})"
            )

            # Print summary every 10 steps
            if seq_tracker.step_count > 0 and seq_tracker.step_count % 10 == 0:
                print(f"\n--- Step {seq_tracker.step_count} Summary ---")
                print(f"Current sequence length: {total_length}")
                print(
                    f"Average tokens per step: {new_length/seq_tracker.step_count:.2f}"
                )
                print("---------------------------\n")

    # Get generation parameters
    prompt = args.get("prompt", "")
    threshold = args.get("threshold", 0.1)
    max_tokens = args.get("max_tokens", 100)

    # Set up generation tracker
    generation_tracker = (
        PerformanceTracker("full_generation", seq_tracker=seq_tracker)
        if enable_profiling
        else None
    )
    if generation_tracker:
        generation_tracker.__enter__()

    start_time = time.time()

    # Run experiment with parameters
    if args.get("early_exit"):
        # Run early exit experiment
        if default_mode:
            print("Error: early_exit not compatible with default_mode.")
            return

        print(
            f"Running early exit experiment with threshold: {threshold}, max tokens: {max_tokens}"
        )
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Make sure debug mode is set
        args["debug_mode"] = debug_mode

        # Adjust params for early exit
        exit_layers = args.get("exit_layers", "")
        confidence_thresholds = args.get("confidence_thresholds", "")

        start_time = time.time()
        results = runner.run_early_exit_experiment(
            prompt=prompt,
            max_tokens=max_tokens,
            threshold=threshold,
            use_pruning=args.get("use_pruning", False),
            exit_layers=exit_layers,
            confidence_thresholds=confidence_thresholds,
            min_steps=args.get("min_steps", 3),
            show_token_ids=args.get("show_token_ids", False),
            debug_mode=debug_mode,
            preserve_all_isolated_tokens=not no_preserve_isolated_tokens,
            isolate_parallel_tokens=not allow_intraset_token_visibility,
        )
    else:
        # Ensure args has the debug_mode flag explicitly set
        args["debug_mode"] = debug_mode
        
        # Run standard experiment
        print(
            f"Running experiment with threshold: {threshold}, max tokens: {max_tokens}"
        )
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        start_time = time.time()
        results = runner.run_experiment(args)

    generation_time = time.time() - start_time

    # Store the generation time in timings dictionary
    if enable_profiling:
        timings["full_generation"] = generation_time

    if generation_tracker:
        generation_tracker.__exit__(None, None, None)

    # Print generation statistics
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    tokens_generated = max_tokens  # Use max_tokens as approximation
    print(f"Average tokens/second: {tokens_generated/generation_time:.2f}")
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
        print_performance_report(seq_tracker)

        # Print step performance analysis
        print_step_performance_analysis(seq_tracker)

        # Print memory usage
        profile_memory_usage()


if __name__ == "__main__":
    main()
