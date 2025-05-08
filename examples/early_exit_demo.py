#!/usr/bin/env python3
"""
Early Exit Transformer Demo

This script demonstrates the use of the Early Exit Transformer with TEMPO.
The early exit capability allows the model to terminate processing before
completing all layers when sufficient confidence is reached, dramatically
improving inference speed.
"""

import torch
import argparse
import time
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.modeling.early_exit_transformer import EarlyExitTransformer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Early Exit Transformer Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="deepcogito/cogito-v1-preview-llama-3B",
        help="Model name or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the difference between a llama and a cow",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_length", type=int, default=50, help="Maximum length for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for computation (cpu, cuda, mps, auto)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for verbose logging"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with standard generation"
    )
    parser.add_argument(
        "--exit_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices for early exits (e.g., '3,7,11,15')",
    )
    parser.add_argument(
        "--confidence_thresholds",
        type=str,
        default=None,
        help="Comma-separated list of confidence thresholds for exit layers (e.g., '0.7,0.8,0.9,0.95')",
    )
    return parser.parse_args()


from src.utils.model_utils import get_best_device, get_device_dtype, load_model

def detect_device(requested_device=None):
    """
    Detect the best available device.
    
    This function is kept for backward compatibility.
    Use src.utils.model_utils.get_best_device() for new code.
    """
    if requested_device and requested_device != "auto":
        return requested_device
    return get_best_device()


def main():
    """Run the early exit transformer demo."""
    args = parse_arguments()

    # Auto-detect device if not specified
    device = detect_device(args.device)
    print(f"Using device: {device}")

    try:
        print(f"Loading model {args.model}...")
        
        # Use centralized model loading function
        model, tokenizer = load_model(
            model_id=args.model,
            device=device,
            load_tokenizer=True,
            use_fast_tokenizer=True,
            low_cpu_mem_usage=True,
            # Don't pass debug mode option to avoid log spam
        )
        
        print(f"Model class: {model.__class__.__name__}")
        print(f"Model loaded on device: {next(model.parameters()).device}")

        # Process exit layers if provided
        exit_layers = None
        if args.exit_layers:
            exit_layers = [int(layer) for layer in args.exit_layers.split(",")]
            print(f"Using custom exit layers: {exit_layers}")
        else:
            # Default to checking layers 6, 12, 18, 24 for Llama models
            num_layers = 0
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                num_layers = len(model.model.layers)
                exit_layers = [
                    int(num_layers * ratio) for ratio in [0.2, 0.4, 0.6, 0.8, 0.95]
                ]
                print(
                    f"Using default exit layers based on model architecture: {exit_layers}"
                )

        # Process confidence thresholds if provided
        confidence_thresholds = None
        if args.confidence_thresholds:
            confidence_thresholds = [
                float(t) for t in args.confidence_thresholds.split(",")
            ]
            print(f"Using custom confidence thresholds: {confidence_thresholds}")
        else:
            # Very low thresholds for testing
            confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
            print(f"Using test confidence thresholds: {confidence_thresholds}")

        # Validate thresholds are in range 0-1
        if confidence_thresholds and any(t < 0 or t > 1 for t in confidence_thresholds):
            print("Warning: Confidence thresholds should be between 0 and 1")

        # Ensure thresholds align with layers
        if (
            exit_layers
            and confidence_thresholds
            and len(confidence_thresholds) != len(exit_layers)
        ):
            print(
                f"Warning: Number of confidence thresholds ({len(confidence_thresholds)}) doesn't match number of exit layers ({len(exit_layers)})"
            )
            print("Using default threshold calculation instead")
            confidence_thresholds = None

        # Create early exit transformer wrapper
        print("Creating early exit transformer...")
        early_exit_model = EarlyExitTransformer(
            model=model,
            exit_layers=exit_layers,
            confidence_thresholds=confidence_thresholds,
            debug_mode=args.debug,
        )

        # Print model architecture information
        print(f"Model architecture: {early_exit_model.model_type}")

        # Print configuration
        num_layers = early_exit_model._get_num_layers()
        print(f"Model has {num_layers} layers")
        print(f"Early exit layers: {early_exit_model.exit_layers}")
        print(
            f"Confidence thresholds: {[f'{t:.2f}' for t in early_exit_model.confidence_thresholds]}"
        )

        # Tokenize input
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

        # Generate with early exits
        print(f"\nGenerating with prompt: '{args.prompt}'")
        print("Using early exit transformer...")

        start_time = time.time()
        output_ids = early_exit_model.generate_with_early_exits(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=args.max_length,
        )
        early_exit_time = time.time() - start_time

        early_exit_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"\nEarly exit generation completed in {early_exit_time:.4f} seconds")
        print(f"Result: {early_exit_text}")

        # Print exit statistics
        early_exit_model.print_exit_statistics()

        # Compare with standard generation if requested
        if args.compare:
            print("\nComparing with standard generation...")

            # Model is already on the correct device thanks to load_model
            # No need to call model.to(device) again

            # Generate with standard approach
            start_time = time.time()
            output_ids = model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + args.max_length,
                do_sample=False,
            )
            standard_time = time.time() - start_time

            standard_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"\nStandard generation completed in {standard_time:.4f} seconds")
            print(f"Result: {standard_text}")

            # Calculate speedup
            speedup = standard_time / early_exit_time
            print(f"\nEarly exit speedup: {speedup:.2f}x")

            # Compare outputs
            same_output = standard_text == early_exit_text
            print(f"Outputs match: {same_output}")

            if not same_output:
                # Find the point of divergence
                min_len = min(len(standard_text), len(early_exit_text))
                divergence_point = min_len
                for i in range(min_len):
                    if standard_text[i] != early_exit_text[i]:
                        divergence_point = i
                        break

                # Show context around divergence
                context_size = 20
                start_idx = max(0, divergence_point - context_size)
                end_idx = min(min_len, divergence_point + context_size)

                print(f"\nOutputs diverge at position {divergence_point}:")
                print(f"Standard: ...{standard_text[start_idx:end_idx]}...")
                print(f"Early Exit: ...{early_exit_text[start_idx:end_idx]}...")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nDetailed traceback:")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
