#!/usr/bin/env python3
"""
Test if attention mask is actually being used during generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from src.modeling.model_wrapper import TEMPOModelWrapper
from src.experiments.experiment_runner import ExperimentRunner
from src.utils.model_utils import get_best_device, get_device_dtype

# Monkey-patch to add logging
original_forward = None

def patched_forward(self, *args, **kwargs):
    """Patched forward to log if attention_mask is provided."""
    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
        mask = kwargs['attention_mask']
        print(f"  [MASK] attention_mask provided: shape={mask.shape}, unique_values={torch.unique(mask).tolist()[:10]}")
    else:
        print(f"  [MASK] NO attention_mask provided")

    return original_forward(*args, **kwargs)


def main():
    """Run a test with debug logging."""
    print("="*80)
    print("TESTING ATTENTION MASK USAGE")
    print("="*80)

    # Load model
    device_str = get_best_device()
    dtype = get_device_dtype(device_str)
    model_id = "deepcogito/cogito-v1-preview-llama-3B"

    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_str,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )

    # Patch the forward method
    global original_forward
    original_forward = model.forward
    model.forward = lambda *args, **kwargs: patched_forward(model, *args, **kwargs)

    model_wrapper = TEMPOModelWrapper(model=model, device=device_str)

    print(f"\nModel loaded, running isolated mode test...")

    # Run experiment
    runner = ExperimentRunner(model=model_wrapper, tokenizer=tokenizer, device=device_str)

    config = {
        "prompt": "The cat sat",
        "max_tokens": 3,
        "selection_threshold": 0.1,
        "allow_intraset_token_visibility": False,  # ISOLATED
        "output_dir": "./test_mask_output"
    }

    print("\n" + "="*80)
    print("GENERATION STARTING (watch for [MASK] lines)")
    print("="*80 + "\n")

    result = runner.run_experiment(config)

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Generated: {result['raw_generated_text']}")


if __name__ == "__main__":
    main()
