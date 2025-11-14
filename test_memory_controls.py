#!/usr/bin/env python3
"""Test script to verify memory controls are working.

This script runs a simple TEMPO generation with memory monitoring to ensure
the 36GB limit is respected.
"""

import sys
import torch
from src.utils.memory_monitor import MemoryMonitor
from src.utils.model_utils import load_tempo_components

def test_memory_controls():
    """Test memory controls with a simple generation."""
    print("="*70)
    print("TEMPO Memory Controls Test")
    print("="*70)

    # Configuration
    MAX_MEMORY_GB = 36.0
    model_name = "deepcogito/cogito-v1-preview-llama-3B"
    prompt = "The future of artificial intelligence is"
    max_tokens = 50
    selection_threshold = 0.1

    # Initialize memory monitor
    print(f"\n1. Initializing memory monitor (limit: {MAX_MEMORY_GB}GB)")
    memory_monitor = MemoryMonitor(max_memory_gb=MAX_MEMORY_GB, device="cuda")
    memory_monitor.log_memory_stats("Initial state")

    try:
        # Load model
        print(f"\n2. Loading model: {model_name}")
        components = load_tempo_components(
            model_id=model_name,
            device="cuda",
            load_model_wrapper=True,
            load_token_generator=False,
            load_parallel_generator=False,
            debug_mode=False,
            low_cpu_mem_usage=True
        )

        model_wrapper = components["model_wrapper"]
        tokenizer = components["tokenizer"]

        memory_monitor.log_memory_stats("After model load")
        memory_monitor.check_memory_limit("model loading")

        # Calculate max parallel tokens
        print(f"\n3. Calculating memory-safe parallel token limit")
        max_parallel_tokens = memory_monitor.calculate_max_parallel_tokens(
            sequence_length=max_tokens
        )
        print(f"   Max parallel tokens: {max_parallel_tokens}")

        # Check available memory
        available = memory_monitor.get_available_memory_gb()
        print(f"   Available memory: {available:.2f}GB")

        # Estimate KV cache memory
        num_layers = len(model_wrapper.model.model.layers)
        num_heads = model_wrapper.model.config.num_attention_heads
        head_dim = model_wrapper.model.config.hidden_size // num_heads

        kv_memory = memory_monitor.estimate_kv_cache_memory(
            batch_size=1,
            sequence_length=max_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )
        print(f"   Estimated KV cache: {kv_memory:.2f}GB")

        # Estimate parallel batch memory
        parallel_memory = memory_monitor.estimate_parallel_batch_memory(
            num_parallel_tokens=max_parallel_tokens,
            sequence_length=max_tokens,
            vocab_size=128256,
            hidden_size=3072
        )
        print(f"   Estimated parallel batch: {parallel_memory:.2f}GB")

        total_estimated = kv_memory + parallel_memory
        print(f"   Total estimated: {total_estimated:.2f}GB")

        # Run a simple generation
        print(f"\n4. Running test generation")
        print(f"   Prompt: '{prompt}'")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Selection threshold: {selection_threshold}")

        from src.experiments import ExperimentRunner

        runner = ExperimentRunner(
            model=model_wrapper,
            tokenizer=tokenizer,
            device="cuda"
        )

        args = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "selection_threshold": selection_threshold,
            "max_memory_gb": MAX_MEMORY_GB,
            "max_parallel_tokens": max_parallel_tokens,
            "debug_mode": False,
            "output_dir": "./output/memory_test",
            "isolate": True,
            "min_steps": 0
        }

        result = runner.run_experiment(args)

        memory_monitor.log_memory_stats("After generation")

        # Verify we stayed under limit
        final_usage = memory_monitor.get_current_usage_gb()
        if final_usage <= MAX_MEMORY_GB:
            print(f"\n✓ SUCCESS: Memory usage {final_usage:.2f}GB <= {MAX_MEMORY_GB}GB")
        else:
            print(f"\n✗ FAILED: Memory usage {final_usage:.2f}GB > {MAX_MEMORY_GB}GB")
            return False

        print(f"\n5. Results:")
        print(f"   Generated text: {result['clean_text'][:100]}...")
        print(f"   Generation time: {result['generation_time']:.2f}s")

        print("\n" + "="*70)
        print("Memory controls test PASSED")
        print("="*70)

        return True

    except MemoryError as e:
        print(f"\n✗ Memory limit exceeded: {e}")
        memory_monitor.log_memory_stats("At memory error")
        return False

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_memory_controls()
    sys.exit(0 if success else 1)
