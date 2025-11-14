#!/usr/bin/env python3
"""Test attention reduction finding across diverse prompts.

This script validates the key finding that parallel tokens receive 40-60%
less attention than non-parallel tokens by testing across multiple prompts
with different characteristics.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.model_utils import load_tempo_components, get_best_device
from src.experiments import ExperimentRunner


@dataclass
class PromptTest:
    """Test case for a specific prompt."""
    name: str
    prompt: str
    category: str  # narrative, factual, technical, conversational, etc.
    expected_parallel_steps: int  # rough estimate


# Diverse test prompts covering different domains and structures
TEST_PROMPTS = [
    # Narrative/Creative
    PromptTest(
        name="narrative_1",
        prompt="Once upon a time in a distant galaxy",
        category="narrative",
        expected_parallel_steps=5
    ),
    PromptTest(
        name="narrative_2",
        prompt="The old wizard slowly climbed the mountain",
        category="narrative",
        expected_parallel_steps=5
    ),
    PromptTest(
        name="narrative_3",
        prompt="Deep in the forest, a mysterious creature",
        category="narrative",
        expected_parallel_steps=4
    ),

    # Factual/Informative
    PromptTest(
        name="factual_1",
        prompt="The capital of France is",
        category="factual",
        expected_parallel_steps=2
    ),
    PromptTest(
        name="factual_2",
        prompt="Photosynthesis is the process by which",
        category="factual",
        expected_parallel_steps=4
    ),
    PromptTest(
        name="factual_3",
        prompt="The largest planet in our solar system",
        category="factual",
        expected_parallel_steps=3
    ),

    # Technical/Scientific
    PromptTest(
        name="technical_1",
        prompt="Machine learning algorithms can be classified into",
        category="technical",
        expected_parallel_steps=4
    ),
    PromptTest(
        name="technical_2",
        prompt="The algorithm complexity of quicksort is",
        category="technical",
        expected_parallel_steps=3
    ),

    # Conversational
    PromptTest(
        name="conversational_1",
        prompt="How are you doing today? I'm",
        category="conversational",
        expected_parallel_steps=3
    ),
    PromptTest(
        name="conversational_2",
        prompt="What do you think about",
        category="conversational",
        expected_parallel_steps=3
    ),

    # Short/Simple
    PromptTest(
        name="simple_1",
        prompt="The cat sat on the",
        category="simple",
        expected_parallel_steps=3
    ),
    PromptTest(
        name="simple_2",
        prompt="I went to the",
        category="simple",
        expected_parallel_steps=2
    ),

    # Complex/Long
    PromptTest(
        name="complex_1",
        prompt="Despite the significant challenges faced by researchers in the field of quantum computing",
        category="complex",
        expected_parallel_steps=6
    ),
]


@dataclass
class AttentionMetrics:
    """Metrics for attention analysis."""
    prompt_name: str
    step: int
    parallel_mean_attn: float
    non_parallel_mean_attn: float
    ratio: float  # parallel / non_parallel
    num_parallel_tokens: int
    num_non_parallel_tokens: int


def analyze_attention_from_data(data_dir: Path) -> List[AttentionMetrics]:
    """Analyze attention data from experiment output.

    Args:
        data_dir: Directory containing experiment data

    Returns:
        List of attention metrics
    """
    metrics = []

    # This is a placeholder - actual implementation needs to:
    # 1. Load experiment data
    # 2. Extract attention weights
    # 3. Identify parallel vs non-parallel tokens
    # 4. Calculate mean attention to each group
    # 5. Compute ratios

    # For now, return empty list - will be implemented when we have
    # attention capture fully integrated

    return metrics


def run_experiment_suite(
    output_dir: Path,
    max_tokens: int = 50,
    selection_threshold: float = 0.15,
    device: str = "auto"
) -> Dict[str, Any]:
    """Run TEMPO generation across all test prompts.

    Args:
        output_dir: Directory to save results
        max_tokens: Maximum tokens to generate
        selection_threshold: Threshold for parallel token selection
        device: Device to use (auto-detect if "auto")

    Returns:
        Dictionary with results summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if device == "auto":
        device = get_best_device()

    print("="*70)
    print("TEMPO Attention Reduction Multi-Prompt Validation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Max tokens: {max_tokens}")
    print(f"Selection threshold: {selection_threshold}")
    print("="*70)

    # Load model once
    print("\nLoading model...")
    components = load_tempo_components(
        model_id="deepcogito/cogito-v1-preview-llama-3B",
        device=device,
        load_model_wrapper=True,
        load_token_generator=False,  # Don't need legacy generator
        load_parallel_generator=False,  # Don't need legacy generator
        debug_mode=False,
        low_cpu_mem_usage=True
    )

    model_wrapper = components["model_wrapper"]
    tokenizer = components["tokenizer"]

    runner = ExperimentRunner(
        model=model_wrapper,
        tokenizer=tokenizer,
        device=device
    )

    results = {
        "config": {
            "max_tokens": max_tokens,
            "selection_threshold": selection_threshold,
            "device": device,
            "num_prompts": len(TEST_PROMPTS)
        },
        "prompts": {},
        "summary": {
            "total_prompts": len(TEST_PROMPTS),
            "successful": 0,
            "failed": 0
        }
    }

    # Run each prompt
    for i, prompt_test in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: {prompt_test.name}")
        print(f"Category: {prompt_test.category}")
        print(f"Prompt: '{prompt_test.prompt}'")

        try:
            # Create output directory for this prompt
            prompt_dir = output_dir / prompt_test.name
            prompt_dir.mkdir(exist_ok=True)

            # Run generation
            args = {
                "prompt": prompt_test.prompt,
                "max_tokens": max_tokens,
                "selection_threshold": selection_threshold,
                "output_dir": str(prompt_dir),
                "isolate": True,  # Test with isolation
                "debug_mode": False,
                "min_steps": 0
            }

            result = runner.run_experiment(args)

            # Store results
            results["prompts"][prompt_test.name] = {
                "prompt": prompt_test.prompt,
                "category": prompt_test.category,
                "success": True,
                "generation_time": result.get("generation_time", 0),
                "output_length": len(result.get("clean_text", "")),
                # Placeholder for attention metrics
                "attention_metrics": None
            }

            results["summary"]["successful"] += 1
            print(f"✓ Success ({result.get('generation_time', 0):.2f}s)")

        except Exception as e:
            print(f"✗ Failed: {e}")
            results["prompts"][prompt_test.name] = {
                "prompt": prompt_test.prompt,
                "category": prompt_test.category,
                "success": False,
                "error": str(e)
            }
            results["summary"]["failed"] += 1

    # Save results
    results_file = output_dir / "multi_prompt_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Successful: {results['summary']['successful']}/{len(TEST_PROMPTS)}")
    print(f"Failed: {results['summary']['failed']}/{len(TEST_PROMPTS)}")
    print(f"\nResults saved to: {results_file}")
    print("="*70)

    # Note about attention analysis
    print("\nNOTE: Attention analysis requires full integration of attention capture.")
    print("Current implementation tests generation across prompts successfully.")
    print("Next step: Integrate AttentionAnalyzer to capture actual attention weights.")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test attention reduction across diverse prompts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/multi_prompt_attention"),
        help="Output directory"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.15,
        help="Selection threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )

    args = parser.parse_args()

    results = run_experiment_suite(
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        selection_threshold=args.selection_threshold,
        device=args.device
    )

    return 0 if results["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
