#!/usr/bin/env python3
"""Compare isolated vs visible parallel token modes.

This script runs the same prompt with both isolation modes and
shows the differences in generation behavior.
"""

import sys
import time
from pathlib import Path
import json
from typing import Dict

from src.experiments import ArgumentParser, ExperimentRunner
from src.utils.model_utils import load_tempo_components


class IsolationComparison:
    """Compares isolated vs visible parallel token modes."""

    def __init__(self, model_wrapper, tokenizer, device: str):
        """Initialize the comparison.

        Args:
            model_wrapper: Wrapped model
            tokenizer: Tokenizer
            device: Device string
        """
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.device = device

    def run_comparison(
        self,
        prompt: str,
        max_tokens: int,
        selection_threshold: float,
        seed: int = 42
    ) -> Dict:
        """Run generation with both isolation modes.

        Args:
            prompt: Text prompt
            max_tokens: Max tokens to generate
            selection_threshold: Selection threshold
            seed: Random seed

        Returns:
            Dictionary with results from both modes
        """
        print("=" * 70)
        print("TEMPO Parallel Token Isolation Comparison")
        print("=" * 70)
        print(f"Prompt: {prompt}")
        print(f"Max Tokens: {max_tokens}")
        print(f"Threshold: {selection_threshold}")
        print(f"Seed: {seed}")
        print("=" * 70)
        print()

        # Run with isolation (default)
        print("Running with ISOLATED parallel tokens...")
        print("-" * 70)
        isolated_result = self._run_single(
            prompt=prompt,
            max_tokens=max_tokens,
            selection_threshold=selection_threshold,
            seed=seed,
            allow_visibility=False
        )
        print()

        # Run with visibility
        print("Running with VISIBLE parallel tokens...")
        print("-" * 70)
        visible_result = self._run_single(
            prompt=prompt,
            max_tokens=max_tokens,
            selection_threshold=selection_threshold,
            seed=seed,
            allow_visibility=True
        )
        print()

        # Compare results
        comparison = self._compare_results(isolated_result, visible_result)

        return {
            'isolated': isolated_result,
            'visible': visible_result,
            'comparison': comparison
        }

    def _run_single(
        self,
        prompt: str,
        max_tokens: int,
        selection_threshold: float,
        seed: int,
        allow_visibility: bool
    ) -> Dict:
        """Run a single generation experiment.

        Args:
            prompt: Text prompt
            max_tokens: Max tokens
            selection_threshold: Threshold
            seed: Random seed
            allow_visibility: Whether to allow parallel token visibility

        Returns:
            Generation result dictionary
        """
        runner = ExperimentRunner(
            model=self.model_wrapper,
            tokenizer=self.tokenizer,
            device=self.device
        )

        args_dict = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'selection_threshold': selection_threshold,
            'seed': seed,
            'allow_intraset_token_visibility': allow_visibility,
            'use_retroactive_removal': False,  # Disable for cleaner comparison
            'debug_mode': False,
            'output_json': False,
        }

        start_time = time.time()
        result = runner.run_experiment(args_dict)
        elapsed = time.time() - start_time

        return {
            'mode': 'visible' if allow_visibility else 'isolated',
            'generated_text': result.get('generated_text', ''),
            'raw_generated_text': result.get('raw_generated_text', ''),
            'generation_time': elapsed,
            'config': args_dict,
        }

    def _compare_results(self, isolated: Dict, visible: Dict) -> Dict:
        """Compare results from both modes.

        Args:
            isolated: Results from isolated mode
            visible: Results from visible mode

        Returns:
            Comparison metrics
        """
        comparison = {
            'time_difference': visible['generation_time'] - isolated['generation_time'],
            'time_ratio': visible['generation_time'] / isolated['generation_time'] if isolated['generation_time'] > 0 else 0,
            'isolated_text': isolated['raw_generated_text'],
            'visible_text': visible['raw_generated_text'],
            'texts_identical': isolated['raw_generated_text'] == visible['raw_generated_text'],
        }

        # Simple text length comparison
        comparison['isolated_length'] = len(isolated['raw_generated_text'])
        comparison['visible_length'] = len(visible['raw_generated_text'])

        return comparison

    def print_comparison(self, results: Dict):
        """Print formatted comparison.

        Args:
            results: Results dictionary from run_comparison
        """
        print()
        print("=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        print()

        # Show generated text side by side
        print("ISOLATED MODE OUTPUT:")
        print("-" * 70)
        print(results['isolated']['generated_text'])
        print()

        print("VISIBLE MODE OUTPUT:")
        print("-" * 70)
        print(results['visible']['generated_text'])
        print()

        # Show metrics
        comp = results['comparison']
        print("METRICS:")
        print("-" * 70)
        print(f"Isolated Generation Time: {results['isolated']['generation_time']:.3f}s")
        print(f"Visible Generation Time:  {results['visible']['generation_time']:.3f}s")
        print(f"Time Difference:          {comp['time_difference']:+.3f}s")
        print(f"Time Ratio:               {comp['time_ratio']:.2f}x")
        print()
        print(f"Isolated Text Length: {comp['isolated_length']} chars")
        print(f"Visible Text Length:  {comp['visible_length']} chars")
        print(f"Texts Identical:      {comp['texts_identical']}")
        print()

        # Show raw text comparison
        print("RAW TEXT COMPARISON:")
        print("-" * 70)
        print("ISOLATED:")
        print(f"  {comp['isolated_text'][:200]}...")
        print()
        print("VISIBLE:")
        print(f"  {comp['visible_text'][:200]}...")
        print()

    def export_comparison(self, results: Dict, output_path: Path):
        """Export comparison to JSON.

        Args:
            results: Results dictionary
            output_path: Path to save JSON
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results exported to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare isolated vs visible parallel token modes"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The cat sat on the",
        help="Text prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.1,
        help="Selection threshold"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./isolation_comparison.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    # Load model (simplified for new architecture)
    print("Loading model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "deepcogito/cogito-v1-preview-llama-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="mps",
        low_cpu_mem_usage=True
    )

    from src.modeling.model_wrapper import TEMPOModelWrapper
    model_wrapper = TEMPOModelWrapper(model=model, device="mps")

    components = {
        "model_wrapper": model_wrapper,
        "tokenizer": tokenizer
    }

    # Run comparison
    comparison = IsolationComparison(
        model_wrapper=components["model_wrapper"],
        tokenizer=components["tokenizer"],
        device="mps"
    )

    results = comparison.run_comparison(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        selection_threshold=args.selection_threshold,
        seed=args.seed
    )

    # Print results
    comparison.print_comparison(results)

    # Export
    output_path = Path(args.output)
    comparison.export_comparison(results, output_path)


if __name__ == "__main__":
    main()
