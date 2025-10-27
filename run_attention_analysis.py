#!/usr/bin/env python3
"""Run attention pattern analysis experiments.

This script captures attention patterns during TEMPO generation and
provides mechanistic interpretability insights.
"""

import sys
import argparse
from pathlib import Path
import json

from src.experiments.attention_analyzer import AttentionAnalyzer
from src.experiments import ArgumentParser as TempoArgumentParser, ExperimentRunner
from src.utils.model_utils import load_tempo_components


def main():
    parser = argparse.ArgumentParser(description="Analyze TEMPO attention patterns")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--prompt", type=str, default="The cat sat on the")
    parser.add_argument("--max-tokens", type=int, default=15)
    parser.add_argument("--selection-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="./attention_analysis")
    parser.add_argument("--analyze-step", type=int, help="Specific step to analyze in detail")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TEMPO Attention Pattern Analysis")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Threshold: {args.selection_threshold}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()

    # Load model and components
    print("Loading model...")
    components = load_tempo_components(
        model_id="deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_model_wrapper=True,
        debug_mode=False,
    )

    model_wrapper = components["model_wrapper"]
    tokenizer = components["tokenizer"]

    # Create attention analyzer
    from src.infrastructure.tokenization.tokenizer_adapter import TokenizerAdapter
    tokenizer_adapter = TokenizerAdapter(tokenizer, device="mps")
    analyzer = AttentionAnalyzer(tokenizer_adapter, debug_mode=False)

    print("Model loaded. Starting generation with attention capture...")
    print()

    # TODO: Hook into generation to capture attention
    # For now, this is a framework - we'll need to modify the generation
    # code to actually capture attention weights

    print("NOTE: Attention capture requires hooking into the generation process.")
    print("This is a framework for future implementation.")
    print()
    print("To fully implement:")
    print("1. Modify TokenGeneratorImpl to expose attention weights")
    print("2. Hook AttentionAnalyzer into the generation loop")
    print("3. Capture attention after each forward pass")

    # Example of what we would do with captured data
    print()
    print("Example Analysis (with captured data):")
    print("- Attention summary statistics")
    print("- Per-step parallel token attention")
    print("- Cross-parallel-token attention (should be near-zero if isolated)")
    print("- Top attended tokens for each parallel alternative")

    print()
    print(f"Analysis framework ready at: {output_dir}")


if __name__ == "__main__":
    main()
