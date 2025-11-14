#!/usr/bin/env python3
"""Run validation across all 62 prompts in the expanded suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from expanded_prompt_suite import get_all_prompts
from test_attention_across_prompts import run_experiment_suite


def main():
    """Run expanded validation suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Run expanded attention validation")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/expanded_attention_validation"))
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--selection-threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    all_prompts = get_all_prompts()

    print(f"="*70)
    print(f"EXPANDED ATTENTION VALIDATION")
    print(f"="*70)
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    print(f"Selection threshold: {args.selection_threshold}")
    print(f"Estimated time: ~{len(all_prompts) * 5 / 60:.1f} minutes")
    print(f"="*70)

    # This will use the existing test infrastructure
    # but we need to temporarily override the TEST_PROMPTS
    import test_attention_across_prompts as tap
    tap.TEST_PROMPTS = all_prompts

    results = run_experiment_suite(
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        selection_threshold=args.selection_threshold,
        device=args.device,
        capture_attention=True
    )

    print(f"\n" + "="*70)
    print(f"EXPANDED VALIDATION COMPLETE")
    print(f"="*70)
    print(f"Total prompts tested: {len(all_prompts)}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Attention captured: {results['summary']['attention_captured']}")
    print(f"="*70)

    return 0 if results["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
