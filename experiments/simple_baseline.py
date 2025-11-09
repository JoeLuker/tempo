#!/usr/bin/env python3
"""Simplified baseline for testing - uses existing ExperimentRunner."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
import time
from src.utils.model_utils import load_tempo_components
from src.experiments.experiment_runner import ExperimentRunner


def run_baseline(prompt: str, seed: int, isolated: bool = True, max_tokens: int = 10) -> Dict:
    """Run baseline experiment (loads model fresh each time)."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load components fresh
    components = load_tempo_components(
        model_id="deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_model_wrapper=True,
        debug_mode=False
    )

    # Create runner
    runner = ExperimentRunner(
        model=components["model_wrapper"],
        tokenizer=components["tokenizer"],
        device="mps",
        skip_wrapping=True  # Already wrapped
    )

    # Run experiment
    args = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "selection_threshold": 0.1,
        "use_retroactive_removal": False,
        "isolate": isolated,
        "output_dir": "/tmp/tempo_test",
        "debug_mode": False,
        "seed": seed
    }

    result = runner.run_experiment(args)

    # Cleanup
    del runner
    del components
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    result = run_baseline("The cat sat on the", seed=42)
    print(f"Generated: {result['generated_text'][:100]}")
