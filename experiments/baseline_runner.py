#!/usr/bin/env python3
"""Baseline runner - loads model fresh for each experiment (no persistence)."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
import logging
from src.utils.model_utils import load_tempo_components
from src.experiments.experiment_runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_baseline_experiment(prompt: str, seed: int, isolated: bool = True, max_tokens: int = 10) -> Dict:
    """
    Run one experiment with FRESH model load (baseline - no optimization).

    This is the "true" baseline that loads the model from scratch each time,
    exactly like running run_tempo.py multiple times.
    """
    logger.info(f"Loading model (fresh - baseline)...")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load components FRESH (no reuse)
    components = load_tempo_components(
        model_id="deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_model_wrapper=True,
        load_token_generator=False,  # Skip legacy component
        load_parallel_generator=False,  # Don't need this either
        debug_mode=False
    )

    model_wrapper = components["model_wrapper"]
    tokenizer = components["tokenizer"]

    # Create runner
    runner = ExperimentRunner(
        model=model_wrapper,
        tokenizer=tokenizer,
        device="mps",
        skip_wrapping=True
    )

    # Run experiment
    args = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "selection_threshold": 0.1,
        "use_retroactive_removal": False,
        "isolate": isolated,
        "output_dir": "/tmp/tempo_baseline",
        "debug_mode": False,
        "seed": seed
    }

    result = runner.run_experiment(args)

    # Cleanup (free memory)
    del runner
    del model_wrapper
    del tokenizer
    del components
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
