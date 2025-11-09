#!/usr/bin/env python3
"""Simplified persistent runner - loads model once, reuses for all experiments."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
import time
import logging
from src.utils.model_utils import load_tempo_components
from src.experiments.experiment_runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePersistentRunner:
    """Layer 1: Model persistence - load once, run many experiments."""

    def __init__(self):
        logger.info("Loading model (one time only)...")
        start = time.time()

        # Load components ONCE
        components = load_tempo_components(
            model_id="deepcogito/cogito-v1-preview-llama-3B",
            device="mps",
            load_model_wrapper=True,
            load_token_generator=False,  # Skip legacy component
            load_parallel_generator=False,  # Don't need this either
            debug_mode=False
        )

        # Store components
        self.model_wrapper = components["model_wrapper"]
        self.tokenizer = components["tokenizer"]
        self.device = "mps"

        logger.info(f"✓ Model loaded in {time.time() - start:.2f}s (will persist)")

    def run_experiment(self, prompt: str, seed: int, isolated: bool = True, max_tokens: int = 10) -> Dict:
        """Run one experiment with persistent model."""
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create fresh runner (but reuses same model)
        runner = ExperimentRunner(
            model=self.model_wrapper,
            tokenizer=self.tokenizer,
            device=self.device,
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

        # Cleanup runner (but keep model)
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


if __name__ == "__main__":
    runner = SimplePersistentRunner()

    # Run multiple experiments without reloading model
    for i in range(3):
        result = runner.run_experiment("The cat sat on the", seed=42+i)
        print(f"Experiment {i+1}: {result['generated_text'][:50]}")
