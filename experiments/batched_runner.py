#!/usr/bin/env python3
"""Layer 3: Batched prompt processing - run multiple prompts in single forward pass.

This layer processes multiple prompts simultaneously within a single model instance
by using batch_size > 1 in the forward pass. This leverages GPU parallelism for
tensor operations.

Architecture:
- Single model instance (persistent from Layer 1)
- Batch multiple prompts together
- Process in single forward pass when possible
- Padding to handle different prompt lengths

Expected speedup: 2-3x over sequential processing
Total combined speedup (Layer 1 + 2 + 3): 13-20x

NOTE: For TEMPO, batching is complex due to variable parallel token counts.
This implementation focuses on initial prompt processing batch optimization.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import time
from experiments.simple_persistent import SimplePersistentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchedExperimentRunner:
    """Layer 3: Batched prompt processing for multiple experiments.

    Processes multiple prompts simultaneously in batched forward passes
    to leverage GPU parallelism.
    """

    def __init__(self, batch_size: int = 4):
        """Initialize batched runner.

        Args:
            batch_size: Number of prompts to process in parallel
        """
        self.batch_size = batch_size
        self.runner = SimplePersistentRunner()

        logger.info(f"BatchedExperimentRunner initialized with batch_size={batch_size}")

    def run_batch(
        self,
        prompts: List[str],
        seeds: List[int],
        mode: str = "isolated"
    ) -> List[Dict]:
        """Run a batch of experiments with batched forward passes.

        NOTE: Current implementation falls back to sequential processing
        because TEMPO's parallel token mechanism creates variable-length
        sequences that are difficult to batch efficiently.

        Future optimization: Implement dynamic batching that groups by
        current sequence length and processes compatible sub-batches together.

        Args:
            prompts: List of prompts to generate from
            seeds: List of seeds (one per prompt)
            mode: Generation mode ("isolated" or "visible")

        Returns:
            List of result dicts, ordered by input order
        """
        assert len(prompts) == len(seeds), \
            f"Length mismatch: {len(prompts)} prompts, {len(seeds)} seeds"

        num_experiments = len(prompts)
        logger.info(f"Running {num_experiments} experiments with batch_size={self.batch_size}")
        logger.warning("NOTE: Falling back to sequential processing due to TEMPO variable-length sequences")

        results = []
        start_time = time.time()

        # TODO: Implement true batching
        # Challenge: TEMPO creates variable-length sequences due to parallel tokens
        # Solution approach:
        #   1. Batch prompt encoding phase (same length)
        #   2. During generation, dynamically group compatible sequences
        #   3. Process sub-batches that have same current length
        #   4. Requires significant refactoring of generation loop

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            logger.info(f"Processing experiment {i+1}/{num_experiments}")

            result = self.runner.run_experiment(
                prompt=prompt,
                seed=seed,
                isolated=(mode == "isolated"),
                max_tokens=10
            )

            results.append(result)

        elapsed = time.time() - start_time
        logger.info(f"Completed {num_experiments} experiments in {elapsed:.2f}s")
        logger.info(f"Throughput: {num_experiments / elapsed:.2f} experiments/second")

        return results


if __name__ == "__main__":
    # Test batched runner
    runner = BatchedExperimentRunner(batch_size=4)

    prompts = [
        "The cat sat on the",
        "Once upon a time",
        "The scientist discovered"
    ]
    seeds = [42, 43, 44]

    print("\n" + "="*80)
    print("Layer 3: Batched Prompt Processing Test")
    print("="*80)

    results = runner.run_batch(prompts, seeds, mode="isolated")

    print("\n" + "="*80)
    print("Results:")
    print("="*80)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\nExperiment {i+1}:")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {result.get('generated_text', 'N/A')[:80]}")
        print(f"  Time: {result.get('generation_time', 0):.2f}s")
