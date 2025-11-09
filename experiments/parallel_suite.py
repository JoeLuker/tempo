#!/usr/bin/env python3
"""Layer 2: Multi-process batching - run experiments in parallel across cores.

This layer builds on Layer 1 (model persistence) by running multiple experiments
simultaneously using multiprocessing. Each worker process has its own persistent
model instance.

Architecture:
- Main process: Coordinates work distribution
- Worker processes: Each loads model once, runs assigned experiments
- Communication: Queues for work distribution and result collection

Expected speedup: ~6.7x on 8-core system (7 workers + 1 coordinator)
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import time
from experiments.simple_persistent import SimplePersistentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def worker_process(worker_id: int, work_queue: Queue, result_queue: Queue):
    """Worker process that runs experiments with a persistent model.

    Args:
        worker_id: Unique identifier for this worker
        work_queue: Queue to receive work items (prompt, seed, mode)
        result_queue: Queue to send back results
    """
    logger.info(f"Worker {worker_id} starting...")

    # Each worker creates its own persistent runner
    runner = SimplePersistentRunner()
    logger.info(f"Worker {worker_id} model loaded and ready")

    experiments_completed = 0

    while True:
        # Get work from queue
        work_item = work_queue.get()

        # Check for termination signal
        if work_item is None:
            logger.info(f"Worker {worker_id} received termination signal, shutting down...")
            break

        # Unpack work item
        experiment_id, prompt, seed, mode = work_item

        logger.info(f"Worker {worker_id} processing experiment {experiment_id}")

        try:
            # Run experiment with persistent model
            result = runner.run_experiment(
                prompt=prompt,
                seed=seed,
                isolated=(mode == "isolated"),
                max_tokens=10
            )

            # Send result back
            result_queue.put((experiment_id, "success", result))
            experiments_completed += 1

        except Exception as e:
            logger.error(f"Worker {worker_id} error on experiment {experiment_id}: {e}")
            result_queue.put((experiment_id, "error", str(e)))

    logger.info(f"Worker {worker_id} completed {experiments_completed} experiments, exiting")


class ParallelExperimentSuite:
    """Layer 2: Multi-process parallel experiment runner.

    Runs experiments in parallel across multiple worker processes, where each
    worker maintains a persistent model instance (Layer 1 optimization).
    """

    def __init__(self, num_workers: int = None):
        """Initialize parallel suite.

        Args:
            num_workers: Number of worker processes. Defaults to CPU count - 1
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        self.num_workers = num_workers
        self.work_queue = None
        self.result_queue = None
        self.workers = []

        logger.info(f"ParallelExperimentSuite initialized with {num_workers} workers")

    def run_batch(
        self,
        prompts: List[str],
        seeds: List[int],
        modes: List[str]
    ) -> List[Dict]:
        """Run a batch of experiments in parallel.

        Args:
            prompts: List of prompts to generate from
            seeds: List of seeds (one per prompt)
            modes: List of modes ("isolated" or "visible")

        Returns:
            List of result dicts, ordered by input order
        """
        assert len(prompts) == len(seeds) == len(modes), \
            f"Length mismatch: {len(prompts)} prompts, {len(seeds)} seeds, {len(modes)} modes"

        num_experiments = len(prompts)
        logger.info(f"Running {num_experiments} experiments across {self.num_workers} workers")

        # Create queues
        self.work_queue = Queue()
        self.result_queue = Queue()

        # Populate work queue
        for i, (prompt, seed, mode) in enumerate(zip(prompts, seeds, modes)):
            self.work_queue.put((i, prompt, seed, mode))

        # Add termination signals (one per worker)
        for _ in range(self.num_workers):
            self.work_queue.put(None)

        # Start worker processes
        logger.info("Starting worker processes...")
        start_time = time.time()

        for worker_id in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(worker_id, self.work_queue, self.result_queue)
            )
            p.start()
            self.workers.append(p)

        # Collect results
        logger.info("Collecting results...")
        results_dict = {}
        errors = []

        for _ in range(num_experiments):
            experiment_id, status, result = self.result_queue.get()

            if status == "success":
                results_dict[experiment_id] = result
            else:
                errors.append((experiment_id, result))
                logger.error(f"Experiment {experiment_id} failed: {result}")

        # Wait for all workers to finish
        logger.info("Waiting for workers to terminate...")
        for p in self.workers:
            p.join()

        elapsed = time.time() - start_time
        logger.info(f"All {num_experiments} experiments completed in {elapsed:.2f}s")
        logger.info(f"Throughput: {num_experiments / elapsed:.2f} experiments/second")

        # Clean up
        self.work_queue = None
        self.result_queue = None
        self.workers = []

        # Check for errors
        if errors:
            logger.error(f"{len(errors)} experiments failed:")
            for exp_id, error in errors:
                logger.error(f"  Experiment {exp_id}: {error}")
            raise RuntimeError(f"{len(errors)} experiments failed")

        # Return results in order
        return [results_dict[i] for i in range(num_experiments)]


if __name__ == "__main__":
    # Test with a few experiments
    suite = ParallelExperimentSuite(num_workers=2)

    prompts = [
        "The cat sat on the",
        "Once upon a time",
        "The scientist discovered"
    ]
    seeds = [42, 43, 44]
    modes = ["isolated", "isolated", "isolated"]

    print("\n" + "="*80)
    print("Layer 2: Multi-Process Parallel Execution Test")
    print("="*80)

    results = suite.run_batch(prompts, seeds, modes)

    print("\n" + "="*80)
    print("Results:")
    print("="*80)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\nExperiment {i+1}:")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {result.get('generated_text', 'N/A')[:80]}")
        print(f"  Time: {result.get('generation_time', 0):.2f}s")
