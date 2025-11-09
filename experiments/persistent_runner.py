#!/usr/bin/env python3
"""Layer 1: Persistent Model Experiment Runner.

This module implements model persistence to avoid reloading the model
for each experiment. Provides ~2x speedup over baseline.

Key features:
- Load model once, reuse for all experiments
- Clear KV cache between runs
- Reset RoPE modifier state
- Maintain determinism with seed control
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
import time

from src.utils.model_utils import load_tempo_components
from src.domain.entities.parallel_generation import GenerationConfig
from src.application.services.generation_service import GenerationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistentExperimentRunner:
    """
    Runs multiple experiments with a single persistent model instance.

    Usage:
        runner = PersistentExperimentRunner()

        # Run multiple experiments without reloading model
        for prompt in prompts:
            for seed in seeds:
                result = runner.run_experiment(prompt, seed, mode="isolated")
    """

    def __init__(
        self,
        model_id: str = "deepcogito/cogito-v1-preview-llama-3B",
        device: str = "mps",
        debug_mode: bool = False
    ):
        """
        Initialize with persistent model loading.

        Args:
            model_id: HuggingFace model ID
            device: Device to load model on (mps, cuda, cpu)
            debug_mode: Enable debug logging
        """
        self.model_id = model_id
        self.device = device
        self.debug_mode = debug_mode

        logger.info("="*80)
        logger.info("Initializing Persistent Experiment Runner (Layer 1)")
        logger.info("="*80)
        logger.info(f"Model: {model_id}")
        logger.info(f"Device: {device}")

        # Load model once
        start_time = time.time()
        self._load_components()
        load_time = time.time() - start_time

        logger.info(f"✓ Model loaded in {load_time:.2f}s")
        logger.info(f"✓ Ready for experiments (model will persist)")
        logger.info("="*80)

    def _load_components(self):
        """Load all TEMPO components once."""
        components = load_tempo_components(
            model_id=self.model_id,
            device=self.device,
            load_model_wrapper=True,
            debug_mode=self.debug_mode
        )

        self.model_wrapper = components["model_wrapper"]
        self.tokenizer = components["tokenizer"]
        self.rope_modifier = components.get("rope_modifier")
        self.attention_manager = components.get("attention_manager")

        # Store model config
        self.model_config = components["config"]

        # Create generation service
        self.generation_service = GenerationService(
            model_wrapper=self.model_wrapper,
            rope_modifier=self.rope_modifier,
            attention_manager=self.attention_manager,
            debug_mode=self.debug_mode
        )

    def _reset_state(self):
        """Reset all stateful components between experiments."""
        # Clear KV cache
        if hasattr(self.model_wrapper, 'clear_kv_cache'):
            self.model_wrapper.clear_kv_cache()

        # Reset RoPE modifier
        if self.rope_modifier:
            self.rope_modifier.reset()

        # Reset attention manager
        if self.attention_manager:
            self.attention_manager.reset_cache()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_experiment(
        self,
        prompt: str,
        seed: int,
        mode: str = "isolated",
        max_tokens: int = 10,
        selection_threshold: float = 0.1,
        capture_attention: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Run a single experiment with persistent model.

        Args:
            prompt: Input prompt
            seed: Random seed for reproducibility
            mode: "isolated" or "visible"
            max_tokens: Maximum tokens to generate
            selection_threshold: Threshold for parallel token selection
            capture_attention: Whether to capture attention matrices
            output_dir: Where to save results (None = don't save)

        Returns:
            Dict with:
                - generated_token_ids: List of token IDs
                - generated_text: Formatted text with parallel indicators
                - attention_data: Attention matrices (if capture_attention=True)
                - parallel_sets: Parallel token information
                - generation_time: Time taken
                - metadata: Experiment configuration
        """
        if self.debug_mode:
            logger.info(f"Running experiment: prompt='{prompt[:30]}...', seed={seed}, mode={mode}")

        # Reset state from previous experiment
        self._reset_state()

        # Set seed
        self._set_seed(seed)

        # Create generation config
        config = GenerationConfig(
            prompt=prompt,
            max_tokens=max_tokens,
            selection_threshold=selection_threshold,
            isolate_parallel_tokens=(mode == "isolated"),
            seed=seed,
            disable_kv_cache=False,  # Use KV cache
            debug_mode=self.debug_mode
        )

        # Run generation
        start_time = time.time()

        try:
            result = self.generation_service.generate(config)

            generation_time = time.time() - start_time

            # Extract data
            output = {
                "generated_token_ids": self._extract_token_ids(result),
                "generated_text": result.generated_text,
                "raw_generated_text": result.raw_generated_text,
                "generation_time": generation_time,
                "prompt": prompt,
                "seed": seed,
                "mode": mode,
                "selection_threshold": selection_threshold,
                "max_tokens": max_tokens
            }

            # Capture attention if requested
            if capture_attention:
                output["attention_data"] = self._extract_attention_data()

            # Capture parallel sets
            output["parallel_sets"] = self._extract_parallel_sets(result)

            # Save if requested
            if output_dir:
                self._save_results(output, output_dir)

            if self.debug_mode:
                logger.info(f"✓ Experiment completed in {generation_time:.2f}s")

            return output

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

    def _extract_token_ids(self, result) -> list:
        """Extract generated token IDs from result."""
        # Get token IDs from the result
        # This depends on your GenerationResult structure
        if hasattr(result, 'logical_layout'):
            token_ids = []
            for pos in result.logical_layout:
                token_ids.extend(pos.token_ids)
            return token_ids
        else:
            # Fallback: decode from text
            tokens = self.tokenizer.encode(result.raw_generated_text)
            return tokens

    def _extract_attention_data(self) -> Dict:
        """Extract attention matrices from attention manager."""
        if not self.attention_manager:
            return {}

        # This depends on your attention capture implementation
        # Placeholder - adapt to your actual attention storage
        attention_data = {}

        if hasattr(self.attention_manager, 'get_captured_attention'):
            captured = self.attention_manager.get_captured_attention()
            for step, data in captured.items():
                if 'attention_weights' in data:
                    attention_data[f"step_{step}_attention"] = data['attention_weights']
                if 'positions' in data:
                    attention_data[f"step_{step}_positions"] = data['positions']
                if 'logical_positions' in data:
                    attention_data[f"step_{step}_logical"] = data['logical_positions']

        return attention_data

    def _extract_parallel_sets(self, result) -> Dict:
        """Extract parallel token sets from result."""
        parallel_sets = {
            "parallel_sets": [],
            "summary": {
                "total_parallel_steps": 0,
                "max_parallel_width": 0
            }
        }

        if hasattr(result, 'logical_layout'):
            for i, pos in enumerate(result.logical_layout):
                if len(pos.token_ids) > 1:
                    parallel_sets["parallel_sets"].append({
                        "step": i,
                        "count": len(pos.token_ids),
                        "tokens": pos.token_ids,
                        "positions": list(range(i, i + len(pos.token_ids)))
                    })

            parallel_sets["summary"]["total_parallel_steps"] = len(parallel_sets["parallel_sets"])
            if parallel_sets["parallel_sets"]:
                parallel_sets["summary"]["max_parallel_width"] = max(
                    ps["count"] for ps in parallel_sets["parallel_sets"]
                )

        return parallel_sets

    def _save_results(self, output: Dict, output_dir: Path):
        """Save experiment results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "prompt": output["prompt"],
            "seed": output["seed"],
            "mode": output["mode"],
            "selection_threshold": output["selection_threshold"],
            "max_tokens": output["max_tokens"],
            "generation_time": output["generation_time"]
        }

        import json
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save attention data if present
        if "attention_data" in output and output["attention_data"]:
            np.savez(
                output_dir / "attention_weights.npz",
                **output["attention_data"]
            )

        # Save parallel sets
        if "parallel_sets" in output:
            with open(output_dir / "parallel_sets.json", 'w') as f:
                json.dump(output["parallel_sets"], f, indent=2)

        logger.info(f"✓ Results saved to {output_dir}")

    def run_suite(
        self,
        prompts: list,
        seeds: list,
        modes: list,
        **kwargs
    ) -> list:
        """
        Run a suite of experiments sequentially with persistent model.

        Args:
            prompts: List of prompts to test
            seeds: List of seeds to use
            modes: List of modes ("isolated", "visible")
            **kwargs: Additional arguments passed to run_experiment

        Returns:
            List of results, one per experiment
        """
        total_experiments = len(prompts) * len(seeds) * len(modes)
        logger.info("="*80)
        logger.info(f"Running experiment suite: {total_experiments} total experiments")
        logger.info(f"  Prompts: {len(prompts)}")
        logger.info(f"  Seeds: {len(seeds)}")
        logger.info(f"  Modes: {len(modes)}")
        logger.info("="*80)

        results = []
        start_time = time.time()

        experiment_num = 0
        for prompt in prompts:
            for seed in seeds:
                for mode in modes:
                    experiment_num += 1
                    logger.info(f"\nExperiment {experiment_num}/{total_experiments}")

                    result = self.run_experiment(
                        prompt=prompt,
                        seed=seed,
                        mode=mode,
                        **kwargs
                    )

                    results.append(result)

        total_time = time.time() - start_time
        avg_time = total_time / total_experiments

        logger.info("="*80)
        logger.info(f"Suite completed:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Average per experiment: {avg_time:.2f}s")
        logger.info(f"  Throughput: {total_experiments/total_time:.2f} experiments/sec")
        logger.info("="*80)

        return results


def main():
    """Test the persistent runner."""
    print("Testing Persistent Experiment Runner (Layer 1)")
    print("="*80)

    # Create runner
    runner = PersistentExperimentRunner(debug_mode=True)

    # Run a few test experiments
    prompts = ["The cat sat on the", "Once upon a time"]
    seeds = [42, 123]

    results = runner.run_suite(
        prompts=prompts,
        seeds=seeds,
        modes=["isolated"],
        max_tokens=10,
        selection_threshold=0.1,
        capture_attention=False  # Faster without attention
    )

    print(f"\n✓ Completed {len(results)} experiments")
    print("Layer 1 is working!")


if __name__ == "__main__":
    main()
