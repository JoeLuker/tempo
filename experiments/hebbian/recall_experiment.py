#!/usr/bin/env python3
"""Recall-after-eviction experiment for Hebbian consolidation.

Tests whether Hebbian modifications help the model recall information
that has been evicted from the KV cache sliding window.

Hypothesis: When important tokens are evicted, Hebbian weight modifications
should retain some "memory" of that information, improving recall.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load

from src.hebbian.mlx import HebbianMLXEngine
from src.hebbian.config import HebbianConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """Result from a recall trial."""
    variant: str
    scale: float
    target: str
    response: str
    contains_target: bool
    n_evictions: int
    n_modifications: int


class RecallExperiment:
    """Test recall of information after it's been evicted from the window."""

    def __init__(self, model_name: str = "mlx-community/Qwen3-4B-4bit"):
        logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        logger.info("Model loaded")

    def _create_engine(self, config: HebbianConfig) -> HebbianMLXEngine:
        """Create engine reusing loaded model."""
        engine = HebbianMLXEngine.__new__(HebbianMLXEngine)
        engine.model_name = self.model_name
        engine.config = config
        engine.model = self.model
        engine.tokenizer = self.tokenizer

        engine.args = self.model.args
        engine.n_layers = engine.args.num_hidden_layers
        engine.n_heads = engine.args.num_attention_heads
        engine.n_kv_heads = engine.args.num_key_value_heads
        engine.hidden_dim = engine.args.hidden_size
        engine.head_dim = engine.args.head_dim
        engine.k_dim = engine.n_kv_heads * engine.head_dim

        engine._init_hebbian_state()
        return engine

    def create_recall_prompt(self, target: str, filler_tokens: int = 60) -> str:
        """Create a prompt that embeds target info, adds filler, then asks for recall.

        Uses Qwen3 ChatML format.
        """
        # Filler to push the target out of the sliding window
        filler_sentences = [
            "The weather today is quite pleasant with sunny skies.",
            "Many people enjoy walking in the park during spring.",
            "Technology continues to advance at a rapid pace.",
            "Reading books is a wonderful way to spend time.",
            "Music has the power to change our mood instantly.",
            "Gardens are beautiful places to find peace and quiet.",
        ]
        filler = " ".join(filler_sentences[:filler_tokens // 10 + 1])

        # Qwen3 ChatML format
        prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions directly and concisely.<|im_end|>
<|im_start|>user
Remember this secret code: {target}

{filler}

What was the secret code I asked you to remember? Reply with just the code.<|im_end|>
<|im_start|>assistant
The secret code is: """

        return prompt

    def run_trial(
        self,
        target: str,
        config: HebbianConfig,
        variant_name: str = "",
        filler_tokens: int = 60,
        max_response: int = 30,
    ) -> RecallResult:
        """Run a single recall trial."""
        engine = self._create_engine(config)

        prompt = self.create_recall_prompt(target, filler_tokens)

        # Count tokens in prompt to verify eviction should occur
        prompt_tokens = len(self.tokenizer.encode(prompt))
        logger.debug(f"Prompt has {prompt_tokens} tokens (window={config.window_size})")

        result = engine.generate_with_metrics(prompt, max_response, temperature=0.0)

        response = result["text"].strip()
        contains_target = target.lower() in response.lower()

        # Use provided variant name or generate one
        if not variant_name:
            variant_name = "baseline" if config.update_scale == 0 else f"{config.update_target}_{config.update_scale}"

        return RecallResult(
            variant=variant_name,
            scale=config.update_scale,
            target=target,
            response=response,
            contains_target=contains_target,
            n_evictions=result["n_evictions"],
            n_modifications=result["n_modifications"],
        )

    def run_experiment(self):
        """Run the full recall experiment."""
        logger.info("=" * 60)
        logger.info("RECALL AFTER EVICTION EXPERIMENT")
        logger.info("=" * 60)

        # Test targets - distinct codes that should be unambiguous
        targets = ["ALPHA7", "BETA99", "GAMMA42"]

        # First test: verify recall works with large window (no eviction)
        logger.info("\n=== CONTROL: Large window (no eviction) ===")
        large_window_config = HebbianConfig(update_scale=0.0, window_size=512, n_sink_tokens=4)
        for target in targets:
            result = self.run_trial(target, large_window_config)
            status = "✓" if result.contains_target else "✗"
            logger.info(f"  {target}: {status} evictions={result.n_evictions} -> {result.response[:40]}")

        # Configurations to test with small window (forces eviction)
        # Compare K-only, V-only, and both
        configs = [
            ("baseline", HebbianConfig(update_scale=0.0, window_size=32, n_sink_tokens=4)),
            # V-projection updates (stores content)
            ("V_1e-4", HebbianConfig(update_scale=1e-4, window_size=32, n_sink_tokens=4, update_target="v")),
            ("V_1e-3", HebbianConfig(update_scale=1e-3, window_size=32, n_sink_tokens=4, update_target="v")),
            ("V_1e-2", HebbianConfig(update_scale=1e-2, window_size=32, n_sink_tokens=4, update_target="v")),
            ("V_1e-1", HebbianConfig(update_scale=1e-1, window_size=32, n_sink_tokens=4, update_target="v")),
            # K-projection updates (affects matching)
            ("K_1e-3", HebbianConfig(update_scale=1e-3, window_size=32, n_sink_tokens=4, update_target="k")),
            # Both K and V
            ("KV_1e-3", HebbianConfig(update_scale=1e-3, window_size=32, n_sink_tokens=4, update_target="both")),
        ]

        results: list[RecallResult] = []

        for target in targets:
            logger.info(f"\n--- Target: {target} ---")

            for name, config in configs:
                result = self.run_trial(target, config, variant_name=name)
                results.append(result)

                status = "✓ RECALLED" if result.contains_target else "✗ FAILED"
                logger.info(f"  {name}: {status}")
                logger.info(f"    Response: {result.response[:80]}...")
                logger.info(f"    Evictions: {result.n_evictions}, Mods: {result.n_modifications}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        for name, config in configs:
            variant_results = [r for r in results if r.variant == name]
            if variant_results:
                recall_rate = sum(1 for r in variant_results if r.contains_target) / len(variant_results)
                logger.info(f"  {name}: {recall_rate:.0%} recall rate ({sum(1 for r in variant_results if r.contains_target)}/{len(variant_results)})")

        return results


def main():
    experiment = RecallExperiment()
    results = experiment.run_experiment()

    # Check if any Hebbian variant outperforms baseline
    baseline_results = [r for r in results if r.variant == "baseline"]
    baseline_rate = sum(1 for r in baseline_results if r.contains_target) / len(baseline_results) if baseline_results else 0

    # Find best variant
    variants = set(r.variant for r in results if r.variant != "baseline")
    best_rate = 0
    best_variant = None

    for variant in variants:
        variant_results = [r for r in results if r.variant == variant]
        if variant_results:
            rate = sum(1 for r in variant_results if r.contains_target) / len(variant_results)
            if rate > best_rate:
                best_rate = rate
                best_variant = variant

    logger.info(f"\nBaseline recall: {baseline_rate:.0%}")
    if best_variant:
        logger.info(f"Best Hebbian recall: {best_rate:.0%} ({best_variant})")
        if best_rate > baseline_rate:
            logger.info(">>> Hebbian consolidation IMPROVED recall! <<<")
        elif best_rate == baseline_rate:
            logger.info(">>> No difference detected <<<")
        else:
            logger.info(">>> Hebbian consolidation HURT recall <<<")


if __name__ == "__main__":
    main()
