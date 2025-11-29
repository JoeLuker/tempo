#!/usr/bin/env python3
"""
MLX benchmark harness for Hebbian consolidation experiments.

Compares baseline (no modifications) vs Hebbian consolidation on Apple Silicon.
Measures real perplexity from log probabilities.

Usage:
    python3 experiments/hebbian/benchmark_mlx.py --seeds 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
import statistics
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import mlx.core as mx
from mlx_lm import load
from scipy import stats

from src.hebbian.mlx import HebbianMLXEngine
from src.hebbian.config import HebbianConfig

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"benchmark_mlx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result from a single trial."""
    variant: str
    seed: int
    perplexity: float
    n_tokens: int
    n_evictions: int
    n_modifications: int
    tokens_per_second: float
    mean_log_prob: float
    generated_text: str = ""


@dataclass
class AggregateResult:
    """Aggregated results across seeds."""
    variant: str
    n_seeds: int
    mean_perplexity: float
    std_perplexity: float
    ci_lower: float
    ci_upper: float
    mean_tokens_per_second: float
    mean_evictions: float
    mean_modifications: float
    p_value: Optional[float] = None


def compute_stats(values: list) -> tuple[float, float, float, float]:
    """Compute mean, std, and 95% CI."""
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0
        return mean, 0, mean, mean

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    sem = std / (n ** 0.5)

    t_crit = stats.t.ppf(0.975, df=n-1)
    margin = t_crit * sem

    return mean, std, mean - margin, mean + margin


def compute_p_value(group1: list, group2: list) -> float:
    """Compute two-tailed t-test p-value."""
    if len(group1) < 2 or len(group2) < 2:
        return 1.0
    _, p = stats.ttest_ind(group1, group2)
    return p


class MLXBenchmark:
    """MLX benchmark for Hebbian consolidation with real perplexity."""

    def __init__(
        self,
        model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        n_seeds: int = 10,
        output_dir: str = "experiments/hebbian/results_mlx",
    ):
        self.model_name = model_name
        self.n_seeds = n_seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: list[TrialResult] = []

        # Load model once and reuse
        logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        logger.info("Model loaded successfully")

        logger.info(f"MLX Benchmark initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Seeds: {n_seeds}")
        logger.info(f"  Output: {output_dir}")

    def _create_engine(self, config: HebbianConfig) -> HebbianMLXEngine:
        """Create engine reusing loaded model."""
        engine = HebbianMLXEngine.__new__(HebbianMLXEngine)
        engine.model_name = self.model_name
        engine.config = config
        engine.model = self.model
        engine.tokenizer = self.tokenizer

        # Extract model architecture info
        engine.args = self.model.args
        engine.n_layers = engine.args.num_hidden_layers
        engine.n_heads = engine.args.num_attention_heads
        engine.n_kv_heads = engine.args.num_key_value_heads
        engine.hidden_dim = engine.args.hidden_size
        engine.head_dim = engine.args.head_dim
        engine.k_dim = engine.n_kv_heads * engine.head_dim

        # Initialize Hebbian state
        engine._init_hebbian_state()

        return engine

    def _run_trial(
        self,
        config: HebbianConfig,
        variant: str,
        seed: int,
        prompt: str,
        max_tokens: int,
    ) -> TrialResult:
        """Run a single trial with perplexity measurement."""
        # Set random seed
        mx.random.seed(seed)

        # Create engine (reuses loaded model)
        engine = self._create_engine(config)

        # Generate with metrics
        result = engine.generate_with_metrics(prompt, max_tokens, temperature=0.0)

        mean_log_prob = sum(result["log_probs"]) / len(result["log_probs"]) if result["log_probs"] else 0

        return TrialResult(
            variant=variant,
            seed=seed,
            perplexity=result["perplexity"],
            n_tokens=result["n_tokens"],
            n_evictions=result["n_evictions"],
            n_modifications=result["n_modifications"],
            tokens_per_second=result["tokens_per_second"],
            mean_log_prob=mean_log_prob,
            generated_text=result["text"][:200],
        )

    def run_comparison(
        self,
        prompt: str = "The quick brown fox jumps over the lazy dog. In the beginning",
        max_tokens: int = 100,
    ) -> dict[str, AggregateResult]:
        """Run baseline vs Hebbian comparison with real perplexity."""
        logger.info("=" * 60)
        logger.info("EXPERIMENT: Baseline vs Hebbian Comparison")
        logger.info("=" * 60)
        logger.info(f"Prompt: {prompt[:50]}...")
        logger.info(f"Max tokens: {max_tokens}")

        variants = [
            ("baseline", HebbianConfig(update_scale=0.0, window_size=32, n_sink_tokens=4)),
            ("hebbian", HebbianConfig(update_scale=1e-6, window_size=32, n_sink_tokens=4)),
        ]

        all_results: dict[str, list[TrialResult]] = {}

        for variant_name, config in variants:
            logger.info(f"\nRunning variant: {variant_name} (scale={config.update_scale})")
            trial_results = []

            for seed in range(self.n_seeds):
                result = self._run_trial(
                    config=config,
                    variant=variant_name,
                    seed=seed,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                trial_results.append(result)
                self.results.append(result)

                logger.info(f"  Seed {seed + 1}/{self.n_seeds}: "
                           f"ppl={result.perplexity:.2f}, "
                           f"{result.tokens_per_second:.1f} tok/s, "
                           f"{result.n_modifications} mods")

            all_results[variant_name] = trial_results

        # Compute aggregates
        aggregates = {}
        for variant_name, trials in all_results.items():
            perps = [t.perplexity for t in trials]
            speeds = [t.tokens_per_second for t in trials]
            evictions = [t.n_evictions for t in trials]
            mods = [t.n_modifications for t in trials]

            mean, std, ci_low, ci_high = compute_stats(perps)

            aggregates[variant_name] = AggregateResult(
                variant=variant_name,
                n_seeds=self.n_seeds,
                mean_perplexity=mean,
                std_perplexity=std,
                ci_lower=ci_low,
                ci_upper=ci_high,
                mean_tokens_per_second=statistics.mean(speeds),
                mean_evictions=statistics.mean(evictions),
                mean_modifications=statistics.mean(mods),
            )

        # Compute p-value for perplexity difference
        if "baseline" in all_results and "hebbian" in all_results:
            baseline_perps = [t.perplexity for t in all_results["baseline"]]
            hebbian_perps = [t.perplexity for t in all_results["hebbian"]]
            aggregates["hebbian"].p_value = compute_p_value(baseline_perps, hebbian_perps)

        return aggregates

    def run_length_scaling(
        self,
        prompt: str = "Once upon a time in a land far away, there lived a wise old wizard who",
        lengths: list[int] = None,
    ) -> dict[str, AggregateResult]:
        """Test how perplexity scales with generation length."""
        if lengths is None:
            lengths = [50, 100, 150]

        logger.info("=" * 60)
        logger.info("EXPERIMENT: Length Scaling")
        logger.info("=" * 60)

        variants = [
            ("baseline", HebbianConfig(update_scale=0.0, window_size=32, n_sink_tokens=4)),
            ("hebbian", HebbianConfig(update_scale=1e-6, window_size=32, n_sink_tokens=4)),
        ]

        all_results: dict[str, list[TrialResult]] = {}

        for length in lengths:
            for variant_name, config in variants:
                key = f"{variant_name}_len{length}"
                logger.info(f"\nRunning: {key}")

                trial_results = []
                for seed in range(self.n_seeds):
                    result = self._run_trial(
                        config=config,
                        variant=key,
                        seed=seed,
                        prompt=prompt,
                        max_tokens=length,
                    )
                    trial_results.append(result)
                    self.results.append(result)

                all_results[key] = trial_results

                # Quick summary
                perps = [t.perplexity for t in trial_results]
                speeds = [t.tokens_per_second for t in trial_results]
                logger.info(f"  Avg ppl: {statistics.mean(perps):.2f}, "
                           f"{statistics.mean(speeds):.1f} tok/s")

        # Compute aggregates
        aggregates = {}
        for key, trials in all_results.items():
            perps = [t.perplexity for t in trials]
            speeds = [t.tokens_per_second for t in trials]

            mean, std, ci_low, ci_high = compute_stats(perps)

            aggregates[key] = AggregateResult(
                variant=key,
                n_seeds=self.n_seeds,
                mean_perplexity=mean,
                std_perplexity=std,
                ci_lower=ci_low,
                ci_upper=ci_high,
                mean_tokens_per_second=statistics.mean(speeds),
                mean_evictions=statistics.mean([t.n_evictions for t in trials]),
                mean_modifications=statistics.mean([t.n_modifications for t in trials]),
            )

        # Compute p-values for each length
        for length in lengths:
            baseline_key = f"baseline_len{length}"
            hebbian_key = f"hebbian_len{length}"
            if baseline_key in all_results and hebbian_key in all_results:
                baseline_perps = [t.perplexity for t in all_results[baseline_key]]
                hebbian_perps = [t.perplexity for t in all_results[hebbian_key]]
                aggregates[hebbian_key].p_value = compute_p_value(baseline_perps, hebbian_perps)

        return aggregates

    def save_results(self, aggregates: dict):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        raw_path = self.output_dir / f"raw_{timestamp}.json"
        with open(raw_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # Save aggregates
        agg_path = self.output_dir / f"agg_{timestamp}.json"
        with open(agg_path, "w") as f:
            json.dump({k: asdict(v) for k, v in aggregates.items()}, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
        return raw_path, agg_path

    def print_summary(self, aggregates: dict):
        """Print results summary with perplexity."""
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        baseline = aggregates.get("baseline")
        hebbian = aggregates.get("hebbian")

        if baseline and hebbian:
            print("\nBASELINE vs HEBBIAN (Perplexity)")
            print("-" * 50)
            print(f"  Baseline: ppl={baseline.mean_perplexity:.3f} ± {baseline.std_perplexity:.3f}")
            print(f"            [{baseline.ci_lower:.3f}, {baseline.ci_upper:.3f}] 95% CI")
            print(f"            {baseline.mean_tokens_per_second:.1f} tok/s")
            print()
            print(f"  Hebbian:  ppl={hebbian.mean_perplexity:.3f} ± {hebbian.std_perplexity:.3f}")
            print(f"            [{hebbian.ci_lower:.3f}, {hebbian.ci_upper:.3f}] 95% CI")
            print(f"            {hebbian.mean_tokens_per_second:.1f} tok/s")
            print(f"            {hebbian.mean_modifications:.0f} modifications")

            if baseline.mean_perplexity > 0:
                ppl_diff = ((hebbian.mean_perplexity - baseline.mean_perplexity)
                           / baseline.mean_perplexity * 100)
                direction = "worse" if ppl_diff > 0 else "better"
                print(f"\n  Δ Perplexity: {ppl_diff:+.2f}% ({direction})")

            if hebbian.p_value is not None:
                sig = "**" if hebbian.p_value < 0.01 else ("*" if hebbian.p_value < 0.05 else "")
                print(f"  p-value: {hebbian.p_value:.4f} {sig}")

        # Length scaling results
        length_results = {k: v for k, v in aggregates.items() if "_len" in k}
        if length_results:
            print("\nLENGTH SCALING")
            print("-" * 50)

            lengths = sorted(set(int(k.split("_len")[1]) for k in length_results.keys()))
            for length in lengths:
                baseline_key = f"baseline_len{length}"
                hebbian_key = f"hebbian_len{length}"

                if baseline_key in aggregates and hebbian_key in aggregates:
                    b = aggregates[baseline_key]
                    h = aggregates[hebbian_key]

                    ppl_diff = ((h.mean_perplexity - b.mean_perplexity)
                               / b.mean_perplexity * 100) if b.mean_perplexity > 0 else 0

                    sig = ""
                    if h.p_value is not None:
                        sig = "**" if h.p_value < 0.01 else ("*" if h.p_value < 0.05 else "")

                    print(f"  len={length:3d}: baseline={b.mean_perplexity:.2f}, "
                          f"hebbian={h.mean_perplexity:.2f} ({ppl_diff:+.2f}%) {sig}")

        print("\n" + "=" * 70)
        print("* p<0.05, ** p<0.01 (lower perplexity = better)")
        print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MLX Hebbian Benchmark")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per variant")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    benchmark = MLXBenchmark(
        model_name=args.model,
        n_seeds=args.seeds,
    )

    # Run experiments
    all_aggregates = {}

    comparison = benchmark.run_comparison(max_tokens=args.max_tokens)
    all_aggregates.update(comparison)

    scaling = benchmark.run_length_scaling(lengths=[50, 100, 150])
    all_aggregates.update(scaling)

    # Save and print
    benchmark.save_results(all_aggregates)
    benchmark.print_summary(all_aggregates)


if __name__ == "__main__":
    main()
