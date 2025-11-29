#!/usr/bin/env python3
"""
Rigorous benchmark harness for Hebbian consolidation experiments.

Uses functional modification vectors for efficient testing:
- Load model ONCE
- Run all trials by clearing/applying modifications
- Zero-cost reset between trials
"""

import torch
import gc
import json
import logging
import sys
import psutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from datetime import datetime
import statistics
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hebbian.config import HebbianConfig, BenchmarkConfig, BASELINE, HEBBIAN

# Setup logging to both console and file
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {LOG_FILE}")


def check_memory(min_gb: float = 8.0) -> bool:
    """Check if enough memory is available. Returns False and logs warning if not."""
    available_gb = psutil.virtual_memory().available / (1024**3)
    percent_used = psutil.virtual_memory().percent

    if available_gb < min_gb:
        logger.error(f"INSUFFICIENT MEMORY: {available_gb:.1f}GB available, need {min_gb:.1f}GB")
        logger.error(f"System memory {percent_used:.0f}% used. Free up memory before running.")
        return False

    logger.info(f"Memory OK: {available_gb:.1f}GB available ({100-percent_used:.0f}% free)")
    return True


def log_memory(prefix: str = ""):
    """Log current memory state."""
    available_gb = psutil.virtual_memory().available / (1024**3)
    percent_used = psutil.virtual_memory().percent
    logger.debug(f"{prefix} Memory: {available_gb:.1f}GB available, {percent_used:.0f}% used")


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    experiment: str
    variant: str
    seed: int
    perplexity_curve: list = field(default_factory=list)
    final_perplexity: float = 0.0
    total_evictions: int = 0
    total_modifications: int = 0
    generated_text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AggregateResult:
    """Aggregated results across multiple seeds."""
    experiment: str
    variant: str
    n_seeds: int
    mean_perplexity: float
    std_perplexity: float
    ci_lower: float
    ci_upper: float
    mean_evictions: float
    mean_modifications: float
    p_value: Optional[float] = None  # vs baseline


def compute_stats(values: list) -> tuple[float, float, float, float]:
    """Compute mean, std, and 95% CI."""
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0
        return mean, 0, mean, mean

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    sem = std / (n ** 0.5)

    # t-distribution for small samples
    t_crit = stats.t.ppf(0.975, df=n-1)
    margin = t_crit * sem

    return mean, std, mean - margin, mean + margin


def compute_p_value(group1: list, group2: list) -> float:
    """Compute two-tailed t-test p-value."""
    if len(group1) < 2 or len(group2) < 2:
        return 1.0
    _, p = stats.ttest_ind(group1, group2)
    return p


class HebbianBenchmark:
    """Efficient benchmark using functional modifications."""

    def __init__(
        self,
        config: BenchmarkConfig = None,
        device: str = None,
        # Legacy params for backward compatibility
        model_name: str = None,
        n_seeds: int = None,
        output_dir: str = None,
    ):
        # Use config or build from legacy params
        if config is None:
            defaults = BenchmarkConfig()
            config = BenchmarkConfig(
                model_name=model_name or defaults.model_name,
                n_seeds=n_seeds or defaults.n_seeds,
                output_dir=output_dir or defaults.output_dir,
            )

        self.benchmark_config = config
        self.model_name = config.model_name
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_seeds = config.n_seeds
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check memory before loading model (need ~8GB for 3B model in fp16)
        logger.debug("CHECKPOINT: About to check memory before model load")
        if not check_memory(min_gb=config.min_memory_gb):
            raise MemoryError("Insufficient memory to load model. Free up memory first.")

        logger.info(f"Loading model once: {self.model_name}")
        logger.debug("CHECKPOINT: Loading tokenizer")

        # Load model and tokenizer ONCE
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        logger.debug("CHECKPOINT: Tokenizer loaded, clearing cache before model load")

        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        log_memory("Before model load")

        logger.debug("CHECKPOINT: Loading model weights")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
        )
        logger.debug("CHECKPOINT: Model loaded to CPU, moving to device")
        log_memory("After model load, before .to(device)")

        self.model = self.model.to(self.device)
        logger.debug(f"CHECKPOINT: Model moved to {self.device}")
        log_memory("After model.to(device)")

        logger.info(f"Benchmark ready: device={self.device}, seeds={n_seeds}")
        self.results: list[ExperimentResult] = []

    def _run_trial(
        self,
        engine,
        experiment: str,
        variant: str,
        seed: int,
        prompt: str,
        max_tokens: int,
    ) -> ExperimentResult:
        """Run a single trial."""
        logger.debug(f"CHECKPOINT: Starting trial {experiment}/{variant}/seed={seed}")
        log_memory(f"Trial start")

        torch.manual_seed(seed)

        # Clear any previous modifications for clean trial
        logger.debug("CHECKPOINT: Clearing modifications")
        engine.clear_modifications()
        log_memory("After clear_modifications")

        logger.debug(f"CHECKPOINT: Starting generation, max_tokens={max_tokens}")
        try:
            result = engine.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )
            logger.debug("CHECKPOINT: Generation complete")
        except Exception as e:
            logger.error(f"CHECKPOINT: Generation FAILED: {type(e).__name__}: {e}")
            log_memory("At failure")
            raise

        perp_curve = result['perplexity_curve']
        final_perp = statistics.mean(perp_curve[-10:]) if len(perp_curve) >= 10 else (
            statistics.mean(perp_curve) if perp_curve else 0
        )

        # Clear memory between trials
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        # Only store final perplexity, not full curve (memory savings)
        return ExperimentResult(
            experiment=experiment,
            variant=variant,
            seed=seed,
            perplexity_curve=[],  # Don't store full curve
            final_perplexity=final_perp,
            total_evictions=len(result['evictions']),
            total_modifications=result['total_modifications'],
            generated_text=result['text'][:100],  # Shorter
        )

    def experiment_2_update_formulas(self) -> list[AggregateResult]:
        """
        Experiment 2: Compare update formulas.

        Tests: baseline (no mods), k_only
        Uses functional engine for efficient comparison.
        """
        from src.hebbian.functional_engine import FunctionalHebbianEngine

        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: Update formula comparison")
        logger.info("=" * 60)

        prompt = "The quick brown fox jumps over"

        # Create engine once, reuse for all trials
        engine = FunctionalHebbianEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            config=HEBBIAN,
            device=self.device,
        )

        # Use config-defined variants
        variants = self.benchmark_config.variants

        all_results = {}
        aggregates = []

        for variant_name, scale in variants:
            logger.info(f"Running variant: {variant_name} (n={self.n_seeds})")
            engine.update_scale = scale

            trial_results = []
            for seed in range(self.n_seeds):
                result = self._run_trial(
                    engine=engine,
                    experiment="update_formula",
                    variant=variant_name,
                    seed=seed,
                    prompt=prompt,
                    max_tokens=100,
                )
                trial_results.append(result)
                self.results.append(result)

                if (seed + 1) % 5 == 0:
                    logger.info(f"  Completed {seed + 1}/{self.n_seeds} seeds")

            all_results[variant_name] = trial_results
            perps = [r.final_perplexity for r in trial_results]
            mean, std, ci_low, ci_high = compute_stats(perps)

            aggregates.append(AggregateResult(
                experiment="update_formula",
                variant=variant_name,
                n_seeds=self.n_seeds,
                mean_perplexity=mean,
                std_perplexity=std,
                ci_lower=ci_low,
                ci_upper=ci_high,
                mean_evictions=statistics.mean([r.total_evictions for r in trial_results]),
                mean_modifications=statistics.mean([r.total_modifications for r in trial_results]),
            ))

            logger.info(f"  Mean: {mean:.4f} ± {std:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        # Compute p-values vs baseline
        baseline_perps = [r.final_perplexity for r in all_results["baseline"]]
        for agg in aggregates:
            if agg.variant != "baseline":
                variant_perps = [r.final_perplexity for r in all_results[agg.variant]]
                agg.p_value = compute_p_value(baseline_perps, variant_perps)
                logger.info(f"  {agg.variant} vs baseline: p={agg.p_value:.4f}")

        return aggregates

    def experiment_1_compounding(self) -> list[AggregateResult]:
        """
        Experiment 1: Does improvement compound over length?
        """
        from src.hebbian.functional_engine import FunctionalHebbianEngine

        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: Compounding over length")
        logger.info("=" * 60)

        prompt = "In the beginning, there was"
        lengths = [50, 100, 200]

        engine = FunctionalHebbianEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            config=HEBBIAN,
            device=self.device,
        )

        all_results = {}
        aggregates = []

        for max_tokens in lengths:
            for variant_name, scale in self.benchmark_config.variants:
                key = f"{variant_name}_len{max_tokens}"
                logger.info(f"Running: {key}")
                engine.update_scale = scale

                trial_results = []
                for seed in range(self.n_seeds):
                    result = self._run_trial(
                        engine=engine,
                        experiment="compounding",
                        variant=key,
                        seed=seed,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                    trial_results.append(result)
                    self.results.append(result)

                all_results[key] = trial_results
                perps = [r.final_perplexity for r in trial_results]
                mean, std, ci_low, ci_high = compute_stats(perps)

                aggregates.append(AggregateResult(
                    experiment="compounding",
                    variant=key,
                    n_seeds=self.n_seeds,
                    mean_perplexity=mean,
                    std_perplexity=std,
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                    mean_evictions=statistics.mean([r.total_evictions for r in trial_results]),
                    mean_modifications=statistics.mean([r.total_modifications for r in trial_results]),
                ))

        # Compute p-values
        for length in lengths:
            baseline_key = f"baseline_len{length}"
            hebbian_key = f"hebbian_len{length}"
            if baseline_key in all_results and hebbian_key in all_results:
                p = compute_p_value(
                    [r.final_perplexity for r in all_results[baseline_key]],
                    [r.final_perplexity for r in all_results[hebbian_key]],
                )
                for agg in aggregates:
                    if agg.variant == hebbian_key:
                        agg.p_value = p
                logger.info(f"  Length {length}: p={p:.4f}")

        return aggregates

    def run_all(self) -> dict:
        """Run all experiments."""
        start_time = datetime.now()
        all_aggregates = {}

        all_aggregates["update_formula"] = self.experiment_2_update_formulas()
        all_aggregates["compounding"] = self.experiment_1_compounding()

        # Save results
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        raw_path = self.output_dir / f"raw_{timestamp}.json"
        with open(raw_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        agg_path = self.output_dir / f"agg_{timestamp}.json"
        with open(agg_path, "w") as f:
            json.dump({k: [asdict(a) for a in v] for k, v in all_aggregates.items()}, f, indent=2)

        duration = datetime.now() - start_time
        logger.info(f"Complete in {duration}. Results: {self.output_dir}")

        self._print_summary(all_aggregates)
        return all_aggregates

    def _print_summary(self, aggregates: dict):
        """Print summary with p-values."""
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        # Experiment 2
        print("\nUPDATE FORMULA COMPARISON")
        print("-" * 50)
        exp2 = aggregates.get("update_formula", [])
        baseline = next((a for a in exp2 if a.variant == "baseline"), None)

        for agg in exp2:
            diff = 0
            if baseline and baseline.mean_perplexity > 0:
                diff = ((agg.mean_perplexity - baseline.mean_perplexity) / baseline.mean_perplexity) * 100

            p_str = f"p={agg.p_value:.4f}" if agg.p_value else ""
            sig = "**" if agg.p_value and agg.p_value < 0.01 else ("*" if agg.p_value and agg.p_value < 0.05 else "")

            print(f"  {agg.variant:12s}: {agg.mean_perplexity:.4f} ± {agg.std_perplexity:.4f} "
                  f"({diff:+.2f}%) {p_str} {sig}")

        # Experiment 1
        print("\nCOMPOUNDING OVER LENGTH")
        print("-" * 50)
        exp1 = aggregates.get("compounding", [])

        for length in [50, 100, 200]:
            baseline = next((a for a in exp1 if a.variant == f"baseline_len{length}"), None)
            hebbian = next((a for a in exp1 if a.variant == f"hebbian_len{length}"), None)

            if baseline and hebbian:
                diff = ((hebbian.mean_perplexity - baseline.mean_perplexity) / baseline.mean_perplexity) * 100
                p_str = f"p={hebbian.p_value:.4f}" if hebbian.p_value else ""
                sig = "**" if hebbian.p_value and hebbian.p_value < 0.01 else ("*" if hebbian.p_value and hebbian.p_value < 0.05 else "")

                print(f"  len={length:3d}: baseline={baseline.mean_perplexity:.4f}, "
                      f"hebbian={hebbian.mean_perplexity:.4f} ({diff:+.2f}%) {p_str} {sig}")

        print("\n" + "=" * 70)
        print("* p<0.05, ** p<0.01")
        print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--experiment", choices=["all", "1", "2"], default="all")
    args = parser.parse_args()

    benchmark = HebbianBenchmark(n_seeds=args.seeds)

    if args.experiment == "all":
        benchmark.run_all()
    elif args.experiment == "2":
        results = benchmark.experiment_2_update_formulas()
        benchmark._print_summary({"update_formula": results})
    elif args.experiment == "1":
        results = benchmark.experiment_1_compounding()
        benchmark._print_summary({"compounding": results})


if __name__ == "__main__":
    main()
