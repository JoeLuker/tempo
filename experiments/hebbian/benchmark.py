#!/usr/bin/env python3
"""
Rigorous benchmark harness for Hebbian consolidation experiments.

Tests three key questions:
1. Does improvement compound over long generations?
2. What's the optimal update formula?
3. Does learning persist (in-context learning)?
"""

import torch
import gc
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from datetime import datetime
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    experiment: str
    variant: str
    seed: int
    perplexity_curve: list = field(default_factory=list)
    final_perplexity: float = 0.0
    total_evictions: int = 0
    total_updates: int = 0
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
    ci_lower: float  # 95% CI
    ci_upper: float
    mean_evictions: float
    mean_updates: float


def load_model(model_name: str, device: str):
    """Load model with memory cleanup."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
    ).to(device)


def compute_confidence_interval(values: list, confidence: float = 0.95) -> tuple:
    """Compute mean and 95% confidence interval."""
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0
        return mean, mean, mean

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    # t-value for 95% CI (approximation for small samples)
    t_value = 2.0 if n < 30 else 1.96
    margin = t_value * std / (n ** 0.5)

    return mean, mean - margin, mean + margin


class HebbianBenchmark:
    """Benchmark harness for Hebbian experiments."""

    def __init__(
        self,
        model_name: str = "deepcogito/cogito-v1-preview-llama-3B",
        device: str = None,
        n_seeds: int = 5,
        output_dir: str = "experiments/hebbian/results",
    ):
        self.model_name = model_name
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_seeds = n_seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Benchmark initialized: device={self.device}, seeds={n_seeds}")

        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.results: list[ExperimentResult] = []

    def _run_single(
        self,
        experiment: str,
        variant: str,
        seed: int,
        prompt: str,
        max_tokens: int,
        window_size: int,
        update_scale: float,
        update_fn: Optional[Callable] = None,
    ) -> ExperimentResult:
        """Run a single experiment trial."""
        from src.hebbian.minimal_engine import MinimalHebbianEngine

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Fresh model for each trial
        model = load_model(self.model_name, self.device)

        engine = MinimalHebbianEngine(
            model=model,
            tokenizer=self.tokenizer,
            window_size=window_size,
            update_scale=update_scale,
            device=self.device,
        )

        # Override update function if provided
        if update_fn is not None:
            engine._apply_hebbian = lambda *args, **kwargs: update_fn(engine, *args, **kwargs)

        result = engine.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.0,  # Deterministic for reproducibility
        )

        perp_curve = result['perplexity_curve']
        final_perp = statistics.mean(perp_curve[-10:]) if len(perp_curve) >= 10 else (
            statistics.mean(perp_curve) if perp_curve else 0
        )

        # Cleanup
        del model, engine
        gc.collect()

        return ExperimentResult(
            experiment=experiment,
            variant=variant,
            seed=seed,
            perplexity_curve=perp_curve,
            final_perplexity=final_perp,
            total_evictions=len(result['evictions']),
            total_updates=result['total_updates'],
            generated_text=result['text'][:200],
        )

    def experiment_1_compounding(self) -> list[AggregateResult]:
        """
        Experiment 1: Does improvement compound over long generations?

        Generate increasingly long sequences, measure if Hebbian advantage grows.
        """
        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: Compounding over length")
        logger.info("=" * 60)

        prompt = "In the beginning, there was"
        lengths = [50, 100, 200, 400]
        variants = [
            ("baseline", 0.0),
            ("hebbian", 0.01),
        ]

        aggregates = []

        for max_tokens in lengths:
            for variant_name, scale in variants:
                logger.info(f"Running: length={max_tokens}, variant={variant_name}")

                trial_results = []
                for seed in range(self.n_seeds):
                    result = self._run_single(
                        experiment="compounding",
                        variant=f"{variant_name}_len{max_tokens}",
                        seed=seed,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        window_size=32,
                        update_scale=scale,
                    )
                    trial_results.append(result)
                    self.results.append(result)

                # Aggregate
                perps = [r.final_perplexity for r in trial_results]
                mean, ci_low, ci_high = compute_confidence_interval(perps)

                agg = AggregateResult(
                    experiment="compounding",
                    variant=f"{variant_name}_len{max_tokens}",
                    n_seeds=self.n_seeds,
                    mean_perplexity=mean,
                    std_perplexity=statistics.stdev(perps) if len(perps) > 1 else 0,
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                    mean_evictions=statistics.mean([r.total_evictions for r in trial_results]),
                    mean_updates=statistics.mean([r.total_updates for r in trial_results]),
                )
                aggregates.append(agg)

                logger.info(f"  Perplexity: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

        return aggregates

    def experiment_2_update_formulas(self) -> list[AggregateResult]:
        """
        Experiment 2: What's the optimal update formula?

        Test different Hebbian update variants.
        """
        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: Update formula variants")
        logger.info("=" * 60)

        prompt = "The quick brown fox jumps over"

        # Define update formula variants
        def standard_update(engine, layer_idx, key, value, input_hidden, importance):
            """Standard: outer(output, input), normalized."""
            layer = engine.layers[layer_idx]
            with torch.no_grad():
                for proj, output in [('k_proj', key), ('v_proj', value)]:
                    W = layer[proj].weight
                    out = output.to(W.device, W.dtype)
                    inp = input_hidden.to(W.device, W.dtype)
                    if out.size(0) != W.size(0):
                        out = out[:W.size(0)]
                    if inp.size(0) != W.size(1):
                        inp = inp[:W.size(1)]
                    update = torch.outer(out, inp)
                    u_norm = update.norm()
                    if u_norm > 0:
                        update = update / u_norm
                    W.add_(update, alpha=engine.update_scale * importance)
            engine.total_updates += 1

        def k_only_update(engine, layer_idx, key, value, input_hidden, importance):
            """Only update K projection, not V."""
            layer = engine.layers[layer_idx]
            with torch.no_grad():
                W = layer['k_proj'].weight
                out = key.to(W.device, W.dtype)
                inp = input_hidden.to(W.device, W.dtype)
                if out.size(0) != W.size(0):
                    out = out[:W.size(0)]
                if inp.size(0) != W.size(1):
                    inp = inp[:W.size(1)]
                update = torch.outer(out, inp)
                u_norm = update.norm()
                if u_norm > 0:
                    update = update / u_norm
                W.add_(update, alpha=engine.update_scale * importance)
            engine.total_updates += 1

        def v_only_update(engine, layer_idx, key, value, input_hidden, importance):
            """Only update V projection, not K."""
            layer = engine.layers[layer_idx]
            with torch.no_grad():
                W = layer['v_proj'].weight
                out = value.to(W.device, W.dtype)
                inp = input_hidden.to(W.device, W.dtype)
                if out.size(0) != W.size(0):
                    out = out[:W.size(0)]
                if inp.size(0) != W.size(1):
                    inp = inp[:W.size(1)]
                update = torch.outer(out, inp)
                u_norm = update.norm()
                if u_norm > 0:
                    update = update / u_norm
                W.add_(update, alpha=engine.update_scale * importance)
            engine.total_updates += 1

        def importance_squared_update(engine, layer_idx, key, value, input_hidden, importance):
            """Weight by importance^2 for stronger effect on high-attention tokens."""
            layer = engine.layers[layer_idx]
            with torch.no_grad():
                for proj, output in [('k_proj', key), ('v_proj', value)]:
                    W = layer[proj].weight
                    out = output.to(W.device, W.dtype)
                    inp = input_hidden.to(W.device, W.dtype)
                    if out.size(0) != W.size(0):
                        out = out[:W.size(0)]
                    if inp.size(0) != W.size(1):
                        inp = inp[:W.size(1)]
                    update = torch.outer(out, inp)
                    u_norm = update.norm()
                    if u_norm > 0:
                        update = update / u_norm
                    # Square the importance for stronger weighting
                    W.add_(update, alpha=engine.update_scale * (importance ** 2))
            engine.total_updates += 1

        variants = [
            ("baseline", 0.0, None),
            ("standard", 0.01, standard_update),
            ("k_only", 0.01, k_only_update),
            ("v_only", 0.01, v_only_update),
            ("importance_sq", 0.01, importance_squared_update),
        ]

        aggregates = []

        for variant_name, scale, update_fn in variants:
            logger.info(f"Running variant: {variant_name}")

            trial_results = []
            for seed in range(self.n_seeds):
                result = self._run_single(
                    experiment="update_formula",
                    variant=variant_name,
                    seed=seed,
                    prompt=prompt,
                    max_tokens=100,
                    window_size=32,
                    update_scale=scale,
                    update_fn=update_fn,
                )
                trial_results.append(result)
                self.results.append(result)

            perps = [r.final_perplexity for r in trial_results]
            mean, ci_low, ci_high = compute_confidence_interval(perps)

            agg = AggregateResult(
                experiment="update_formula",
                variant=variant_name,
                n_seeds=self.n_seeds,
                mean_perplexity=mean,
                std_perplexity=statistics.stdev(perps) if len(perps) > 1 else 0,
                ci_lower=ci_low,
                ci_upper=ci_high,
                mean_evictions=statistics.mean([r.total_evictions for r in trial_results]),
                mean_updates=statistics.mean([r.total_updates for r in trial_results]),
            )
            aggregates.append(agg)

            logger.info(f"  Perplexity: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

        return aggregates

    def experiment_3_persistent_learning(self) -> list[AggregateResult]:
        """
        Experiment 3: Does learning persist?

        Train on a pattern, evict it, then test recall.
        """
        logger.info("=" * 60)
        logger.info("EXPERIMENT 3: Persistent learning")
        logger.info("=" * 60)

        from src.hebbian.minimal_engine import MinimalHebbianEngine

        # Pattern to learn
        pattern = "ALPHA BETA GAMMA " * 10  # Repeat pattern
        test_prompt = "The sequence continues: ALPHA"  # Should trigger recall

        variants = [
            ("baseline", 0.0),
            ("hebbian_low", 0.01),
            ("hebbian_high", 0.1),
        ]

        aggregates = []

        for variant_name, scale in variants:
            logger.info(f"Running variant: {variant_name}")

            trial_results = []
            for seed in range(self.n_seeds):
                torch.manual_seed(seed)

                # Fresh model
                model = load_model(self.model_name, self.device)

                engine = MinimalHebbianEngine(
                    model=model,
                    tokenizer=self.tokenizer,
                    window_size=24,  # Small window to force eviction
                    update_scale=scale,
                    device=self.device,
                )

                # Phase 1: Learn the pattern (generates and evicts)
                learn_result = engine.generate(
                    prompt=pattern,
                    max_new_tokens=30,
                    temperature=0.0,
                )

                # Phase 2: Test recall with new prompt
                # Reset context but keep weight updates
                engine.slots = {}
                engine.kv_cache = {i: {} for i in range(engine.num_layers)}
                engine.input_cache = {}
                engine.next_pos = 0
                engine.protected = set()

                test_result = engine.generate(
                    prompt=test_prompt,
                    max_new_tokens=20,
                    temperature=0.0,
                )

                # Check if it continues the pattern
                generated = test_result['text']
                # Count pattern matches (BETA, GAMMA after ALPHA)
                pattern_score = 0
                if "BETA" in generated:
                    pattern_score += 1
                if "GAMMA" in generated:
                    pattern_score += 1
                if "ALPHA" in generated:
                    pattern_score += 1

                result = ExperimentResult(
                    experiment="persistent_learning",
                    variant=variant_name,
                    seed=seed,
                    perplexity_curve=test_result['perplexity_curve'],
                    final_perplexity=statistics.mean(test_result['perplexity_curve']) if test_result['perplexity_curve'] else 0,
                    total_evictions=len(learn_result['evictions']),
                    total_updates=learn_result['total_updates'],
                    generated_text=generated,
                    metadata={"pattern_score": pattern_score},
                )
                trial_results.append(result)
                self.results.append(result)

                del model, engine
                gc.collect()

            perps = [r.final_perplexity for r in trial_results]
            scores = [r.metadata.get("pattern_score", 0) for r in trial_results]
            mean, ci_low, ci_high = compute_confidence_interval(perps)

            agg = AggregateResult(
                experiment="persistent_learning",
                variant=variant_name,
                n_seeds=self.n_seeds,
                mean_perplexity=mean,
                std_perplexity=statistics.stdev(perps) if len(perps) > 1 else 0,
                ci_lower=ci_low,
                ci_upper=ci_high,
                mean_evictions=statistics.mean([r.total_evictions for r in trial_results]),
                mean_updates=statistics.mean([r.total_updates for r in trial_results]),
            )
            aggregates.append(agg)

            mean_score = statistics.mean(scores)
            logger.info(f"  Perplexity: {mean:.3f}, Pattern score: {mean_score:.2f}/3")

        return aggregates

    def run_all(self) -> dict:
        """Run all experiments and save results."""
        start_time = datetime.now()

        all_aggregates = {}

        # Run experiments
        all_aggregates["compounding"] = self.experiment_1_compounding()
        all_aggregates["update_formula"] = self.experiment_2_update_formulas()
        all_aggregates["persistent_learning"] = self.experiment_3_persistent_learning()

        # Save results
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        # Save raw results
        raw_path = self.output_dir / f"raw_results_{timestamp}.json"
        with open(raw_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # Save aggregates
        agg_path = self.output_dir / f"aggregates_{timestamp}.json"
        agg_data = {
            exp: [asdict(a) for a in aggs]
            for exp, aggs in all_aggregates.items()
        }
        with open(agg_path, "w") as f:
            json.dump(agg_data, f, indent=2)

        # Print summary
        duration = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"Duration: {duration}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)

        self._print_summary(all_aggregates)

        return all_aggregates

    def _print_summary(self, aggregates: dict):
        """Print a human-readable summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Experiment 1: Compounding
        print("\n1. COMPOUNDING OVER LENGTH")
        print("-" * 40)
        exp1 = aggregates.get("compounding", [])
        for length in [50, 100, 200, 400]:
            baseline = next((a for a in exp1 if a.variant == f"baseline_len{length}"), None)
            hebbian = next((a for a in exp1 if a.variant == f"hebbian_len{length}"), None)
            if baseline and hebbian:
                diff = ((hebbian.mean_perplexity - baseline.mean_perplexity) / baseline.mean_perplexity) * 100
                print(f"  Length {length:3d}: baseline={baseline.mean_perplexity:.3f}, "
                      f"hebbian={hebbian.mean_perplexity:.3f} ({diff:+.2f}%)")

        # Experiment 2: Update formulas
        print("\n2. UPDATE FORMULA COMPARISON")
        print("-" * 40)
        exp2 = aggregates.get("update_formula", [])
        baseline = next((a for a in exp2 if a.variant == "baseline"), None)
        for agg in exp2:
            if baseline:
                diff = ((agg.mean_perplexity - baseline.mean_perplexity) / baseline.mean_perplexity) * 100
                sig = "*" if abs(agg.mean_perplexity - baseline.mean_perplexity) > agg.std_perplexity else ""
                print(f"  {agg.variant:15s}: {agg.mean_perplexity:.3f} ± {agg.std_perplexity:.3f} "
                      f"({diff:+.2f}%) {sig}")

        # Experiment 3: Persistent learning
        print("\n3. PERSISTENT LEARNING")
        print("-" * 40)
        exp3 = aggregates.get("persistent_learning", [])
        for agg in exp3:
            print(f"  {agg.variant:15s}: perplexity={agg.mean_perplexity:.3f}, "
                  f"evictions={agg.mean_evictions:.0f}")

        print("\n" + "=" * 70)


def main():
    """Run the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Hebbian consolidation benchmark")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--experiment", type=str, choices=["all", "1", "2", "3"],
                        default="all", help="Which experiment to run")
    args = parser.parse_args()

    benchmark = HebbianBenchmark(n_seeds=args.seeds)

    if args.experiment == "all":
        benchmark.run_all()
    elif args.experiment == "1":
        results = benchmark.experiment_1_compounding()
        benchmark._print_summary({"compounding": results})
    elif args.experiment == "2":
        results = benchmark.experiment_2_update_formulas()
        benchmark._print_summary({"update_formula": results})
    elif args.experiment == "3":
        results = benchmark.experiment_3_persistent_learning()
        benchmark._print_summary({"persistent_learning": results})


if __name__ == "__main__":
    main()
