#!/usr/bin/env python3
"""
Hebbian Consolidation Experiments

Tests the hypothesis: transformers can learn during inference through
attention-gated weight updates, without backpropagation.

Experiments:
1. Pattern learning - does repeating "ABCABC" get easier over time?
2. Grammar eviction - do function words evict before content words?
3. Arithmetic with forcing - does injecting correct answers help?
4. Catastrophic forgetting - does learning B destroy knowledge of A?
"""

import torch
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hebbian import HebbianInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    window_size: int = 256
    alpha: float = 1e-5
    decay: float = 0.99
    max_tokens: int = 100


@dataclass
class ExperimentResult:
    """Result of an experiment."""
    config: ExperimentConfig
    metrics: dict
    eviction_log: list
    perplexity_curve: list
    duration_seconds: float
    timestamp: str


class HebbianExperiments:
    """Run Hebbian consolidation experiments."""

    def __init__(self, model_name: str = "deepcogito/cogito-v1-preview-llama-3B"):
        """
        Args:
            model_name: HuggingFace model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def load_model(self):
        """Load model lazily."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = 'mps' if torch.backends.mps.is_available() else \
                      'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
            attn_implementation="eager"  # Need this for attention weights
        ).to(self.device)

        logger.info(f"Model loaded on {self.device}")

    def run_pattern_learning(
        self,
        config: Optional[ExperimentConfig] = None
    ) -> ExperimentResult:
        """
        Test: Does the model learn repeating patterns?

        Feed: "ABCABCABC..." repeated many times
        Measure: Perplexity at start vs end
        Hypothesis: Perplexity should decrease as pattern consolidates
        """
        config = config or ExperimentConfig(name="pattern_learning", max_tokens=200)
        self.load_model()

        import time
        start = time.time()

        # Create engine
        engine = HebbianInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            window_size=config.window_size,
            alpha=config.alpha,
            decay=config.decay,
            device=self.device
        )

        # Generate repeating pattern
        pattern = "ABC"
        prompt = pattern * 10  # Start with some pattern

        result = engine.generate(
            prompt=prompt,
            max_new_tokens=config.max_tokens,
            temperature=0.7
        )

        duration = time.time() - start

        # Analyze perplexity trend
        perp = result.perplexity_curve
        if len(perp) >= 10:
            start_perp = sum(perp[:10]) / 10
            end_perp = sum(perp[-10:]) / 10
            perp_change = (end_perp - start_perp) / start_perp * 100
        else:
            start_perp = end_perp = perp_change = 0

        metrics = {
            "start_perplexity": start_perp,
            "end_perplexity": end_perp,
            "perplexity_change_pct": perp_change,
            "total_evictions": len(result.eviction_log),
            "generated_text": result.text[:200],
            "updater_stats": engine.updater.get_stats()
        }

        logger.info(f"Pattern learning: perplexity {start_perp:.2f} -> {end_perp:.2f} ({perp_change:+.1f}%)")

        return ExperimentResult(
            config=config,
            metrics=metrics,
            eviction_log=result.eviction_log,
            perplexity_curve=result.perplexity_curve,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

    def run_grammar_eviction(
        self,
        config: Optional[ExperimentConfig] = None
    ) -> ExperimentResult:
        """
        Test: Which token types evict first?

        Hypothesis: Function words (the, is, a) evict before content words (dog, run)
        because function words "do their job" early and stop being referenced.

        We'll use a simple POS tagger to categorize evicted tokens.
        """
        config = config or ExperimentConfig(name="grammar_eviction", max_tokens=300)
        self.load_model()

        import time
        start = time.time()

        engine = HebbianInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            window_size=config.window_size,
            alpha=config.alpha,
            decay=config.decay,
            device=self.device
        )

        # Use natural text prompt
        prompt = """The quick brown fox jumps over the lazy dog. A small bird sits on the fence.
The sun shines brightly in the blue sky. Children play happily in the park."""

        result = engine.generate(
            prompt=prompt,
            max_new_tokens=config.max_tokens,
            temperature=0.8
        )

        duration = time.time() - start

        # Analyze eviction patterns
        # Simple heuristic: categorize by token text
        function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'but'}

        eviction_categories = {
            "function_words": 0,
            "content_words": 0,
            "punctuation": 0,
            "unknown": 0
        }

        for eviction in result.eviction_log:
            pos = eviction.get('position', 0)
            if pos < len(result.tokens):
                token_id = result.tokens[pos]
                token_text = self.tokenizer.decode([token_id]).strip().lower()

                if token_text in function_words:
                    eviction_categories["function_words"] += 1
                elif token_text in '.,;:!?"\'()[]{}':
                    eviction_categories["punctuation"] += 1
                elif token_text.isalpha():
                    eviction_categories["content_words"] += 1
                else:
                    eviction_categories["unknown"] += 1

        metrics = {
            "eviction_categories": eviction_categories,
            "total_evictions": len(result.eviction_log),
            "hypothesis_supported": eviction_categories["function_words"] > eviction_categories["content_words"],
            "generated_text": result.text[:200],
            "updater_stats": engine.updater.get_stats()
        }

        logger.info(f"Grammar eviction: function={eviction_categories['function_words']}, content={eviction_categories['content_words']}")

        return ExperimentResult(
            config=config,
            metrics=metrics,
            eviction_log=result.eviction_log,
            perplexity_curve=result.perplexity_curve,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

    def run_arithmetic_forcing(
        self,
        config: Optional[ExperimentConfig] = None
    ) -> ExperimentResult:
        """
        Test: Does forcing correct answers improve future accuracy?

        1. Run arithmetic problems with forced correct answers
        2. Then run similar problems without forcing
        3. Compare accuracy

        Hypothesis: Forcing helps the model "learn" the pattern.
        """
        config = config or ExperimentConfig(name="arithmetic_forcing", max_tokens=50)
        self.load_model()

        import time
        start = time.time()

        engine = HebbianInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            window_size=config.window_size,
            alpha=config.alpha,
            decay=config.decay,
            device=self.device
        )

        # Training problems with forced answers
        training_problems = [
            ("12 + 5 = ", "17"),
            ("23 + 14 = ", "37"),
            ("8 + 9 = ", "17"),
            ("15 + 20 = ", "35"),
            ("6 + 11 = ", "17"),
        ]

        # Process training problems with forcing
        for prompt, answer in training_problems:
            # Encode the answer to get token IDs
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            forced = {i: answer_ids[i] for i in range(len(answer_ids))}

            engine.generate(
                prompt=prompt,
                max_new_tokens=len(answer_ids) + 2,
                forced_tokens=forced,
                temperature=0.0
            )

        # Test problems (no forcing)
        test_problems = [
            ("7 + 10 = ", "17"),
            ("11 + 4 = ", "15"),
            ("9 + 8 = ", "17"),
        ]

        correct = 0
        results_detail = []

        for prompt, expected in test_problems:
            result = engine.generate(
                prompt=prompt,
                max_new_tokens=5,
                temperature=0.0
            )

            generated = result.text.strip()
            is_correct = expected in generated
            if is_correct:
                correct += 1

            results_detail.append({
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "correct": is_correct
            })

        duration = time.time() - start

        accuracy = correct / len(test_problems) if test_problems else 0

        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_problems),
            "training_problems": len(training_problems),
            "results_detail": results_detail,
            "updater_stats": engine.updater.get_stats()
        }

        logger.info(f"Arithmetic forcing: accuracy={accuracy:.1%} ({correct}/{len(test_problems)})")

        return ExperimentResult(
            config=config,
            metrics=metrics,
            eviction_log=[],
            perplexity_curve=[],
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

    def run_catastrophic_forgetting(
        self,
        config: Optional[ExperimentConfig] = None
    ) -> ExperimentResult:
        """
        Test: Does learning new patterns destroy old ones?

        1. Learn pattern A
        2. Learn pattern B
        3. Test pattern A - does it still work?

        Hypothesis: Some forgetting, but not complete destruction.
        """
        config = config or ExperimentConfig(name="catastrophic_forgetting", max_tokens=100)
        self.load_model()

        import time
        start = time.time()

        engine = HebbianInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            window_size=config.window_size,
            alpha=config.alpha,
            decay=config.decay,
            device=self.device
        )

        # Phase 1: Learn pattern A
        pattern_a = "XYZ" * 20
        result_a1 = engine.generate(prompt=pattern_a, max_new_tokens=50, temperature=0.5)
        perp_a1 = sum(result_a1.perplexity_curve[-10:]) / 10 if len(result_a1.perplexity_curve) >= 10 else 0

        # Phase 2: Learn pattern B
        pattern_b = "123" * 20
        result_b = engine.generate(prompt=pattern_b, max_new_tokens=50, temperature=0.5)
        perp_b = sum(result_b.perplexity_curve[-10:]) / 10 if len(result_b.perplexity_curve) >= 10 else 0

        # Phase 3: Test pattern A again
        result_a2 = engine.generate(prompt=pattern_a, max_new_tokens=50, temperature=0.5)
        perp_a2 = sum(result_a2.perplexity_curve[-10:]) / 10 if len(result_a2.perplexity_curve) >= 10 else 0

        duration = time.time() - start

        forgetting_pct = (perp_a2 - perp_a1) / perp_a1 * 100 if perp_a1 > 0 else 0

        metrics = {
            "pattern_a_before": perp_a1,
            "pattern_b": perp_b,
            "pattern_a_after": perp_a2,
            "forgetting_pct": forgetting_pct,
            "catastrophic": forgetting_pct > 50,  # Arbitrary threshold
            "updater_stats": engine.updater.get_stats()
        }

        logger.info(f"Catastrophic forgetting: A before={perp_a1:.2f}, after={perp_a2:.2f} ({forgetting_pct:+.1f}%)")

        return ExperimentResult(
            config=config,
            metrics=metrics,
            eviction_log=result_a2.eviction_log,
            perplexity_curve=result_a2.perplexity_curve,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

    def run_baseline_comparison(
        self,
        config: Optional[ExperimentConfig] = None
    ) -> ExperimentResult:
        """
        Compare: Hebbian updates ON vs OFF

        Same generation task, measure perplexity difference.
        """
        config = config or ExperimentConfig(name="baseline_comparison", max_tokens=150)
        self.load_model()

        import time
        start = time.time()

        prompt = "Once upon a time in a land far away, there lived a wise old wizard who"

        # With Hebbian updates
        engine_on = HebbianInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            window_size=config.window_size,
            alpha=config.alpha,
            decay=config.decay,
            device=self.device
        )
        result_on = engine_on.generate(prompt=prompt, max_new_tokens=config.max_tokens, temperature=0.8)
        perp_on = sum(result_on.perplexity_curve) / len(result_on.perplexity_curve) if result_on.perplexity_curve else 0

        # Without Hebbian updates (alpha=0)
        engine_off = HebbianInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            window_size=config.window_size,
            alpha=0.0,  # No updates
            decay=config.decay,
            device=self.device
        )
        result_off = engine_off.generate(prompt=prompt, max_new_tokens=config.max_tokens, temperature=0.8)
        perp_off = sum(result_off.perplexity_curve) / len(result_off.perplexity_curve) if result_off.perplexity_curve else 0

        duration = time.time() - start

        diff_pct = (perp_on - perp_off) / perp_off * 100 if perp_off > 0 else 0

        metrics = {
            "perplexity_with_hebbian": perp_on,
            "perplexity_without_hebbian": perp_off,
            "difference_pct": diff_pct,
            "hebbian_helps": perp_on < perp_off,
            "evictions_on": len(result_on.eviction_log),
            "evictions_off": len(result_off.eviction_log),
            "text_with_hebbian": result_on.text[:200],
            "text_without_hebbian": result_off.text[:200],
        }

        logger.info(f"Baseline comparison: Hebbian={perp_on:.2f}, No Hebbian={perp_off:.2f} ({diff_pct:+.1f}%)")

        return ExperimentResult(
            config=config,
            metrics=metrics,
            eviction_log=result_on.eviction_log,
            perplexity_curve=result_on.perplexity_curve,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

    def run_all(self) -> Dict[str, ExperimentResult]:
        """Run all experiments."""
        results = {}

        experiments = [
            ("pattern_learning", self.run_pattern_learning),
            ("grammar_eviction", self.run_grammar_eviction),
            ("arithmetic_forcing", self.run_arithmetic_forcing),
            ("catastrophic_forgetting", self.run_catastrophic_forgetting),
            ("baseline_comparison", self.run_baseline_comparison),
        ]

        for name, func in experiments:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {name}")
            logger.info(f"{'='*60}")

            try:
                results[name] = func()
            except Exception as e:
                logger.error(f"Experiment {name} failed: {e}")
                import traceback
                traceback.print_exc()

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict[str, ExperimentResult]):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"hebbian_experiments_{timestamp}.json"

        # Convert to serializable format
        serializable = {}
        for name, result in results.items():
            serializable[name] = {
                "config": asdict(result.config),
                "metrics": result.metrics,
                "eviction_count": len(result.eviction_log),
                "perplexity_curve": result.perplexity_curve[:50],  # Truncate
                "duration_seconds": result.duration_seconds,
                "timestamp": result.timestamp
            }

        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")


def main():
    """Run experiments from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Hebbian Consolidation Experiments")
    parser.add_argument("--experiment", type=str, choices=[
        "all", "pattern", "grammar", "arithmetic", "forgetting", "baseline"
    ], default="all", help="Which experiment to run")
    parser.add_argument("--alpha", type=float, default=1e-5, help="Hebbian learning rate")
    parser.add_argument("--window", type=int, default=256, help="Context window size")
    parser.add_argument("--model", type=str, default="deepcogito/cogito-v1-preview-llama-3B")

    args = parser.parse_args()

    experiments = HebbianExperiments(model_name=args.model)

    config = ExperimentConfig(
        name=args.experiment,
        window_size=args.window,
        alpha=args.alpha
    )

    if args.experiment == "all":
        results = experiments.run_all()
    elif args.experiment == "pattern":
        results = {"pattern": experiments.run_pattern_learning(config)}
    elif args.experiment == "grammar":
        results = {"grammar": experiments.run_grammar_eviction(config)}
    elif args.experiment == "arithmetic":
        results = {"arithmetic": experiments.run_arithmetic_forcing(config)}
    elif args.experiment == "forgetting":
        results = {"forgetting": experiments.run_catastrophic_forgetting(config)}
    elif args.experiment == "baseline":
        results = {"baseline": experiments.run_baseline_comparison(config)}

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name}:")
        for key, value in result.metrics.items():
            if not isinstance(value, (list, dict)):
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
