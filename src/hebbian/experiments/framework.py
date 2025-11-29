"""Experimental framework for rigorous hypothesis testing.

Provides structure for:
- Pre-registered hypotheses
- Reproducible experiment configs
- Proper statistical analysis
- Result persistence and reporting
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .metrics import (
    compute_effect_size,
    compute_confidence_interval,
    power_analysis,
    test_difference,
    test_proportion,
    HypothesisTest,
    EffectSize,
    ConfidenceInterval,
    PowerAnalysis,
)

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A testable hypothesis with statistical requirements."""
    name: str
    null_hypothesis: str
    alternative_hypothesis: str
    metric: str  # What we're measuring
    expected_effect_size: float  # Cohen's d we expect
    alpha: float = 0.05  # Significance level
    power: float = 0.80  # Desired power

    def required_samples(self) -> int:
        """Compute required sample size for this hypothesis."""
        result = power_analysis(
            effect_size=self.expected_effect_size,
            alpha=self.alpha,
            power=self.power,
        )
        return result.required_n


@dataclass
class ExperimentResult:
    """Result from a single experimental trial."""
    trial_id: int
    condition: str  # e.g., "baseline" or "hebbian_v_1e-3"
    seed: int
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConditionResult:
    """Aggregated results for a single condition."""
    condition: str
    n_trials: int
    metrics: dict[str, ConfidenceInterval]
    raw_values: dict[str, list[float]]


@dataclass
class ComparisonResult:
    """Statistical comparison between two conditions."""
    baseline: str
    treatment: str
    metric: str
    hypothesis_test: HypothesisTest
    effect_size: EffectSize
    baseline_ci: ConfidenceInterval
    treatment_ci: ConfidenceInterval
    conclusion: str


class Experiment(ABC):
    """Base class for rigorous experiments.

    Subclasses must implement:
    - hypotheses: List of hypotheses being tested
    - run_trial: Execute a single trial and return metrics
    """

    def __init__(
        self,
        name: str,
        output_dir: str = "experiments/hebbian/results",
        n_seeds: int = 30,  # Default to reasonable statistical power
    ):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_seeds = n_seeds
        self.results: list[ExperimentResult] = []

    @property
    @abstractmethod
    def hypotheses(self) -> list[Hypothesis]:
        """Define the hypotheses being tested."""
        pass

    @abstractmethod
    def conditions(self) -> list[str]:
        """List of experimental conditions to test."""
        pass

    @abstractmethod
    def run_trial(self, condition: str, seed: int) -> dict[str, float]:
        """Execute a single trial and return metrics."""
        pass

    def check_sample_size(self) -> dict[str, dict]:
        """Verify sample size is adequate for each hypothesis."""
        results = {}
        for h in self.hypotheses:
            required = h.required_samples()
            adequate = self.n_seeds >= required
            results[h.name] = {
                "required_n": required,
                "actual_n": self.n_seeds,
                "adequate": adequate,
                "expected_effect_size": h.expected_effect_size,
            }
            if not adequate:
                logger.warning(
                    f"Hypothesis '{h.name}' needs {required} samples but only {self.n_seeds} provided. "
                    f"Power will be reduced."
                )
        return results

    def run(self, conditions: list[str] | None = None) -> None:
        """Run all trials for specified conditions."""
        if conditions is None:
            conditions = self.conditions()

        # Pre-flight check
        sample_check = self.check_sample_size()
        logger.info(f"Sample size check: {json.dumps(sample_check, indent=2)}")

        for condition in conditions:
            logger.info(f"Running condition: {condition}")
            for seed in range(self.n_seeds):
                try:
                    metrics = self.run_trial(condition, seed)
                    result = ExperimentResult(
                        trial_id=len(self.results),
                        condition=condition,
                        seed=seed,
                        metrics=metrics,
                    )
                    self.results.append(result)

                    if (seed + 1) % 10 == 0:
                        logger.info(f"  Completed {seed + 1}/{self.n_seeds} seeds")

                except Exception as e:
                    logger.error(f"Trial failed: condition={condition}, seed={seed}, error={e}")
                    raise

    def aggregate_condition(self, condition: str) -> ConditionResult:
        """Aggregate results for a single condition."""
        trials = [r for r in self.results if r.condition == condition]
        if not trials:
            raise ValueError(f"No trials found for condition: {condition}")

        # Collect all metrics
        all_metrics = set()
        for trial in trials:
            all_metrics.update(trial.metrics.keys())

        raw_values = {metric: [] for metric in all_metrics}
        for trial in trials:
            for metric in all_metrics:
                if metric in trial.metrics:
                    raw_values[metric].append(trial.metrics[metric])

        metrics = {
            metric: compute_confidence_interval(values)
            for metric, values in raw_values.items()
            if values
        }

        return ConditionResult(
            condition=condition,
            n_trials=len(trials),
            metrics=metrics,
            raw_values=raw_values,
        )

    def compare_conditions(
        self,
        baseline: str,
        treatment: str,
        metric: str,
    ) -> ComparisonResult:
        """Statistically compare two conditions."""
        baseline_result = self.aggregate_condition(baseline)
        treatment_result = self.aggregate_condition(treatment)

        baseline_values = baseline_result.raw_values.get(metric, [])
        treatment_values = treatment_result.raw_values.get(metric, [])

        hypothesis_test = test_difference(baseline_values, treatment_values)
        effect = compute_effect_size(treatment_values, baseline_values)

        # Generate conclusion
        if hypothesis_test.reject_null:
            direction = "higher" if effect.d > 0 else "lower"
            conclusion = (
                f"Significant difference detected (p={hypothesis_test.p_value:.4f}). "
                f"Treatment shows {direction} {metric} with {effect.interpretation} effect size (d={effect.d:.3f})."
            )
        else:
            conclusion = (
                f"No significant difference detected (p={hypothesis_test.p_value:.4f}). "
                f"Effect size: {effect.interpretation} (d={effect.d:.3f})."
            )

        return ComparisonResult(
            baseline=baseline,
            treatment=treatment,
            metric=metric,
            hypothesis_test=hypothesis_test,
            effect_size=effect,
            baseline_ci=baseline_result.metrics.get(metric, ConfidenceInterval(0, 0, 0.95, 0)),
            treatment_ci=treatment_result.metrics.get(metric, ConfidenceInterval(0, 0, 0.95, 0)),
            conclusion=conclusion,
        )

    def save_results(self) -> Path:
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{self.name}_{timestamp}.json"

        data = {
            "experiment": self.name,
            "timestamp": timestamp,
            "n_seeds": self.n_seeds,
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "results": [asdict(r) for r in self.results],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {path}")
        return path

    def print_summary(self) -> None:
        """Print a summary of results."""
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {self.name}")
        print("=" * 70)

        conditions = list(set(r.condition for r in self.results))
        for condition in conditions:
            agg = self.aggregate_condition(condition)
            print(f"\n{condition} (n={agg.n_trials}):")
            for metric, ci in agg.metrics.items():
                print(f"  {metric}: {ci.point_estimate:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")


class ExperimentSuite:
    """Collection of related experiments."""

    def __init__(self, name: str, experiments: list[Experiment]):
        self.name = name
        self.experiments = experiments

    def run_all(self) -> None:
        """Run all experiments in the suite."""
        for exp in self.experiments:
            logger.info(f"Running experiment: {exp.name}")
            exp.run()
            exp.save_results()
            exp.print_summary()
