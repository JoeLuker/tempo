"""Logits distribution analysis for mechanistic interpretability.

Analyzes logits distributions to understand if isolated vs visible modes
produce different probability distributions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.special import softmax
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import logging

from .experiment_loader import ExperimentData, LogitsData

logger = logging.getLogger(__name__)


@dataclass
class DistributionComparison:
    """Comparison between two probability distributions."""
    step: int
    kl_divergence: float  # KL(P || Q)
    js_divergence: float  # Jensen-Shannon divergence (symmetric)
    cosine_similarity: float  # 1 - cosine distance
    top_k_overlap: Dict[int, float]  # k -> overlap ratio

    def __post_init__(self):
        """Validate data."""
        assert 0.0 <= self.cosine_similarity <= 1.0, "Cosine similarity must be in [0, 1]"
        assert self.kl_divergence >= 0.0, "KL divergence must be non-negative"
        assert 0.0 <= self.js_divergence <= 1.0, "JS divergence must be in [0, 1]"


@dataclass
class LogitsComparisonResult:
    """Result of comparing logits between experiments."""
    exp1_name: str
    exp2_name: str

    # Per-step comparisons
    step_comparisons: List[DistributionComparison]

    # Aggregate metrics
    mean_kl_divergence: float
    mean_js_divergence: float
    mean_cosine_similarity: float
    mean_top_k_overlaps: Dict[int, float]  # k -> average overlap

    @property
    def num_steps(self) -> int:
        """Number of steps compared."""
        return len(self.step_comparisons)


class LogitsAnalyzer:
    """Analyzes logits distributions from experiments."""

    def __init__(self, debug_mode: bool = False):
        """Initialize logits analyzer.

        Args:
            debug_mode: Enable detailed logging
        """
        self.debug_mode = debug_mode
        if debug_mode:
            logger.setLevel(logging.DEBUG)

    def compare_distributions(
        self,
        exp1: ExperimentData,
        exp2: ExperimentData,
        top_k_values: List[int] = [1, 5, 10, 20, 50]
    ) -> LogitsComparisonResult:
        """Compare probability distributions between two experiments.

        This answers: Do isolated and visible modes produce the same distributions?

        Args:
            exp1: First experiment
            exp2: Second experiment
            top_k_values: K values for top-k overlap analysis

        Returns:
            LogitsComparisonResult with comparison metrics
        """
        assert exp1.has_logits and exp2.has_logits, "Both experiments must have logits data"

        # Find common steps
        common_steps = set(exp1.logits.steps.keys()) & set(exp2.logits.steps.keys())

        if not common_steps:
            raise ValueError("No common steps between experiments")

        logger.info(f"Comparing logits for {len(common_steps)} common steps")

        step_comparisons = []

        for step in sorted(common_steps):
            logits1 = exp1.logits.steps[step]
            logits2 = exp2.logits.steps[step]

            # Ensure same shape
            if logits1.shape != logits2.shape:
                logger.warning(f"Step {step}: shape mismatch {logits1.shape} vs {logits2.shape}")
                continue

            # Convert to probabilities
            probs1 = softmax(logits1, axis=-1).flatten()
            probs2 = softmax(logits2, axis=-1).flatten()

            # Compute metrics
            kl_div = self._kl_divergence(probs1, probs2)
            js_div = self._js_divergence(probs1, probs2)
            cos_sim = 1.0 - cosine(probs1, probs2)

            # Top-k overlap
            top_k_overlaps = {}
            for k in top_k_values:
                overlap = self._top_k_overlap(probs1, probs2, k)
                top_k_overlaps[k] = overlap

            comparison = DistributionComparison(
                step=step,
                kl_divergence=float(kl_div),
                js_divergence=float(js_div),
                cosine_similarity=float(cos_sim),
                top_k_overlap=top_k_overlaps
            )

            step_comparisons.append(comparison)

            if self.debug_mode:
                logger.debug(f"Step {step}:")
                logger.debug(f"  KL divergence: {kl_div:.6f}")
                logger.debug(f"  JS divergence: {js_div:.6f}")
                logger.debug(f"  Cosine similarity: {cos_sim:.6f}")
                logger.debug(f"  Top-1 overlap: {top_k_overlaps.get(1, 0):.3f}")

        # Compute aggregate metrics
        mean_kl = np.mean([c.kl_divergence for c in step_comparisons])
        mean_js = np.mean([c.js_divergence for c in step_comparisons])
        mean_cos = np.mean([c.cosine_similarity for c in step_comparisons])

        mean_top_k = {}
        for k in top_k_values:
            mean_top_k[k] = np.mean([c.top_k_overlap[k] for c in step_comparisons])

        result = LogitsComparisonResult(
            exp1_name=exp1.experiment_name,
            exp2_name=exp2.experiment_name,
            step_comparisons=step_comparisons,
            mean_kl_divergence=float(mean_kl),
            mean_js_divergence=float(mean_js),
            mean_cosine_similarity=float(mean_cos),
            mean_top_k_overlaps=mean_top_k
        )

        logger.info(f"Comparison complete:")
        logger.info(f"  Mean KL divergence: {mean_kl:.6f}")
        logger.info(f"  Mean JS divergence: {mean_js:.6f}")
        logger.info(f"  Mean cosine similarity: {mean_cos:.6f}")
        logger.info(f"  Mean top-1 overlap: {mean_top_k.get(1, 0):.3f}")

        return result

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Compute KL divergence KL(P || Q).

        Args:
            p: Probability distribution P
            q: Probability distribution Q
            epsilon: Small value to avoid log(0)

        Returns:
            KL divergence value
        """
        # Add epsilon to avoid log(0)
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)

        return entropy(p, q)

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence (symmetric).

        Args:
            p: Probability distribution P
            q: Probability distribution Q

        Returns:
            JS divergence value in [0, 1]
        """
        # Average distribution
        m = 0.5 * (p + q)

        # JS divergence = 0.5 * (KL(P||M) + KL(Q||M))
        js = 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)

        # Normalize to [0, 1] (max JS divergence is log(2))
        return js / np.log(2)

    def _top_k_overlap(self, p: np.ndarray, q: np.ndarray, k: int) -> float:
        """Compute overlap between top-k tokens.

        Args:
            p: Probability distribution P
            q: Probability distribution Q
            k: Number of top tokens to consider

        Returns:
            Overlap ratio in [0, 1]
        """
        # Get top-k indices
        top_k_p = set(np.argsort(p)[-k:])
        top_k_q = set(np.argsort(q)[-k:])

        # Compute intersection
        overlap = len(top_k_p & top_k_q)

        return overlap / k

    def analyze_distribution_entropy(
        self,
        experiment: ExperimentData
    ) -> Dict[int, float]:
        """Compute entropy of distributions at each step.

        High entropy = more uncertainty (flatter distribution)
        Low entropy = more confident (peaked distribution)

        Args:
            experiment: Experiment data with logits

        Returns:
            Dictionary mapping step to entropy value
        """
        assert experiment.has_logits, "Experiment must have logits data"

        entropies = {}

        for step, logits in experiment.logits.steps.items():
            probs = softmax(logits, axis=-1).flatten()
            ent = entropy(probs)
            entropies[step] = float(ent)

            if self.debug_mode:
                logger.debug(f"Step {step}: entropy = {ent:.4f}")

        return entropies

    def get_top_k_tokens(
        self,
        experiment: ExperimentData,
        step: int,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """Get top-k token IDs and their probabilities.

        Args:
            experiment: Experiment data with logits
            step: Generation step
            k: Number of top tokens

        Returns:
            List of (token_id, probability) tuples, sorted by probability
        """
        assert experiment.has_logits, "Experiment must have logits data"
        assert step in experiment.logits.steps, f"Step {step} not found"

        logits = experiment.logits.steps[step].flatten()
        probs = softmax(logits)

        # Get top-k indices
        top_indices = np.argsort(probs)[-k:][::-1]

        return [(int(idx), float(probs[idx])) for idx in top_indices]
