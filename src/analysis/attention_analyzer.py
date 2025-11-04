"""Attention pattern analysis for mechanistic interpretability.

Analyzes attention weights to understand how tokens attend to each other,
with special focus on parallel token interactions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

from .experiment_loader import ExperimentData, AttentionData

logger = logging.getLogger(__name__)


@dataclass
class AttentionStats:
    """Statistics about attention patterns."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile


@dataclass
class CrossParallelAttention:
    """Analysis of attention between parallel tokens at same position."""
    step: int
    parallel_count: int
    parallel_positions: List[int]

    # Attention from parallel tokens to each other
    mean_cross_attention: float
    max_cross_attention: float
    min_cross_attention: float

    # Attention from parallel tokens to prior context
    mean_prior_attention: float
    max_prior_attention: float

    # Ratio: how much more do they attend to prior context vs each other
    attention_ratio: float  # prior / cross


@dataclass
class AttentionComparisonResult:
    """Result of comparing attention patterns between experiments."""
    exp1_name: str
    exp2_name: str

    # Overall similarity metrics
    mean_absolute_difference: float
    max_absolute_difference: float
    correlation: float

    # Per-step differences
    step_differences: Dict[int, float] = field(default_factory=dict)

    # Layer-wise analysis
    layer_correlations: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Validate data."""
        assert -1.0 <= self.correlation <= 1.0, "Correlation must be between -1 and 1"


class AttentionAnalyzer:
    """Analyzes attention patterns from experiments."""

    def __init__(self, debug_mode: bool = False):
        """Initialize attention analyzer.

        Args:
            debug_mode: Enable detailed logging
        """
        self.debug_mode = debug_mode
        if debug_mode:
            logger.setLevel(logging.DEBUG)

    def analyze_cross_parallel_attention(
        self,
        experiment: ExperimentData
    ) -> List[CrossParallelAttention]:
        """Analyze attention between parallel tokens at same position.

        This is the key analysis for understanding if parallel tokens
        attend to each other (visible mode) or are isolated.

        Args:
            experiment: Experiment data with attention weights

        Returns:
            List of CrossParallelAttention for each parallel step
        """
        assert experiment.has_attention, "Experiment must have attention data"
        assert experiment.parallel_sets is not None, "Experiment must have parallel sets"

        results = []

        for parallel_set in experiment.parallel_sets.sets:
            step = parallel_set['step']
            positions = parallel_set['positions']
            count = parallel_set['count']

            if step not in experiment.attention.steps:
                logger.warning(f"Step {step} not found in attention data")
                continue

            # Get attention for this step: (num_layers, batch, heads, seq_len, seq_len)
            attention = experiment.attention.steps[step]

            # Average across layers and heads: (seq_len, seq_len)
            avg_attention = attention.mean(axis=(0, 1, 2))

            # Extract attention from parallel positions to all positions
            parallel_attention = avg_attention[positions, :]  # (num_parallel, seq_len)

            # Cross-parallel: attention from one parallel token to another
            cross_parallel = parallel_attention[:, positions]  # (num_parallel, num_parallel)

            # Exclude self-attention (diagonal)
            mask = ~np.eye(count, dtype=bool)
            cross_values = cross_parallel[mask]

            # Prior context: attention to tokens before this parallel set
            prior_positions = list(range(0, min(positions)))
            if prior_positions:
                prior_attention = parallel_attention[:, prior_positions]  # (num_parallel, num_prior)
                mean_prior = prior_attention.mean()
                max_prior = prior_attention.max()
            else:
                mean_prior = 0.0
                max_prior = 0.0

            # Calculate statistics
            mean_cross = cross_values.mean() if len(cross_values) > 0 else 0.0
            max_cross = cross_values.max() if len(cross_values) > 0 else 0.0
            min_cross = cross_values.min() if len(cross_values) > 0 else 0.0

            # Ratio: how much more attention to prior vs siblings
            if mean_cross > 0:
                ratio = mean_prior / mean_cross
            else:
                ratio = float('inf') if mean_prior > 0 else 1.0

            result = CrossParallelAttention(
                step=step,
                parallel_count=count,
                parallel_positions=positions,
                mean_cross_attention=float(mean_cross),
                max_cross_attention=float(max_cross),
                min_cross_attention=float(min_cross),
                mean_prior_attention=float(mean_prior),
                max_prior_attention=float(max_prior),
                attention_ratio=float(ratio)
            )

            results.append(result)

            if self.debug_mode:
                logger.debug(f"Step {step}: {count} parallel tokens")
                logger.debug(f"  Cross-parallel attention: {mean_cross:.6f} (max: {max_cross:.6f})")
                logger.debug(f"  Prior context attention: {mean_prior:.6f}")
                logger.debug(f"  Ratio (prior/cross): {ratio:.2f}")

        return results

    def compare_attention_patterns(
        self,
        exp1: ExperimentData,
        exp2: ExperimentData
    ) -> AttentionComparisonResult:
        """Compare attention patterns between two experiments.

        This is used to compare isolated vs visible modes.

        Args:
            exp1: First experiment
            exp2: Second experiment

        Returns:
            AttentionComparisonResult with comparison metrics
        """
        assert exp1.has_attention and exp2.has_attention, "Both experiments must have attention data"

        # Find common steps
        common_steps = set(exp1.attention.steps.keys()) & set(exp2.attention.steps.keys())

        if not common_steps:
            raise ValueError("No common steps between experiments")

        logger.info(f"Comparing {len(common_steps)} common steps")

        all_diffs = []
        step_diffs = {}

        for step in sorted(common_steps):
            attn1 = exp1.attention.steps[step]
            attn2 = exp2.attention.steps[step]

            # Ensure same shape
            if attn1.shape != attn2.shape:
                logger.warning(f"Step {step}: shape mismatch {attn1.shape} vs {attn2.shape}")
                continue

            # Calculate absolute difference
            diff = np.abs(attn1 - attn2)
            mean_diff = diff.mean()

            step_diffs[step] = float(mean_diff)
            all_diffs.append(diff.flatten())

        # Overall statistics
        all_diffs_array = np.concatenate(all_diffs)
        mean_abs_diff = float(all_diffs_array.mean())
        max_abs_diff = float(all_diffs_array.max())

        # Correlation across all attention values
        attn1_values = []
        attn2_values = []
        for step in sorted(common_steps):
            attn1_values.append(exp1.attention.steps[step].flatten())
            attn2_values.append(exp2.attention.steps[step].flatten())

        attn1_all = np.concatenate(attn1_values)
        attn2_all = np.concatenate(attn2_values)
        correlation = float(np.corrcoef(attn1_all, attn2_all)[0, 1])

        # Layer-wise correlations
        if exp1.attention.num_layers > 0:
            layer_corrs = []
            for layer in range(exp1.attention.num_layers):
                layer1_values = []
                layer2_values = []
                for step in sorted(common_steps):
                    layer1_values.append(exp1.attention.steps[step][layer].flatten())
                    layer2_values.append(exp2.attention.steps[step][layer].flatten())

                layer1_all = np.concatenate(layer1_values)
                layer2_all = np.concatenate(layer2_values)
                corr = np.corrcoef(layer1_all, layer2_all)[0, 1]
                layer_corrs.append(corr)

            layer_correlations = np.array(layer_corrs)
        else:
            layer_correlations = np.array([])

        result = AttentionComparisonResult(
            exp1_name=exp1.experiment_name,
            exp2_name=exp2.experiment_name,
            mean_absolute_difference=mean_abs_diff,
            max_absolute_difference=max_abs_diff,
            correlation=correlation,
            step_differences=step_diffs,
            layer_correlations=layer_correlations
        )

        logger.info(f"Comparison complete:")
        logger.info(f"  Mean absolute difference: {mean_abs_diff:.6f}")
        logger.info(f"  Correlation: {correlation:.4f}")

        return result

    def compute_attention_statistics(
        self,
        experiment: ExperimentData,
        step: int
    ) -> AttentionStats:
        """Compute statistics for attention at a specific step.

        Args:
            experiment: Experiment data
            step: Generation step

        Returns:
            AttentionStats with computed metrics
        """
        assert experiment.has_attention, "Experiment must have attention data"
        assert step in experiment.attention.steps, f"Step {step} not found"

        attention = experiment.attention.steps[step]

        # Flatten for statistics
        values = attention.flatten()

        return AttentionStats(
            mean=float(values.mean()),
            std=float(values.std()),
            min=float(values.min()),
            max=float(values.max()),
            median=float(np.median(values)),
            q25=float(np.percentile(values, 25)),
            q75=float(np.percentile(values, 75))
        )
