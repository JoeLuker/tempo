"""Statistical metrics for experiment analysis.

Provides rigorous statistical tools for hypothesis testing,
effect size calculation, and power analysis.
"""

import math
from dataclasses import dataclass
from typing import Optional
from scipy import stats
import numpy as np


@dataclass
class EffectSize:
    """Cohen's d effect size with interpretation."""
    d: float
    interpretation: str  # "negligible", "small", "medium", "large"

    @classmethod
    def from_groups(cls, group1: list[float], group2: list[float]) -> "EffectSize":
        """Compute Cohen's d from two groups."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return cls(d=0.0, interpretation="insufficient_data")

        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std < 1e-10:
            return cls(d=0.0, interpretation="no_variance")

        d = (mean1 - mean2) / pooled_std

        # Cohen's conventions
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return cls(d=d, interpretation=interpretation)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a statistic."""
    lower: float
    upper: float
    confidence: float  # e.g., 0.95 for 95% CI
    point_estimate: float


def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Compute confidence interval for the mean."""
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0.0
        return ConfidenceInterval(
            lower=mean, upper=mean,
            confidence=confidence, point_estimate=mean
        )

    mean = np.mean(values)
    sem = stats.sem(values)

    # t-distribution critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * sem

    return ConfidenceInterval(
        lower=mean - margin,
        upper=mean + margin,
        confidence=confidence,
        point_estimate=mean,
    )


def compute_effect_size(group1: list[float], group2: list[float]) -> EffectSize:
    """Compute Cohen's d effect size between two groups."""
    return EffectSize.from_groups(group1, group2)


@dataclass
class PowerAnalysis:
    """Result of power analysis."""
    required_n: int
    actual_power: float
    effect_size: float
    alpha: float


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> PowerAnalysis:
    """Compute required sample size for desired power.

    Args:
        effect_size: Expected Cohen's d effect size
        alpha: Significance level (Type I error rate)
        power: Desired power (1 - Type II error rate)
        ratio: Ratio of group sizes (n2/n1)

    Returns:
        PowerAnalysis with required sample size per group
    """
    if abs(effect_size) < 0.01:
        return PowerAnalysis(
            required_n=float('inf'),
            actual_power=0.0,
            effect_size=effect_size,
            alpha=alpha,
        )

    # Use normal approximation for two-sample t-test
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Sample size formula for two-sample t-test
    n1 = (1 + 1/ratio) * ((z_alpha + z_beta) / effect_size) ** 2
    n1 = math.ceil(n1)

    return PowerAnalysis(
        required_n=n1,
        actual_power=power,
        effect_size=effect_size,
        alpha=alpha,
    )


@dataclass
class HypothesisTest:
    """Result of hypothesis test."""
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    test_type: str


def test_difference(
    group1: list[float],
    group2: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTest:
    """Perform t-test for difference between groups.

    Args:
        group1: First group of measurements
        group2: Second group of measurements
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"

    Returns:
        HypothesisTest result
    """
    if len(group1) < 2 or len(group2) < 2:
        return HypothesisTest(
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            test_type="insufficient_data",
        )

    statistic, p_value = stats.ttest_ind(group1, group2, alternative=alternative)

    return HypothesisTest(
        statistic=statistic,
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
        test_type=f"welch_t_{alternative}",
    )


def test_proportion(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    alpha: float = 0.05,
) -> HypothesisTest:
    """Fisher's exact test for proportion difference (e.g., recall rates).

    Args:
        successes1: Number of successes in group 1
        n1: Total trials in group 1
        successes2: Number of successes in group 2
        n2: Total trials in group 2
        alpha: Significance level

    Returns:
        HypothesisTest result
    """
    # Contingency table
    table = [
        [successes1, n1 - successes1],
        [successes2, n2 - successes2],
    ]

    _, p_value = stats.fisher_exact(table, alternative='two-sided')

    return HypothesisTest(
        statistic=0.0,  # Fisher's exact doesn't have a test statistic
        p_value=p_value,
        reject_null=p_value < alpha,
        alpha=alpha,
        test_type="fisher_exact",
    )
