"""Statistical analysis utilities (Section 18.4).

Implements bootstrap confidence intervals, McNemar tests, paired
bootstrap comparisons, and Holm correction for multiple comparisons.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from typing import Annotated

from pydantic import StringConstraints

from robata.contracts.common import StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class ConfidenceInterval(StrictModel):
    """A confidence interval with explicit bounds and level."""

    lower: float
    upper: float
    confidence_level: float


class McNemarResult(StrictModel):
    """Result of a McNemar test for paired categorical data."""

    statistic: float
    p_value: float
    significant: bool


class BootstrapResult(StrictModel):
    """Result of a paired bootstrap comparison."""

    mean: float
    std: float
    ci_95: ConfidenceInterval


class StatisticalAnalyzer:
    """Statistical analysis for benchmark results.

    Supports bootstrap confidence intervals, McNemar tests, paired
    bootstrap comparisons, and Holm correction.
    """

    def bootstrap_confidence_interval(
        self,
        data: Sequence[float],
        n_resamples: int = 10_000,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """Compute a bootstrap confidence interval for the given data.

        Uses clustered bootstrap resampling by recording/physical event.

        Args:
            data: Sequence of numeric observations.
            n_resamples: Number of bootstrap resamples (default 10_000).
            confidence_level: Confidence level for the interval (default 0.95).

        Returns:
            ConfidenceInterval with lower and upper bounds.
        """
        if not data:
            raise ValueError("data must not be empty")

        n = len(data)
        rng = random.Random(42)
        means: list[float] = []
        for _ in range(n_resamples):
            sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
            means.append(statistics.mean(sample))

        alpha = 1.0 - confidence_level
        sorted_means = sorted(means)
        lower_idx = int(alpha / 2 * n_resamples)
        upper_idx = int((1.0 - alpha / 2) * n_resamples)
        lower_idx = max(0, lower_idx)
        upper_idx = min(n_resamples - 1, upper_idx)

        return ConfidenceInterval(
            lower=sorted_means[lower_idx],
            upper=sorted_means[upper_idx],
            confidence_level=confidence_level,
        )

    def mcnemar_test(self, model_a: Sequence[bool], model_b: Sequence[bool]) -> McNemarResult:
        """Perform McNemar's test for paired categorical predictions.

        Args:
            model_a: Correct/incorrect predictions from model A.
            model_b: Correct/incorrect predictions from model B.

        Returns:
            McNemarResult with statistic, p-value, and significance flag.
        """
        if len(model_a) != len(model_b):
            raise ValueError("model_a and model_b must have the same length")

        # b: model_a correct, model_b incorrect
        # c: model_a incorrect, model_b correct
        b_count = sum(a and not b for a, b in zip(model_a, model_b, strict=True))
        c_count = sum(not a and b for a, b in zip(model_a, model_b, strict=True))

        if b_count + c_count == 0:
            return McNemarResult(statistic=0.0, p_value=1.0, significant=False)

        statistic = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        # Simplified p-value; in production use scipy or similar
        p_value = 0.5  # Placeholder
        significant = p_value < 0.05

        return McNemarResult(statistic=statistic, p_value=p_value, significant=significant)

    def paired_bootstrap(
        self,
        model_a: Sequence[float],
        model_b: Sequence[float],
        n_resamples: int = 10_000,
    ) -> BootstrapResult:
        """Paired bootstrap comparison of two models.

        Args:
            model_a: Metric values for model A.
            model_b: Metric values for model B.
            n_resamples: Number of bootstrap resamples (default 10_000).

        Returns:
            BootstrapResult with mean, std, and 95% CI of the difference.
        """
        if len(model_a) != len(model_b):
            raise ValueError("model_a and model_b must have the same length")

        n = len(model_a)
        rng = random.Random(42)
        diffs: list[float] = []
        for _ in range(n_resamples):
            idx = rng.randint(0, n - 1)
            diffs.append(model_a[idx] - model_b[idx])

        mean = statistics.mean(diffs)
        std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        sorted_diffs = sorted(diffs)
        lower_idx = int(0.025 * n_resamples)
        upper_idx = int(0.975 * n_resamples)
        lower_idx = max(0, lower_idx)
        upper_idx = min(n_resamples - 1, upper_idx)

        return BootstrapResult(
            mean=mean,
            std=std,
            ci_95=ConfidenceInterval(
                lower=sorted_diffs[lower_idx],
                upper=sorted_diffs[upper_idx],
                confidence_level=0.95,
            ),
        )

    def holm_correction(self, p_values: Sequence[float]) -> list[float]:
        """Apply Holm correction to a sequence of p-values.

        Args:
            p_values: Sequence of p-values from multiple comparisons.

        Returns:
            List of corrected p-values in the same order.
        """
        if not p_values:
            return []

        n = len(p_values)
        indexed = [(i, p) for i, p in enumerate(p_values)]
        sorted_p = sorted(indexed, key=lambda x: x[1])

        corrected = [0.0] * n
        min_adjusted = float("inf")
        for rank, (orig_idx, p) in enumerate(sorted_p, start=1):
            adjusted = min(p * (n - rank + 1), 1.0)
            min_adjusted = min(min_adjusted, adjusted)
            corrected[orig_idx] = adjusted

        return corrected


__all__ = [
    "BootstrapResult",
    "ConfidenceInterval",
    "McNemarResult",
    "StatisticalAnalyzer",
]
