"""Statistical analysis utilities (Section 18.4).

The explicit clustered APIs resample recording/physical-event clusters as
the independent units required by the benchmark design.  The original flat
bootstrap APIs remain available as IID development conveniences.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from math import comb, isfinite
from numbers import Real

from robata.contracts.common import StrictModel


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

    Governed benchmark evidence should use the clustered methods.  Flat IID
    methods are retained for compatibility and local exploratory analysis.
    """

    def bootstrap_confidence_interval(
        self,
        data: Sequence[float],
        n_resamples: int = 10_000,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """Compute an IID percentile bootstrap interval for flat observations.

        This compatibility API treats every value as independent.  It is
        suitable for local exploratory analysis, not governed benchmark
        evidence whose independent units are recordings or physical events;
        use :meth:`clustered_bootstrap_confidence_interval` for that case.

        Args:
            data: Sequence of numeric observations.
            n_resamples: Number of bootstrap resamples (default 10_000).
            confidence_level: Confidence level for the interval (default 0.95).

        Returns:
            ConfidenceInterval with lower and upper bounds.
        """
        _validate_numeric_sequence(data, name="data")
        _validate_resamples(n_resamples)
        _validate_confidence_level(confidence_level)

        n = len(data)
        rng = random.Random(42)
        means: list[float] = []
        for _ in range(n_resamples):
            sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
            means.append(statistics.mean(sample))

        alpha = 1.0 - confidence_level
        sorted_means = sorted(means)

        return ConfidenceInterval(
            lower=_percentile(sorted_means, alpha / 2),
            upper=_percentile(sorted_means, 1.0 - alpha / 2),
            confidence_level=confidence_level,
        )

    def clustered_bootstrap_confidence_interval(
        self,
        clusters: Mapping[str, Sequence[float]],
        n_resamples: int = 10_000,
        confidence_level: float = 0.95,
        *,
        seed: int = 42,
    ) -> ConfidenceInterval:
        """Bootstrap an interval by resampling whole recording/event clusters.

        ``clusters`` maps a stable recording or physical-event identifier to
        all observations belonging to that independent unit.  Each resample
        draws the same number of cluster identifiers with replacement and
        includes every observation from each selected cluster.  The statistic
        is therefore the observation-weighted mean after whole-cluster
        resampling.  Cluster identifiers are sorted before sampling, so mapping
        insertion order cannot change a fixed-seed result.
        """
        normalized = _normalize_clusters(clusters, name="clusters")
        _validate_resamples(n_resamples)
        _validate_confidence_level(confidence_level)
        _validate_seed(seed)

        means = _resample_cluster_means(normalized, n_resamples=n_resamples, seed=seed)
        alpha = 1.0 - confidence_level
        sorted_means = sorted(means)
        return ConfidenceInterval(
            lower=_percentile(sorted_means, alpha / 2),
            upper=_percentile(sorted_means, 1.0 - alpha / 2),
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
        _validate_bool_sequence(model_a, name="model_a")
        _validate_bool_sequence(model_b, name="model_b")
        if len(model_a) != len(model_b):
            raise ValueError("model_a and model_b must have the same length")
        if not model_a:
            raise ValueError("model_a and model_b must not be empty")

        # b: model_a correct, model_b incorrect
        # c: model_a incorrect, model_b correct
        b_count = sum(a and not b for a, b in zip(model_a, model_b, strict=True))
        c_count = sum(not a and b for a, b in zip(model_a, model_b, strict=True))

        if b_count + c_count == 0:
            return McNemarResult(statistic=0.0, p_value=1.0, significant=False)

        discordant = b_count + c_count
        statistic = (abs(b_count - c_count) - 1) ** 2 / discordant

        # Exact two-sided binomial test under the McNemar null hypothesis.
        # Computing the lower tail with integer binomial coefficients keeps the
        # result deterministic and avoids a scipy/runtime dependency.
        lower_tail = sum(comb(discordant, i) for i in range(min(b_count, c_count) + 1))
        p_value = min(1.0, 2 * lower_tail / (2**discordant))
        significant = p_value < 0.05

        return McNemarResult(statistic=statistic, p_value=p_value, significant=significant)

    def paired_bootstrap(
        self,
        model_a: Sequence[float],
        model_b: Sequence[float],
        n_resamples: int = 10_000,
    ) -> BootstrapResult:
        """IID paired bootstrap comparison of flat observations.

        This compatibility API assumes each paired position is independent.
        Use :meth:`clustered_paired_bootstrap` when pairs are nested within
        recordings or physical events.

        Args:
            model_a: Metric values for model A.
            model_b: Metric values for model B.
            n_resamples: Number of bootstrap resamples (default 10_000).

        Returns:
            BootstrapResult with mean, std, and 95% CI of the difference.
        """
        _validate_numeric_sequence(model_a, name="model_a")
        _validate_numeric_sequence(model_b, name="model_b")
        if len(model_a) != len(model_b):
            raise ValueError("model_a and model_b must have the same length")
        if not model_a:
            raise ValueError("model_a and model_b must not be empty")
        _validate_resamples(n_resamples)

        n = len(model_a)
        rng = random.Random(42)
        diffs: list[float] = []
        for _ in range(n_resamples):
            sample = [rng.randint(0, n - 1) for _ in range(n)]
            diffs.append(statistics.mean(model_a[idx] - model_b[idx] for idx in sample))

        mean = statistics.mean(model_a[idx] - model_b[idx] for idx in range(n))
        std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        sorted_diffs = sorted(diffs)

        return BootstrapResult(
            mean=mean,
            std=std,
            ci_95=ConfidenceInterval(
                lower=_percentile(sorted_diffs, 0.025),
                upper=_percentile(sorted_diffs, 0.975),
                confidence_level=0.95,
            ),
        )

    def clustered_paired_bootstrap(
        self,
        model_a: Mapping[str, Sequence[float]],
        model_b: Mapping[str, Sequence[float]],
        n_resamples: int = 10_000,
        *,
        seed: int = 42,
    ) -> BootstrapResult:
        """Compare models by resampling whole matched clusters.

        Both mappings must contain the same cluster identifiers and the same
        number of paired observations inside each cluster.  A resample uses
        one shared cluster draw for both models and retains every pair in each
        selected cluster.  The reported mean is the observed, pair-weighted
        mean difference ``model_a - model_b``; the interval is the percentile
        interval of whole-cluster resampled means.
        """
        normalized_a = _normalize_clusters(model_a, name="model_a")
        normalized_b = _normalize_clusters(model_b, name="model_b")
        _validate_resamples(n_resamples)
        _validate_seed(seed)

        differences = _paired_cluster_differences(normalized_a, normalized_b)
        resampled_differences = _resample_cluster_means(
            differences,
            n_resamples=n_resamples,
            seed=seed,
        )
        sorted_differences = sorted(resampled_differences)
        observed = [value for _, cluster in differences for value in cluster]

        return BootstrapResult(
            mean=statistics.mean(observed),
            std=(
                statistics.stdev(resampled_differences) if len(resampled_differences) > 1 else 0.0
            ),
            ci_95=ConfidenceInterval(
                lower=_percentile(sorted_differences, 0.025),
                upper=_percentile(sorted_differences, 0.975),
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
        if any(
            not isinstance(p, Real) or isinstance(p, bool) or not isfinite(float(p))
            for p in p_values
        ):
            raise ValueError("p_values must contain finite real numbers")
        if any(not 0.0 <= float(p) <= 1.0 for p in p_values):
            raise ValueError("p_values must be between 0 and 1")

        n = len(p_values)
        indexed = [(i, float(p)) for i, p in enumerate(p_values)]
        sorted_p = sorted(indexed, key=lambda x: x[1])

        corrected = [0.0] * n
        max_adjusted = 0.0
        for rank, (orig_idx, p) in enumerate(sorted_p, start=1):
            adjusted = min(p * (n - rank + 1), 1.0)
            max_adjusted = max(max_adjusted, adjusted)
            corrected[orig_idx] = max_adjusted

        return corrected


def _validate_numeric_sequence(values: Sequence[float], *, name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must contain real numbers")
    if any(not isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain finite numbers")


def _validate_bool_sequence(values: Sequence[bool], *, name: str) -> None:
    if any(not isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must contain booleans")


def _validate_resamples(n_resamples: int) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")


def _validate_confidence_level(confidence_level: float) -> None:
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not isfinite(float(confidence_level))
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _normalize_clusters(
    clusters: Mapping[str, Sequence[float]],
    *,
    name: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not isinstance(clusters, Mapping):
        raise ValueError(f"{name} must be a mapping of cluster IDs to observations")
    if not clusters:
        raise ValueError(f"{name} must not be empty")

    normalized: list[tuple[str, tuple[float, ...]]] = []
    for cluster_id, observations in clusters.items():
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError(f"{name} cluster IDs must be non-empty strings")
        if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
            raise ValueError(f"{name} cluster observations must be numeric sequences")
        _validate_numeric_sequence(observations, name=f"{name}[{cluster_id!r}]")
        normalized.append((cluster_id, tuple(float(value) for value in observations)))

    return tuple(sorted(normalized, key=lambda item: item[0]))


def _paired_cluster_differences(
    model_a: tuple[tuple[str, tuple[float, ...]], ...],
    model_b: tuple[tuple[str, tuple[float, ...]], ...],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    keys_a = tuple(cluster_id for cluster_id, _ in model_a)
    keys_b = tuple(cluster_id for cluster_id, _ in model_b)
    if keys_a != keys_b:
        raise ValueError("model_a and model_b must contain identical cluster IDs")

    differences: list[tuple[str, tuple[float, ...]]] = []
    for (cluster_id, values_a), (_, values_b) in zip(model_a, model_b, strict=True):
        if len(values_a) != len(values_b):
            raise ValueError(
                "model_a and model_b must contain the same number of paired "
                f"observations in cluster {cluster_id!r}"
            )
        cluster_differences = tuple(
            value_a - value_b for value_a, value_b in zip(values_a, values_b, strict=True)
        )
        if any(not isfinite(value) for value in cluster_differences):
            raise ValueError("paired differences must be finite")
        differences.append((cluster_id, cluster_differences))

    return tuple(differences)


def _resample_cluster_means(
    clusters: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    n_resamples: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    cluster_count = len(clusters)
    means: list[float] = []
    for _ in range(n_resamples):
        sampled_values: list[float] = []
        for _ in range(cluster_count):
            _, values = clusters[rng.randint(0, cluster_count - 1)]
            sampled_values.extend(values)
        means.append(statistics.mean(sampled_values))
    return means


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction)


__all__ = [
    "BootstrapResult",
    "ConfidenceInterval",
    "McNemarResult",
    "StatisticalAnalyzer",
]
