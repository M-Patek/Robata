from __future__ import annotations

import math

import pytest

from robata.benchmark import StatisticalAnalyzer


def test_mcnemar_uses_exact_two_sided_binomial_probability() -> None:
    result = StatisticalAnalyzer().mcnemar_test(
        (True,) * 10,
        (False,) * 10,
    )

    assert result.statistic == pytest.approx(8.1)
    assert result.p_value == pytest.approx(2 / 1024)
    assert result.significant is True


def test_mcnemar_requires_nonempty_strict_boolean_pairs() -> None:
    analyzer = StatisticalAnalyzer()
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.mcnemar_test((), ())
    with pytest.raises(ValueError, match="must contain booleans"):
        analyzer.mcnemar_test((1,), (True,))  # type: ignore[arg-type]


def test_paired_bootstrap_reports_observed_mean_and_resamples_pairs() -> None:
    result = StatisticalAnalyzer().paired_bootstrap(
        (0.0, 1.0),
        (0.0, 0.0),
        n_resamples=4,
    )

    assert result.mean == pytest.approx(0.5)
    assert result.std == pytest.approx(0.25)
    assert result.ci_95.lower == pytest.approx(0.0)
    assert result.ci_95.upper == pytest.approx(0.4625)


def test_bootstrap_rejects_invalid_parameters_and_nonfinite_values() -> None:
    analyzer = StatisticalAnalyzer()
    with pytest.raises(ValueError, match="positive integer"):
        analyzer.bootstrap_confidence_interval((1.0,), n_resamples=0)
    with pytest.raises(ValueError, match="strictly between"):
        analyzer.bootstrap_confidence_interval((1.0,), confidence_level=1.0)
    with pytest.raises(ValueError, match="finite"):
        analyzer.bootstrap_confidence_interval((math.nan,))


def test_clustered_bootstrap_resamples_whole_clusters_independent_of_mapping_order() -> None:
    analyzer = StatisticalAnalyzer()
    result = analyzer.clustered_bootstrap_confidence_interval(
        {"recording-b": (10.0,), "recording-a": (0.0, 0.0)},
        n_resamples=4,
    )
    reordered = analyzer.clustered_bootstrap_confidence_interval(
        {"recording-a": (0.0, 0.0), "recording-b": (10.0,)},
        n_resamples=4,
    )

    # With seed 42 the four cluster draws yield means 0, 10/3, 0, 0.
    # The 10/3 sample includes both observations from recording-a.
    assert result == reordered
    assert result.lower == pytest.approx(0.0)
    assert result.upper == pytest.approx(37 / 12)


def test_clustered_paired_bootstrap_resamples_matched_whole_clusters() -> None:
    analyzer = StatisticalAnalyzer()
    result = analyzer.clustered_paired_bootstrap(
        {"event-b": (10.0,), "event-a": (1.0, 3.0)},
        {"event-a": (0.0, 0.0), "event-b": (4.0,)},
        n_resamples=4,
    )
    reordered = analyzer.clustered_paired_bootstrap(
        {"event-a": (1.0, 3.0), "event-b": (10.0,)},
        {"event-b": (4.0,), "event-a": (0.0, 0.0)},
        n_resamples=4,
    )

    assert result == reordered
    assert result.mean == pytest.approx(10 / 3)
    assert result.std == pytest.approx(2 / 3)
    assert result.ci_95.lower == pytest.approx(2.0)
    assert result.ci_95.upper == pytest.approx(97 / 30)


def test_clustered_bootstrap_strictly_validates_clusters_and_seed() -> None:
    analyzer = StatisticalAnalyzer()
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.clustered_bootstrap_confidence_interval({})
    with pytest.raises(ValueError, match="non-empty strings"):
        analyzer.clustered_bootstrap_confidence_interval({"": (1.0,)})
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.clustered_bootstrap_confidence_interval({"recording": ()})
    with pytest.raises(ValueError, match="numeric sequences"):
        analyzer.clustered_bootstrap_confidence_interval(
            {"recording": 1.0},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="real numbers"):
        analyzer.clustered_bootstrap_confidence_interval({"recording": (True,)})
    with pytest.raises(ValueError, match="finite"):
        analyzer.clustered_bootstrap_confidence_interval({"recording": (math.inf,)})
    with pytest.raises(ValueError, match="seed must be an integer"):
        analyzer.clustered_bootstrap_confidence_interval(
            {"recording": (1.0,)},
            seed=True,  # type: ignore[arg-type]
        )


def test_clustered_paired_bootstrap_requires_identical_pair_structure() -> None:
    analyzer = StatisticalAnalyzer()
    with pytest.raises(ValueError, match="identical cluster IDs"):
        analyzer.clustered_paired_bootstrap({"a": (1.0,)}, {"b": (1.0,)})
    with pytest.raises(ValueError, match="same number of paired observations"):
        analyzer.clustered_paired_bootstrap({"a": (1.0, 2.0)}, {"a": (1.0,)})


def test_holm_correction_is_monotone_and_restores_original_order() -> None:
    corrected = StatisticalAnalyzer().holm_correction((0.01, 0.04, 0.03))

    assert corrected == pytest.approx([0.03, 0.06, 0.06])


def test_holm_rejects_values_outside_probability_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        StatisticalAnalyzer().holm_correction((0.2, 1.1))
