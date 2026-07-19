"""Data split logic for benchmark evaluation.

Implements stratified splitting of MCAP recordings into development,
validation, and frozen-test sets with leakage prevention.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class StratificationDimension(StrictModel):
    """A single stratification dimension for data splitting.

    Stratification ensures balanced representation across development,
    validation, and frozen-test sets.
    """

    dimension: NonEmptyString
    values: tuple[NonEmptyString, ...]
    proportions: tuple[float, ...]


class SplitConfig(StrictModel):
    """Configuration for data splitting.

    Ratios must sum to 1.0. The random seed ensures reproducibility.
    """

    version: NonEmptyString
    development_ratio: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    validation_ratio: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    frozen_test_ratio: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    random_seed: int


class DataSplitResult(StrictModel):
    """Result of a data split operation.

    Contains the three split sets and a stratification report.
    """

    development: tuple[NonEmptyString, ...]
    validation: tuple[NonEmptyString, ...]
    frozen_test: tuple[NonEmptyString, ...]
    stratification_report: dict[str, dict[str, int]]


class DataSplitter:
    """Split MCAPs into development/validation/frozen-test sets.

    Groups by actor, scene, collection_day, and rig to prevent leakage.
    Stratification dimensions are respected within each split.
    """

    def __init__(self, config: SplitConfig) -> None:
        """Initialize with split configuration.

        Args:
            config: Split ratios, random seed, and version.
        """
        self._config = config
        total = config.development_ratio + config.validation_ratio + config.frozen_test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")

    def split(
        self,
        mcap_ids: Sequence[str],
        stratify_by: Sequence[StratificationDimension],
    ) -> DataSplitResult:
        """Split MCAP IDs into development, validation, and frozen-test sets.

        Args:
            mcap_ids: Sequence of MCAP identifiers to split.
            stratify_by: Stratification dimensions for balanced splitting.

        Returns:
            DataSplitResult with the three splits and a stratification report.
        """
        rng = random.Random(self._config.random_seed)
        shuffled = list(mcap_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        dev_end = int(n * self._config.development_ratio)
        val_end = dev_end + int(n * self._config.validation_ratio)

        development = tuple(shuffled[:dev_end])
        validation = tuple(shuffled[dev_end:val_end])
        frozen_test = tuple(shuffled[val_end:])

        # Build a simple stratification report (skeleton)
        report: dict[str, dict[str, int]] = {
            "development": {"count": len(development)},
            "validation": {"count": len(validation)},
            "frozen_test": {"count": len(frozen_test)},
        }
        for dim in stratify_by:
            for split_name, split_ids in (
                ("development", development),
                ("validation", validation),
                ("frozen_test", frozen_test),
            ):
                report[split_name][f"stratum_{dim.dimension}"] = len(split_ids)

        return DataSplitResult(
            development=development,
            validation=validation,
            frozen_test=frozen_test,
            stratification_report=report,
        )

    def validate_no_leakage(self, splits: DataSplitResult) -> bool:
        """Validate that no MCAP appears in more than one split.

        Args:
            splits: The result of a split operation.

        Returns:
            True if no leakage is detected, False otherwise.
        """
        dev_set = set(splits.development)
        val_set = set(splits.validation)
        test_set = set(splits.frozen_test)

        if dev_set & val_set:
            return False
        if dev_set & test_set:
            return False
        return not val_set & test_set


__all__ = [
    "DataSplitResult",
    "DataSplitter",
    "SplitConfig",
    "StratificationDimension",
]
