"""Benchmark core models (Section 18.1).

Defines the versioned, reproducible benchmark manifest and data split contracts.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import (
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class StratificationDimension(StrictModel):
    """A single stratification dimension for data splitting.

    Stratification ensures balanced representation across development,
    validation, and frozen-test sets.
    """

    dimension: NonEmptyString
    values: tuple[NonEmptyString, ...]
    proportions: tuple[float, ...]


class DataSplit(StrictModel):
    """One named data split within a benchmark.

    Splits are registered before tuning and never change without creating
    a new benchmark version.
    """

    split_id: OpaqueUuid
    name: NonEmptyString
    mcap_ids: tuple[OpaqueUuid, ...]
    purpose: Literal["DEVELOPMENT", "VALIDATION", "FROZEN_TEST"]
    stratification_dimensions: tuple[StratificationDimension, ...]


class BenchmarkManifest(StrictModel):
    """Immutable manifest for one benchmark run (Section 18.1).

    Pins every versioned component so that the benchmark is reproducible
    and comparable across runs.
    """

    schema_version: Literal["1.0"]
    benchmark_id: OpaqueUuid
    version: SchemaVersion
    annotation_version: SchemaVersion
    mcap_ids: tuple[OpaqueUuid, ...]
    data_split: DataSplit
    sampling_plan_version: SchemaVersion
    adaptive_feature_detector_version: SchemaVersion
    random_seed: NonNegativeInt
    alignment_version: SchemaVersion
    decoder_version: SchemaVersion
    image_encoder_version: SchemaVersion
    package_schema_version: SchemaVersion
    provider: NonEmptyString
    model: NonEmptyString
    adapter_version: SchemaVersion
    prompt_version: SchemaVersion
    output_schema_version: SchemaVersion
    generation_config: dict[NonEmptyString, str | int | float | bool | None]
    fusion_version: SchemaVersion
    qa_version: SchemaVersion
    event_reduction_version: SchemaVersion
    calibration_version: SchemaVersion
    ontology_version: SchemaVersion
    hardware_class: NonEmptyString
    provider_region: NonEmptyString
    time_window: tuple[Nanoseconds, Nanoseconds]
    quota: NonNegativeInt
    package_hashes: tuple[Sha256Digest, ...]


__all__ = [
    "BenchmarkManifest",
    "DataSplit",
    "StratificationDimension",
]
