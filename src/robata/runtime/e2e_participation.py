"""Non-canonical run participation manifests for E2E qualification.

A :class:`E2ETraceRuntimeFragment` tells us what instrumentation observed.  It
cannot, by itself, tell us whether a boundary was intentionally skipped, not
configured, failed, or merely missing instrumentation.  This module binds an
operator-declared participation plan to one frozen runtime fragment and derives
an honest coverage conclusion without changing any published contract.

The resulting JSON is a qualification sidecar only.  It is never a canonical
identity, evidence, selection, or publication input.  The sidecar digest is
returned by :func:`write_e2e_participation_manifest` so callers can retain it
next to the trace/report digest.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.runtime.e2e_trace import (
    E2ETraceFragmentRole,
    E2ETraceMeasurementStatus,
    E2ETraceRuntimeFragment,
    E2ETraceStage,
)

PARTICIPATION_MANIFEST_VERSION: Literal["robata-e2e-participation-v1"] = (
    "robata-e2e-participation-v1"
)

NonEmptyString = StringConstraints(strict=True, min_length=1, max_length=512)
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class E2EParticipationBoundary(StrEnum):
    """Stable boundary order shared with :class:`E2ETraceStage`."""

    ORCHESTRATION = "ORCHESTRATION"
    SOURCE = "SOURCE"
    SCHEDULING = "SCHEDULING"
    INFERENCE = "INFERENCE"
    EVIDENCE = "EVIDENCE"
    REDUCTION = "REDUCTION"
    PUBLICATION = "PUBLICATION"

    @classmethod
    def from_trace_stage(cls, stage: E2ETraceStage) -> Self:
        if not isinstance(stage, E2ETraceStage):
            raise TypeError("stage must be E2ETraceStage")
        return cls(stage.value)

    def to_trace_stage(self) -> E2ETraceStage:
        return E2ETraceStage(self.value)


class E2EParticipationState(StrEnum):
    """Operator declaration for whether a boundary is part of this run."""

    PARTICIPATING = "PARTICIPATING"
    BYPASSED = "BYPASSED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    FAILED = "FAILED"


# Friendly aliases used by callers that call this a status/declaration.
E2EParticipationStatus = E2EParticipationState
E2EParticipationDeclarationStatus = E2EParticipationState


class E2EParticipationCoverage(StrEnum):
    """Qualification coverage, independent of model or data quality."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


E2EQualificationCoverage = E2EParticipationCoverage


class E2EParticipationIssueCode(StrEnum):
    """Deterministic reasons for incomplete or failed participation coverage."""

    DECLARED_FAILURE = "DECLARED_FAILURE"
    MEASURED_WHILE_EXCLUDED = "MEASURED_WHILE_EXCLUDED"
    PARTICIPATING_NOT_MEASURED = "PARTICIPATING_NOT_MEASURED"
    REQUIRED_BOUNDARY_UNAVAILABLE = "REQUIRED_BOUNDARY_UNAVAILABLE"
    UNCLASSIFIED_SPANS = "UNCLASSIFIED_SPANS"


class E2EParticipationDeclaration(StrictModel):
    """A declared boundary state before looking at the runtime fragment."""

    boundary: E2EParticipationBoundary
    state: E2EParticipationState
    required: bool = True
    reason: Annotated[str, NonEmptyString] | None = None

    @model_validator(mode="after")
    def validate_declaration(self) -> Self:
        if self.state is not E2EParticipationState.PARTICIPATING and self.reason is None:
            raise ValueError("non-participating declarations require a reason")
        return self


class E2EParticipationIssue(StrictModel):
    """A machine-readable sidecar finding; not a canonical validation issue."""

    code: E2EParticipationIssueCode
    boundary: E2EParticipationBoundary | None = None
    detail: Annotated[str, NonEmptyString]


class E2EParticipationBoundaryObservation(StrictModel):
    """One declared boundary reconciled with the observed trace stage."""

    boundary: E2EParticipationBoundary
    state: E2EParticipationState
    required: bool
    reason: Annotated[str, NonEmptyString] | None = None
    observed_measurement_status: E2ETraceMeasurementStatus
    observed_span_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_observation_shape(self) -> Self:
        if (
            self.observed_measurement_status is E2ETraceMeasurementStatus.MEASURED
            and self.observed_span_count < 1
        ):
            raise ValueError("MEASURED participation boundary requires observed spans")
        if (
            self.observed_measurement_status is E2ETraceMeasurementStatus.NOT_MEASURED
            and self.observed_span_count != 0
        ):
            raise ValueError("NOT_MEASURED participation boundary cannot retain a span count")
        if self.state is not E2EParticipationState.PARTICIPATING and self.reason is None:
            raise ValueError("non-participating observations require a reason")
        return self


# Short alias for integrations that refer to entries rather than observations.
E2EParticipationEntry = E2EParticipationBoundaryObservation


def _evaluate_coverage(
    boundaries: tuple[E2EParticipationBoundaryObservation, ...],
    *,
    unclassified_span_count: int,
) -> tuple[E2EParticipationCoverage, tuple[E2EParticipationIssue, ...]]:
    issues: list[E2EParticipationIssue] = []
    partial = False
    failed = False

    for item in boundaries:
        measured = item.observed_measurement_status is E2ETraceMeasurementStatus.MEASURED
        if item.state is E2EParticipationState.FAILED:
            failed = True
            issues.append(
                E2EParticipationIssue(
                    code=E2EParticipationIssueCode.DECLARED_FAILURE,
                    boundary=item.boundary,
                    detail=item.reason or "boundary declared FAILED",
                )
            )
            continue
        if measured and item.state in (
            E2EParticipationState.BYPASSED,
            E2EParticipationState.NOT_CONFIGURED,
        ):
            failed = True
            issues.append(
                E2EParticipationIssue(
                    code=E2EParticipationIssueCode.MEASURED_WHILE_EXCLUDED,
                    boundary=item.boundary,
                    detail=(
                        f"{item.boundary.value} is declared {item.state.value} but the trace "
                        "contains measured spans"
                    ),
                )
            )
            continue
        if item.state is E2EParticipationState.PARTICIPATING and not measured:
            partial = True
            issues.append(
                E2EParticipationIssue(
                    code=E2EParticipationIssueCode.PARTICIPATING_NOT_MEASURED,
                    boundary=item.boundary,
                    detail=(
                        f"{item.boundary.value} is declared PARTICIPATING but has no "
                        "classified runtime spans"
                    ),
                )
            )
            continue
        if item.required and item.state in (
            E2EParticipationState.BYPASSED,
            E2EParticipationState.NOT_CONFIGURED,
        ):
            partial = True
            issues.append(
                E2EParticipationIssue(
                    code=E2EParticipationIssueCode.REQUIRED_BOUNDARY_UNAVAILABLE,
                    boundary=item.boundary,
                    detail=(
                        f"required boundary {item.boundary.value} is declared {item.state.value}"
                    ),
                )
            )

    if unclassified_span_count:
        partial = True
        issues.append(
            E2EParticipationIssue(
                code=E2EParticipationIssueCode.UNCLASSIFIED_SPANS,
                detail=(
                    f"{unclassified_span_count} runtime span(s) are not assigned to a "
                    "declared boundary"
                ),
            )
        )

    if failed:
        coverage = E2EParticipationCoverage.FAILED
    elif partial:
        coverage = E2EParticipationCoverage.PARTIAL
    else:
        coverage = E2EParticipationCoverage.COMPLETE
    return coverage, tuple(issues)


def _derive_boundary_lists(
    boundaries: tuple[E2EParticipationBoundaryObservation, ...],
) -> tuple[
    tuple[E2EParticipationBoundary, ...],
    tuple[E2EParticipationBoundary, ...],
]:
    """Derive machine-readable inclusion summaries from reconciled entries."""

    excluded = tuple(
        item.boundary
        for item in boundaries
        if item.state in (E2EParticipationState.BYPASSED, E2EParticipationState.NOT_CONFIGURED)
        and item.observed_measurement_status is E2ETraceMeasurementStatus.NOT_MEASURED
    )
    measured = tuple(
        item.boundary
        for item in boundaries
        if item.observed_measurement_status is E2ETraceMeasurementStatus.MEASURED
    )
    return excluded, measured


class E2EParticipationManifest(StrictModel):
    """Run-level, non-canonical participation and coverage sidecar."""

    format_version: Literal["robata-e2e-participation-v1"] = PARTICIPATION_MANIFEST_VERSION
    trace_id: OpaqueUuid
    trace_role: E2ETraceFragmentRole
    observed_at: Rfc3339Timestamp
    runtime_fragment_sha256: Sha256Digest
    boundaries: tuple[E2EParticipationBoundaryObservation, ...]
    # These lists are persisted facts, not caller-controlled annotations.  The
    # validator below derives them from ``boundaries`` and rejects any payload
    # whose machine-readable summary was altered after reconciliation.
    excluded_boundaries: tuple[E2EParticipationBoundary, ...]
    measured_boundaries: tuple[E2EParticipationBoundary, ...]
    unclassified_span_count: NonNegativeInt
    coverage: E2EParticipationCoverage
    issues: tuple[E2EParticipationIssue, ...] = ()

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected = tuple(E2EParticipationBoundary)
        observed = tuple(item.boundary for item in self.boundaries)
        if observed != expected:
            raise ValueError("participation boundaries must use the stable stage order")
        computed_coverage, computed_issues = _evaluate_coverage(
            self.boundaries,
            unclassified_span_count=self.unclassified_span_count,
        )
        if self.coverage is not computed_coverage:
            raise ValueError(
                "participation coverage does not match declarations and runtime observations"
            )
        if self.issues != computed_issues:
            raise ValueError("participation issues do not match declarations and observations")
        computed_excluded, computed_measured = _derive_boundary_lists(self.boundaries)
        if self.excluded_boundaries != computed_excluded:
            raise ValueError("excluded boundaries do not match declarations and observations")
        if self.measured_boundaries != computed_measured:
            raise ValueError("measured boundaries do not match declarations and observations")
        return self

    @property
    def trace_fragment_sha256(self) -> Sha256Digest:
        """Compatibility alias for integrations that call the binding a trace digest."""

        return self.runtime_fragment_sha256

    @property
    def trace_digest(self) -> Sha256Digest:
        """Short alias for the digest of the bound runtime fragment."""

        return self.runtime_fragment_sha256


def _runtime_fragment_bytes(fragment: E2ETraceRuntimeFragment) -> bytes:
    return canonical_json_bytes(fragment.model_dump(mode="json")) + b"\n"


def runtime_fragment_digest(fragment: E2ETraceRuntimeFragment) -> Sha256Digest:
    """Return the digest bound by a manifest for a frozen runtime fragment."""

    if not isinstance(fragment, E2ETraceRuntimeFragment):
        raise TypeError("fragment must be E2ETraceRuntimeFragment")
    return exact_bytes_sha256(_runtime_fragment_bytes(fragment))


def _coerce_boundary(value: object) -> E2EParticipationBoundary:
    if isinstance(value, E2EParticipationBoundary):
        return value
    if isinstance(value, E2ETraceStage):
        return E2EParticipationBoundary(value.value)
    if isinstance(value, str):
        return E2EParticipationBoundary(value)
    raise TypeError("boundary keys must be E2EParticipationBoundary, E2ETraceStage, or str")


def _coerce_state(value: object) -> E2EParticipationState:
    if isinstance(value, E2EParticipationState):
        return value
    if isinstance(value, str):
        return E2EParticipationState(value)
    raise TypeError("participation states must be E2EParticipationState or str")


def _normalize_declarations(
    declarations: Mapping[object, object] | Iterable[E2EParticipationDeclaration],
    *,
    required_boundaries: Iterable[object] | None,
) -> tuple[E2EParticipationDeclaration, ...]:
    required = (
        {_coerce_boundary(value) for value in required_boundaries}
        if required_boundaries is not None
        else set(E2EParticipationBoundary)
    )
    if isinstance(declarations, Mapping):
        normalized: list[E2EParticipationDeclaration] = []
        for key, value in declarations.items():
            boundary = _coerce_boundary(key)
            if isinstance(value, E2EParticipationDeclaration):
                if value.boundary is not boundary:
                    raise ValueError("declaration key and boundary do not match")
                normalized.append(value)
                continue
            state = _coerce_state(value)
            reason = None
            if state is not E2EParticipationState.PARTICIPATING:
                reason = f"declared {state.value} by caller"
            normalized.append(
                E2EParticipationDeclaration(
                    boundary=boundary,
                    state=state,
                    required=boundary in required,
                    reason=reason,
                )
            )
        declarations_tuple = tuple(normalized)
    else:
        declarations_tuple = tuple(declarations)
        if not all(isinstance(item, E2EParticipationDeclaration) for item in declarations_tuple):
            raise TypeError("declarations iterable must contain E2EParticipationDeclaration values")
    expected = tuple(E2EParticipationBoundary)
    if tuple(item.boundary for item in declarations_tuple) != expected:
        raise ValueError("declarations must use the stable stage order and cover every boundary")
    return declarations_tuple


def build_e2e_participation_manifest(
    *,
    runtime_fragment: E2ETraceRuntimeFragment,
    declarations: Mapping[object, object] | Iterable[E2EParticipationDeclaration],
    trace_id: str | None = None,
    observed_at: str | None = None,
    runtime_fragment_sha256: Sha256Digest | None = None,
    trace_digest: Sha256Digest | None = None,
    required_boundaries: Iterable[object] | None = None,
) -> E2EParticipationManifest:
    """Reconcile declarations with one immutable runtime fragment.

    ``runtime_fragment_sha256`` may be supplied when the fragment is already
    retained as part of a larger trace sidecar.  When omitted it is computed
    from the canonical JSON representation of ``runtime_fragment`` plus its
    terminating newline, matching the bytes used by the sidecar writer.
    """

    if not isinstance(runtime_fragment, E2ETraceRuntimeFragment):
        raise TypeError("runtime_fragment must be E2ETraceRuntimeFragment")
    declarations_tuple = _normalize_declarations(
        declarations,
        required_boundaries=required_boundaries,
    )
    expected_stages = tuple(runtime_fragment.stages)
    if tuple(item.stage.value for item in expected_stages) != tuple(
        item.value for item in E2ETraceStage
    ):
        raise ValueError("runtime fragment stages must use the stable stage order")
    observations = tuple(
        E2EParticipationBoundaryObservation(
            boundary=declaration.boundary,
            state=declaration.state,
            required=declaration.required,
            reason=declaration.reason,
            observed_measurement_status=stage.measurement_status,
            observed_span_count=stage.observed_span_count,
        )
        for declaration, stage in zip(declarations_tuple, expected_stages, strict=True)
    )
    coverage, issues = _evaluate_coverage(
        observations,
        unclassified_span_count=runtime_fragment.unclassified_span_count,
    )
    if (
        runtime_fragment_sha256 is not None
        and trace_digest is not None
        and runtime_fragment_sha256 != trace_digest
    ):
        raise ValueError("runtime fragment and trace digest bindings do not match")
    supplied_digest = (
        runtime_fragment_sha256 if runtime_fragment_sha256 is not None else trace_digest
    )
    fragment_digest = (
        runtime_fragment_digest(runtime_fragment) if supplied_digest is None else supplied_digest
    )
    if (
        not isinstance(fragment_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", fragment_digest) is None
    ):
        raise ValueError("runtime_fragment_sha256 must be a lowercase SHA-256 digest")
    excluded_boundaries, measured_boundaries = _derive_boundary_lists(observations)
    return E2EParticipationManifest(
        trace_id=str(uuid4()) if trace_id is None else trace_id,
        trace_role=runtime_fragment.role,
        observed_at=_utc_now() if observed_at is None else observed_at,
        runtime_fragment_sha256=fragment_digest,
        boundaries=observations,
        excluded_boundaries=excluded_boundaries,
        measured_boundaries=measured_boundaries,
        unclassified_span_count=runtime_fragment.unclassified_span_count,
        coverage=coverage,
        issues=issues,
    )


def validate_e2e_participation_manifest_against_fragment(
    manifest: E2EParticipationManifest,
    runtime_fragment: E2ETraceRuntimeFragment,
) -> None:
    """Raise when a manifest no longer binds to the fragment it describes."""

    if not isinstance(manifest, E2EParticipationManifest):
        raise TypeError("manifest must be E2EParticipationManifest")
    if not isinstance(runtime_fragment, E2ETraceRuntimeFragment):
        raise TypeError("runtime_fragment must be E2ETraceRuntimeFragment")
    if manifest.trace_role is not runtime_fragment.role:
        raise ValueError("participation manifest role does not match runtime fragment")
    if manifest.runtime_fragment_sha256 != runtime_fragment_digest(runtime_fragment):
        raise ValueError("participation manifest runtime fragment digest does not match")
    for entry, stage in zip(manifest.boundaries, runtime_fragment.stages, strict=True):
        if entry.boundary.value != stage.stage.value:
            raise ValueError("participation boundary order does not match runtime fragment")
        if entry.observed_measurement_status is not stage.measurement_status:
            raise ValueError("participation measurement status does not match runtime fragment")
        if entry.observed_span_count != stage.observed_span_count:
            raise ValueError("participation span count does not match runtime fragment")
    if manifest.unclassified_span_count != runtime_fragment.unclassified_span_count:
        raise ValueError("participation unclassified span count does not match runtime fragment")


# Alias with a shorter name for worker integrations.
verify_e2e_participation_manifest = validate_e2e_participation_manifest_against_fragment


def serialize_e2e_participation_manifest(manifest: E2EParticipationManifest) -> bytes:
    """Serialize a manifest as canonical JSON sidecar bytes."""

    if not isinstance(manifest, E2EParticipationManifest):
        raise TypeError("manifest must be E2EParticipationManifest")
    return canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"


def e2e_participation_manifest_digest(manifest: E2EParticipationManifest) -> Sha256Digest:
    """Return the exact SHA-256 digest of the serialized sidecar."""

    return exact_bytes_sha256(serialize_e2e_participation_manifest(manifest))


def write_e2e_participation_manifest(
    manifest: E2EParticipationManifest,
    output_path: Path,
) -> Sha256Digest:
    """Atomically write a sidecar and return the digest bound to its bytes."""

    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    payload = serialize_e2e_participation_manifest(manifest)
    digest = exact_bytes_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except FileExistsError as error:
        raise RuntimeError(
            f"temporary participation output path is already occupied: {temporary}"
        ) from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return digest


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "PARTICIPATION_MANIFEST_VERSION",
    "E2EParticipationBoundary",
    "E2EParticipationBoundaryObservation",
    "E2EParticipationCoverage",
    "E2EParticipationDeclaration",
    "E2EParticipationDeclarationStatus",
    "E2EParticipationEntry",
    "E2EParticipationIssue",
    "E2EParticipationIssueCode",
    "E2EParticipationManifest",
    "E2EParticipationState",
    "E2EParticipationStatus",
    "E2EQualificationCoverage",
    "build_e2e_participation_manifest",
    "e2e_participation_manifest_digest",
    "runtime_fragment_digest",
    "serialize_e2e_participation_manifest",
    "validate_e2e_participation_manifest_against_fragment",
    "verify_e2e_participation_manifest",
    "write_e2e_participation_manifest",
]
