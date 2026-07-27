"""Internal scope and evidence contracts for reproducible measurements.

These models describe qualification metadata rather than a product wire payload.  They
bind a measurement to the exact code, catalog, workload, policy, and identity inputs
that produced it while keeping historical/local observations out of production status.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import Rfc3339Timestamp

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]

SCOPE_FINGERPRINT_VERSION: Final[Literal['scope-fingerprint-v1']] = 'scope-fingerprint-v1'
SCOPE_EVIDENCE_REGISTER_VERSION: Final[Literal['scope-evidence-register-v1']] = (
    'scope-evidence-register-v1'
)
IDENTITY_FORMULA_VERSION: Final[Literal['recording-identity-v1']] = 'recording-identity-v1'

class EvidenceClass(StrEnum):
    """Evidence boundary used by every profile or qualification artifact."""

    LOCAL_CONFORMANCE = "LOCAL_CONFORMANCE"
    LOCAL_BENCHMARK = "LOCAL_BENCHMARK"
    REPRESENTATIVE_BENCHMARK = "REPRESENTATIVE_BENCHMARK"
    EXTERNAL_QUALIFICATION = "EXTERNAL_QUALIFICATION"
    PRODUCTION_QUALIFIED = "PRODUCTION_QUALIFIED"


# Descriptive aliases keep call sites readable without creating multiple enums.
MeasurementEvidenceClass = EvidenceClass
QualificationEvidenceClass = EvidenceClass


class MeasurementExecutionMode(StrEnum):
    FRESH = "FRESH"
    REPLAY = "REPLAY"
    UNKNOWN = "UNKNOWN"


class MeasurementStatus(StrEnum):
    MEASURED = "MEASURED"
    NOT_MEASURED = "NOT_MEASURED"


class ScopeDigestInputs(StrictModel):
    """The complete, path-independent preimage for one scope fingerprint."""

    code_revision: NonEmptyString
    code_digest: Sha256Digest
    schema_catalog_digest: Sha256Digest
    workload_digest: Sha256Digest
    policy_digest: Sha256Digest
    identity_formula_version: SchemaVersion
    identity_projection_digest: Sha256Digest
    # Policy seam names are descriptive identifiers and may include a separator such as
    # `task:policy-version`.  They are intentionally broader than wire schema versions.
    seam_versions: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_seams(self) -> Self:
        if tuple(sorted(set(self.seam_versions))) != self.seam_versions:
            raise ValueError("seam_versions must be unique and ordered")
        return self


def scope_fingerprint_projection(inputs: ScopeDigestInputs) -> dict[str, object]:
    """Return the versioned semantic projection used for scope identity."""

    if not isinstance(inputs, ScopeDigestInputs):
        raise TypeError("inputs must be ScopeDigestInputs")
    return {
        "domain": "robata.measurement-scope",
        "projection_version": SCOPE_FINGERPRINT_VERSION,
        **inputs.model_dump(mode="json"),
    }


class ScopeFingerprint(StrictModel):
    """Content-addressed scope identity independent of local paths and timestamps."""

    schema_version: Literal["1.0"]
    projection_version: Literal["scope-fingerprint-v1"]
    scope_digest: Sha256Digest
    inputs: ScopeDigestInputs

    @classmethod
    def create(cls, *, inputs: ScopeDigestInputs) -> Self:
        digest = semantic_sha256(scope_fingerprint_projection(inputs))
        return cls(
            schema_version="1.0",
            projection_version=SCOPE_FINGERPRINT_VERSION,
            scope_digest=digest,
            inputs=inputs,
        )

    @property
    def fingerprint(self) -> str:
        """Compatibility name for callers that call the digest a fingerprint."""

        return self.scope_digest

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        expected = semantic_sha256(scope_fingerprint_projection(self.inputs))
        if self.scope_digest != expected:
            raise ValueError("scope_digest does not match its inputs")
        return self


class MeasurementWorkload(StrictModel):
    """Unitized workload facts shared by fresh and replay observations."""

    workload_fingerprint: Sha256Digest
    recording_count: PositiveInt
    camera_count: PositiveInt
    recording_duration_ns: NonNegativeInt | None = None
    frame_count: NonNegativeInt | None = None
    source_bytes: NonNegativeInt | None = None


class MeasurementEnvironment(StrictModel):
    """Provider and hardware labels; labels never imply qualification."""

    provider: NonEmptyString
    provider_mode: NonEmptyString
    hardware: NonEmptyString
    accelerator: NonEmptyString | None = None


class MeasurementAxes(StrictModel):
    """Observed counters with null meaning not instrumented, never zero."""

    recording_hours: float | None = None
    camera_hours: float | None = None
    decoded_frames: NonNegativeInt | None = None
    selected_images: NonNegativeInt | None = None
    provider_images: NonNegativeInt | None = None
    provider_calls: NonNegativeInt | None = None
    http_requests: NonNegativeInt | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    process_cpu_ns: NonNegativeInt | None = None
    gpu_time_ns: NonNegativeInt | None = None
    nvme_read_bytes: NonNegativeInt | None = None
    nvme_write_bytes: NonNegativeInt | None = None
    queue_backlog: NonNegativeInt | None = None
    terminal_count: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_finite_rates(self) -> Self:
        for name in ("recording_hours", "camera_hours"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        return self


def _default_observed_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def scope_evidence_register_projection(register: ScopeEvidenceRegister) -> dict[str, object]:
    """Return the stable register identity projection, excluding observation time."""

    if not isinstance(register, ScopeEvidenceRegister):
        raise TypeError("register must be ScopeEvidenceRegister")
    return register.model_dump(
        mode="json",
        exclude={"register_digest", "observed_at"},
    )


class ScopeEvidenceRegister(StrictModel):
    """Historical evidence record that cannot self-promote to production."""

    schema_version: Literal["1.0"]
    register_version: Literal["scope-evidence-register-v1"]
    register_digest: Sha256Digest
    scope: ScopeFingerprint
    evidence_class: EvidenceClass
    execution_mode: MeasurementExecutionMode
    workload: MeasurementWorkload
    environment: MeasurementEnvironment
    axes: MeasurementAxes
    observed_at: Rfc3339Timestamp
    measurement_status: MeasurementStatus
    production_eligible: Literal[False] = False
    historical_snapshot: Literal[True] = True
    profile_manifest_digest: Sha256Digest | None = None
    profile_report_digest: Sha256Digest | None = None

    @classmethod
    def create(
        cls,
        *,
        scope: ScopeFingerprint,
        evidence_class: EvidenceClass,
        execution_mode: MeasurementExecutionMode,
        workload: MeasurementWorkload,
        environment: MeasurementEnvironment,
        axes: MeasurementAxes,
        observed_at: str | None = None,
        measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED,
        profile_manifest_digest: Sha256Digest | None = None,
        profile_report_digest: Sha256Digest | None = None,
    ) -> Self:
        observed_at_value = (
            _default_observed_at() if observed_at is None else observed_at
        )
        values: dict[str, object] = {
            "schema_version": "1.0",
            "register_version": SCOPE_EVIDENCE_REGISTER_VERSION,
            "register_digest": "0" * 64,
            "scope": scope,
            "evidence_class": evidence_class,
            "execution_mode": execution_mode,
            "workload": workload,
            "environment": environment,
            "axes": axes,
            "observed_at": observed_at_value,
            "measurement_status": measurement_status,
            "production_eligible": False,
            "historical_snapshot": True,
            "profile_manifest_digest": profile_manifest_digest,
            "profile_report_digest": profile_report_digest,
        }
        draft = cls.model_construct(
            schema_version='1.0',
            register_version=SCOPE_EVIDENCE_REGISTER_VERSION,
            register_digest='0' * 64,
            scope=scope,
            evidence_class=evidence_class,
            execution_mode=execution_mode,
            workload=workload,
            environment=environment,
            axes=axes,
            observed_at=observed_at_value,
            measurement_status=measurement_status,
            production_eligible=False,
            historical_snapshot=True,
            profile_manifest_digest=profile_manifest_digest,
            profile_report_digest=profile_report_digest,
        )
        digest = semantic_sha256(scope_evidence_register_projection(draft))
        return cls.model_validate({**values, "register_digest": digest}, strict=True)

    @model_validator(mode="after")
    def validate_register(self) -> Self:
        expected = semantic_sha256(scope_evidence_register_projection(self))
        if self.register_digest != expected:
            raise ValueError("register_digest does not match the evidence register")
        if self.production_eligible:
            raise ValueError('scope evidence cannot grant production eligibility')
        if self.scope.inputs.workload_digest != self.workload.workload_fingerprint:
            raise ValueError('scope workload digest must match workload fingerprint')
        if (
            self.evidence_class is EvidenceClass.LOCAL_CONFORMANCE
            and self.measurement_status is not MeasurementStatus.NOT_MEASURED
        ):
            raise ValueError("local conformance evidence must remain NOT_MEASURED")
        return self


# Short alias used by qualification callers.
EvidenceRegister = ScopeEvidenceRegister


__all__ = [
    "IDENTITY_FORMULA_VERSION",
    "SCOPE_EVIDENCE_REGISTER_VERSION",
    "SCOPE_FINGERPRINT_VERSION",
    "EvidenceClass",
    "EvidenceRegister",
    "MeasurementAxes",
    "MeasurementEnvironment",
    "MeasurementEvidenceClass",
    "MeasurementExecutionMode",
    "MeasurementStatus",
    "MeasurementWorkload",
    "QualificationEvidenceClass",
    "ScopeDigestInputs",
    "ScopeEvidenceRegister",
    "ScopeFingerprint",
    "scope_evidence_register_projection",
    "scope_fingerprint_projection",
]
