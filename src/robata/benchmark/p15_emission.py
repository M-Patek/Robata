"""Local-only P15 qualification package assembly and atomic emission.

This module turns a frozen local P0 scope register plus an explicit manifest of
real local files into a content-addressed P15 package.  It deliberately cannot
accept measured external gates or production-qualified evidence.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Final, Literal, Self
from urllib.parse import unquote, urlsplit

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.benchmark.p15_qualification import (
    P15ExternalGateEvidence,
    P15ExternalGateId,
    P15ExternalGateStatus,
    P15ParetoSelection,
    P15PhaseArtifactReference,
    P15QualificationPackage,
    P15TradeoffAxis,
    P15UnresolvedRisk,
)
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.measurement_truth import (
    EvidenceClass,
    MeasurementStatus,
    ScopeEvidenceRegister,
)
from robata.contracts.phase_contract_decisions import OptimizationPhase

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
P15_LOCAL_QUALIFICATION_MANIFEST_VERSION: Final = "p15-local-qualification-manifest-v1"
_REQUIRED_PHASES: Final = tuple(
    phase for phase in OptimizationPhase if phase is not OptimizationPhase.P15
)
_LOCAL_GATE_REASONS: Final[dict[P15ExternalGateId, str]] = {
    P15ExternalGateId.E0: "External evidence freeze has not been measured from this local run.",
    P15ExternalGateId.E1: "Governed quality labels and sign-off have not been measured locally.",
    P15ExternalGateId.E2: (
        "Target media, storage, and fault evidence has not been measured locally."
    ),
    P15ExternalGateId.E3: (
        "Real provider topology and saturation evidence has not been measured locally."
    ),
    P15ExternalGateId.E4: (
        "Representative reliability and soak evidence has not been measured locally."
    ),
    P15ExternalGateId.E5: (
        "Representative capacity, deadline, and cost evidence has not been measured locally."
    ),
    P15ExternalGateId.E6: "Independent release review is pending.",
}


class P15LocalPhaseArtifactManifestEntry(StrictModel):
    """One real local file included in a P15 package."""

    phase: OptimizationPhase
    artifact_id: NonEmptyString
    artifact_path: NonEmptyString
    scope_digest: Sha256Digest
    evidence_class: EvidenceClass = EvidenceClass.LOCAL_CONFORMANCE
    measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    summary: NonEmptyString
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_local_artifact(self) -> Self:
        if self.phase is OptimizationPhase.P15:
            raise ValueError("P15 local artifact manifest may only reference P1 through P14")
        if self.evidence_class is not EvidenceClass.LOCAL_CONFORMANCE:
            raise ValueError("P15 local artifact evidence must be LOCAL_CONFORMANCE")
        if self.measurement_status is not MeasurementStatus.NOT_MEASURED:
            raise ValueError("P15 local artifact evidence must be NOT_MEASURED")
        return self


class P15LocalParetoSelectionManifest(StrictModel):
    """Local-only Pareto selection fields supplied by the explicit manifest."""

    source_artifact_id: NonEmptyString
    candidate_operating_point_ids: tuple[NonEmptyString, ...] = Field(min_length=2)
    selected_operating_point_id: NonEmptyString
    preference_rationale: NonEmptyString

    @field_validator("candidate_operating_point_ids")
    @classmethod
    def validate_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("Pareto candidate operating point IDs must be unique and ordered")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.selected_operating_point_id not in self.candidate_operating_point_ids:
            raise ValueError("selected operating point must belong to the Pareto candidate set")
        return self


class P15LocalQualificationManifest(StrictModel):
    """Explicit local input manifest for one P15 package emission."""

    manifest_version: Literal["p15-local-qualification-manifest-v1"] = (
        P15_LOCAL_QUALIFICATION_MANIFEST_VERSION
    )
    scope_digest: Sha256Digest
    phase_artifacts: tuple[P15LocalPhaseArtifactManifestEntry, ...] = Field(min_length=14)
    pareto_selection: P15LocalParetoSelectionManifest

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        phase_order = tuple(OptimizationPhase)
        artifact_keys = tuple((item.phase, item.artifact_id) for item in self.phase_artifacts)
        if artifact_keys != tuple(
            sorted(artifact_keys, key=lambda item: (phase_order.index(item[0]), item[1]))
        ):
            raise ValueError("phase artifacts must be canonically ordered")
        artifact_ids = tuple(item.artifact_id for item in self.phase_artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("phase artifact IDs must be unique")
        if {item.phase for item in self.phase_artifacts} != set(_REQUIRED_PHASES):
            raise ValueError("P15 local manifest must cover P1 through P14")
        if any(item.scope_digest != self.scope_digest for item in self.phase_artifacts):
            raise ValueError("every local artifact must bind the manifest scope digest")
        if self.pareto_selection.source_artifact_id not in set(artifact_ids):
            raise ValueError("Pareto selection source artifact is absent from the manifest")
        return self


def load_local_scope_evidence_register(path: Path) -> ScopeEvidenceRegister:
    """Load and validate an actual P0 scope evidence register JSON file."""

    source = _require_regular_file(path, description="scope evidence register")
    try:
        register = ScopeEvidenceRegister.model_validate_json(source.read_bytes(), strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot read scope evidence register JSON: {error}") from error
    _require_local_scope(register)
    return register


def load_local_p15_qualification_manifest(path: Path) -> P15LocalQualificationManifest:
    """Load and validate an explicit local P15 artifact manifest JSON file."""

    source = _require_regular_file(path, description="P15 local qualification manifest")
    try:
        return P15LocalQualificationManifest.model_validate_json(source.read_bytes(), strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot read P15 local qualification manifest JSON: {error}") from error


def build_local_p15_qualification_package(
    *,
    scope_evidence_register: ScopeEvidenceRegister,
    manifest: P15LocalQualificationManifest,
    manifest_directory: Path,
) -> P15QualificationPackage:
    """Build a local-only P15 package from a verified scope and manifest files."""

    if not isinstance(scope_evidence_register, ScopeEvidenceRegister):
        raise TypeError("scope_evidence_register must be a ScopeEvidenceRegister")
    if not isinstance(manifest, P15LocalQualificationManifest):
        raise TypeError("manifest must be a P15LocalQualificationManifest")
    if not isinstance(manifest_directory, Path):
        raise TypeError("manifest_directory must be a pathlib.Path")

    _require_local_scope(scope_evidence_register)
    resolved_manifest_directory = manifest_directory.expanduser().resolve()
    if not resolved_manifest_directory.is_dir():
        raise ValueError("manifest_directory must be an existing directory")
    scope_digest = scope_evidence_register.scope.scope_digest
    if manifest.scope_digest != scope_digest:
        raise ValueError("manifest scope digest does not match the scope evidence register")

    phase_artifacts = tuple(
        _build_phase_artifact_reference(
            entry,
            manifest_directory=resolved_manifest_directory,
            scope_digest=scope_digest,
        )
        for entry in manifest.phase_artifacts
    )
    package = P15QualificationPackage.create(
        scope_evidence_register=scope_evidence_register,
        phase_artifacts=phase_artifacts,
        pareto_selection=P15ParetoSelection(
            source_artifact_id=manifest.pareto_selection.source_artifact_id,
            candidate_operating_point_ids=manifest.pareto_selection.candidate_operating_point_ids,
            selected_operating_point_id=manifest.pareto_selection.selected_operating_point_id,
            tradeoff_axes=tuple(P15TradeoffAxis),
            preference_rationale=manifest.pareto_selection.preference_rationale,
            measurement_status=MeasurementStatus.NOT_MEASURED,
        ),
        external_gates=_local_external_gates(),
        unresolved_risks=_local_unresolved_risks(),
    )
    verify_local_p15_qualification_package_artifacts(package)
    return package


def assemble_local_p15_qualification_package(
    *,
    scope_path: Path,
    manifest_path: Path,
) -> P15QualificationPackage:
    """Read scope and manifest files, then assemble one local-only P15 package."""

    resolved_manifest_path = _require_regular_file(
        manifest_path,
        description="P15 local qualification manifest",
    )
    return build_local_p15_qualification_package(
        scope_evidence_register=load_local_scope_evidence_register(scope_path),
        manifest=load_local_p15_qualification_manifest(resolved_manifest_path),
        manifest_directory=resolved_manifest_path.parent,
    )


def emit_local_p15_qualification_package(
    *,
    scope_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> P15QualificationPackage:
    """Assemble, atomically write, reload, and verify a local-only P15 package."""

    scope_source = _require_regular_file(scope_path, description="scope evidence register")
    manifest_source = _require_regular_file(
        manifest_path,
        description="P15 local qualification manifest",
    )
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    destination = output_path.expanduser().resolve()
    package = assemble_local_p15_qualification_package(
        scope_path=scope_source,
        manifest_path=manifest_source,
    )
    _reject_input_overwrite(
        destination,
        scope_source=scope_source,
        manifest_source=manifest_source,
        package=package,
    )
    _atomic_write(destination, canonical_json_bytes(package.as_dict()) + b"\n")
    try:
        reloaded = P15QualificationPackage.model_validate_json(
            destination.read_bytes(), strict=True
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot reload emitted P15 qualification package: {error}") from error
    if reloaded != package:
        raise ValueError("emitted P15 qualification package does not round-trip")
    verify_local_p15_qualification_package_artifacts(reloaded)
    return reloaded


def verify_local_p15_qualification_package_artifacts(package: P15QualificationPackage) -> None:
    """Verify local file references still match the exact hashes retained by a package."""

    if not isinstance(package, P15QualificationPackage):
        raise TypeError("package must be a P15QualificationPackage")
    P15QualificationPackage.model_validate(package.model_dump(mode="python"), strict=True)
    _require_local_scope(package.scope_evidence_register)
    if package.pareto_selection.measurement_status is not MeasurementStatus.NOT_MEASURED:
        raise ValueError("local P15 Pareto selection must remain NOT_MEASURED")
    _verify_local_gates_and_risks(package)
    for artifact in package.phase_artifacts:
        if artifact.evidence_class is not EvidenceClass.LOCAL_CONFORMANCE:
            raise ValueError("local P15 artifact reference must be LOCAL_CONFORMANCE")
        if artifact.measurement_status is not MeasurementStatus.NOT_MEASURED:
            raise ValueError("local P15 artifact reference must be NOT_MEASURED")
        artifact_path = _path_from_file_uri(artifact.artifact_uri)
        try:
            observed_sha256 = exact_bytes_sha256(artifact_path.read_bytes())
        except OSError as error:
            raise ValueError(
                f"cannot read referenced local artifact {artifact_path}: {error}"
            ) from error
        if observed_sha256 != artifact.artifact_sha256:
            raise ValueError(
                f"referenced local artifact SHA-256 changed for {artifact.artifact_id}"
            )


def _build_phase_artifact_reference(
    entry: P15LocalPhaseArtifactManifestEntry,
    *,
    manifest_directory: Path,
    scope_digest: Sha256Digest,
) -> P15PhaseArtifactReference:
    if entry.scope_digest != scope_digest:
        raise ValueError(
            f"{entry.artifact_id} scope digest does not match the scope evidence register"
        )
    if entry.evidence_class is not EvidenceClass.LOCAL_CONFORMANCE:
        raise ValueError(f"{entry.artifact_id} must be LOCAL_CONFORMANCE")
    if entry.measurement_status is not MeasurementStatus.NOT_MEASURED:
        raise ValueError(f"{entry.artifact_id} must be NOT_MEASURED")
    path = Path(entry.artifact_path)
    if not path.is_absolute():
        path = manifest_directory / path
    artifact_path = _require_regular_file(path, description=f"artifact {entry.artifact_id}")
    try:
        artifact_sha256 = exact_bytes_sha256(artifact_path.read_bytes())
    except OSError as error:
        raise ValueError(f"cannot read artifact {entry.artifact_id}: {error}") from error
    return P15PhaseArtifactReference(
        phase=entry.phase,
        artifact_id=entry.artifact_id,
        artifact_uri=artifact_path.as_uri(),
        artifact_sha256=artifact_sha256,
        scope_digest=scope_digest,
        evidence_class=EvidenceClass.LOCAL_CONFORMANCE,
        measurement_status=MeasurementStatus.NOT_MEASURED,
        summary=entry.summary,
    )


def _local_external_gates() -> tuple[P15ExternalGateEvidence, ...]:
    return tuple(
        P15ExternalGateEvidence(
            gate_id=gate_id,
            status=(
                P15ExternalGateStatus.PENDING_INDEPENDENT_REVIEW
                if gate_id is P15ExternalGateId.E6
                else P15ExternalGateStatus.NOT_MEASURED
            ),
            unresolved_reason=_LOCAL_GATE_REASONS[gate_id],
        )
        for gate_id in P15ExternalGateId
    )


def _local_unresolved_risks() -> tuple[P15UnresolvedRisk, ...]:
    return tuple(
        P15UnresolvedRisk(
            risk_id=f"unresolved-{gate_id.value.lower()}",
            gate_id=gate_id,
            description=_LOCAL_GATE_REASONS[gate_id],
            required_follow_up=(
                f"Run and retain the declared {gate_id.value} qualification evidence."
            ),
        )
        for gate_id in P15ExternalGateId
    )


def _require_local_scope(scope_evidence_register: ScopeEvidenceRegister) -> None:
    if scope_evidence_register.evidence_class is not EvidenceClass.LOCAL_CONFORMANCE:
        raise ValueError("P15 local emission requires LOCAL_CONFORMANCE scope evidence")
    if scope_evidence_register.measurement_status is not MeasurementStatus.NOT_MEASURED:
        raise ValueError("P15 local emission requires NOT_MEASURED scope evidence")


def _verify_local_gates_and_risks(package: P15QualificationPackage) -> None:
    expected_gates = _local_external_gates()
    if package.external_gates != expected_gates:
        raise ValueError("local P15 package must retain only unresolved E0-E5 and pending E6")
    expected_risks = _local_unresolved_risks()
    if package.unresolved_risks != expected_risks:
        raise ValueError(
            "local P15 package must retain one unresolved risk for every external gate"
        )


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("local P15 artifact URI must be a local file URI")
    if parsed.query or parsed.fragment:
        raise ValueError("local P15 artifact URI cannot include a query or fragment")
    path_text = unquote(parsed.path)
    if os.name == "nt" and len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
        path_text = path_text[1:]
    return _require_regular_file(Path(path_text), description="referenced local artifact")


def _reject_input_overwrite(
    destination: Path,
    *,
    scope_source: Path,
    manifest_source: Path,
    package: P15QualificationPackage,
) -> None:
    referenced_paths = tuple(
        _path_from_file_uri(artifact.artifact_uri) for artifact in package.phase_artifacts
    )
    if destination in (scope_source, manifest_source) or destination in referenced_paths:
        raise ValueError(
            "P15 output path cannot overwrite a scope, manifest, or referenced artifact"
        )
    if destination.exists() and destination.is_dir():
        raise ValueError("P15 output path must not be a directory")


def _require_regular_file(path: Path, *, description: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{description} path must be a pathlib.Path")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{description} path cannot be resolved: {error}") from error
    if not resolved.is_file():
        raise ValueError(f"{description} path must name an existing regular file")
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except OSError:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "P15_LOCAL_QUALIFICATION_MANIFEST_VERSION",
    "P15LocalParetoSelectionManifest",
    "P15LocalPhaseArtifactManifestEntry",
    "P15LocalQualificationManifest",
    "assemble_local_p15_qualification_package",
    "build_local_p15_qualification_package",
    "emit_local_p15_qualification_package",
    "load_local_p15_qualification_manifest",
    "load_local_scope_evidence_register",
    "verify_local_p15_qualification_package_artifacts",
]
