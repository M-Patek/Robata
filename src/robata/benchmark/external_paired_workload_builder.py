"""Fail-closed builder for an external paired observation workload.

The external paired launcher deliberately accepts only a complete
``robata-external-paired-workload-v1`` manifest.  This module is the narrow,
non-production bridge from a frozen local real-model E2E report to that
manifest.  It never invents a policy, prompt, capability snapshot, endpoint,
or deployment binding: those values must be supplied in each explicit target
configuration.  The local report and six persisted camera artifacts are only
used as immutable source evidence and to verify that both target input plans
refer to exactly the same bytes.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import unquote, urlparse

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.benchmark.external_paired_qualification import (
    ExternalPairedQualificationError,
    ExternalPairedWorkloadManifest,
    ExternalPairedWorkloadTarget,
)
from robata.benchmark.local_real_model_e2e import LocalRealModelE2EReport
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.inference.adapter import PackageInput
from robata.inference.input_plan import InferenceInputPlan, TransformOperation
from robata.inference.models import InputMode
from robata.inference.orchestrator import InferencePolicy
from robata.inference.routing import ExperimentInputRepresentation, ExperimentIsolationProfile
from robata.runtime.e2e_participation import (
    E2EParticipationCoverage,
    E2EParticipationManifest,
    validate_e2e_participation_manifest_against_fragment,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=5)]

TARGET_CONFIG_VERSION: Literal["robata-external-paired-target-v1"] = (
    "robata-external-paired-target-v1"
)


class ExternalPairedWorkloadBuilderError(ExternalPairedQualificationError):
    """Raised when source evidence or explicit target configuration is unsafe."""


class ExternalPairedWorkloadSourceConfig(StrictModel):
    """Explicit common workload fields repeated in both target config files.

    Repeating this small binding in both files is intentional: the builder
    rejects any mismatch before writing a paired manifest.  This keeps the
    command independent of a mutable sidecar or an implicit route default.
    """

    experiment_id: NonEmptyString
    contract_version: SchemaVersion
    route_id: NonEmptyString
    route_policy_version: SchemaVersion
    arrival_schedule_sha256: Sha256Digest
    comparison_config: dict[str, object]
    input_representation: ExperimentInputRepresentation
    isolation_profile: ExperimentIsolationProfile
    package_set_id: OpaqueUuid | None = None
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    input_config: dict[str, object] = Field(default_factory=dict)
    sampling_config: dict[str, object] = Field(default_factory=dict)
    metadata: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)
    attempt: PositiveInt = 1
    retry_count: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_interval(self) -> ExternalPairedWorkloadSourceConfig:
        if self.start_ns >= self.end_ns:
            raise ValueError("source interval must be nonempty")
        if self.retry_count >= self.attempt:
            raise ValueError("retry_count must be lower than attempt")
        return self


class ExternalPairedTargetConfig(StrictModel):
    """One explicit policy/input-plan/deployment binding for the builder."""

    format_version: Literal["robata-external-paired-target-v1"] = TARGET_CONFIG_VERSION
    deployment_id: NonEmptyString
    policy: InferencePolicy
    input_plan: InferenceInputPlan
    input_plan_part_ordinal: NonNegativeInt
    source: ExternalPairedWorkloadSourceConfig

    @model_validator(mode="after")
    def validate_part_ordinal(self) -> ExternalPairedTargetConfig:
        if self.input_plan_part_ordinal >= len(self.input_plan.call_plan.parts):
            raise ValueError("input_plan_part_ordinal is outside the call plan")
        return self


class ExternalPairedWorkloadBuildResult(StrictModel):
    """Non-secret builder evidence returned alongside the generated manifest."""

    workload: ExternalPairedWorkloadManifest
    workload_sha256: Sha256Digest
    source_report_sha256: Sha256Digest
    input_identity_sha256: Sha256Digest
    camera_artifact_sha256: tuple[Sha256Digest, ...]


class _FrozenReportEvidence:
    __slots__ = ("camera_projection", "report", "report_sha256")

    def __init__(
        self,
        *,
        report: LocalRealModelE2EReport,
        report_sha256: Sha256Digest,
        camera_projection: tuple[dict[str, object], ...],
    ) -> None:
        self.report = report
        self.report_sha256 = report_sha256
        self.camera_projection = camera_projection


def build_external_paired_workload(
    *,
    report_path: Path,
    control_target_path: Path,
    candidate_target_path: Path,
) -> ExternalPairedWorkloadBuildResult:
    """Build one strict paired workload from frozen local source evidence.

    No endpoint is contacted and no production route is changed.  The target
    files are independent, complete policy/input-plan documents.  A mismatch
    in any common source binding or rendered camera byte identity fails closed.
    """

    report_raw, _report_document = _load_exact_json_object(report_path, "local E2E report")
    try:
        report = LocalRealModelE2EReport.model_validate_json(report_raw, strict=True)
    except ValidationError as error:
        raise ExternalPairedWorkloadBuilderError("local E2E report is invalid") from error
    evidence = _verify_frozen_report(
        report_path=report_path,
        report=report,
        report_raw=report_raw,
    )

    control = _load_target(control_target_path, "control target")
    candidate = _load_target(candidate_target_path, "candidate target")
    _validate_targets(evidence, control, candidate)

    source = control.source
    input_identity = _input_identity(evidence.report, evidence.camera_projection)
    package_inputs = _package_inputs(control.input_plan)
    workload = ExternalPairedWorkloadManifest(
        experiment_id=source.experiment_id,
        contract_version=source.contract_version,
        route_id=source.route_id,
        route_policy_version=source.route_policy_version,
        source_workload_manifest_sha256=evidence.report_sha256,
        arrival_schedule_sha256=source.arrival_schedule_sha256,
        comparison_config=source.comparison_config,
        input_representation=source.input_representation,
        isolation_profile=source.isolation_profile,
        input_identity_sha256=input_identity,
        task=control.policy.task,
        package_set_id=source.package_set_id,
        mcap_id=source.mcap_id,
        camera_mapping_run_id=source.camera_mapping_run_id,
        alignment_id=source.alignment_id,
        start_ns=source.start_ns,
        end_ns=source.end_ns,
        package_inputs=package_inputs,
        input_config=source.input_config,
        sampling_config=source.sampling_config,
        metadata=source.metadata,
        attempt=source.attempt,
        retry_count=source.retry_count,
        control=ExternalPairedWorkloadTarget(
            deployment_id=control.deployment_id,
            policy=control.policy,
            input_plan=control.input_plan,
            input_plan_part_ordinal=control.input_plan_part_ordinal,
        ),
        candidate=ExternalPairedWorkloadTarget(
            deployment_id=candidate.deployment_id,
            policy=candidate.policy,
            input_plan=candidate.input_plan,
            input_plan_part_ordinal=candidate.input_plan_part_ordinal,
        ),
    )
    workload_bytes = canonical_json_bytes(workload.model_dump(mode="json")) + b"\n"
    return ExternalPairedWorkloadBuildResult(
        workload=workload,
        workload_sha256=exact_bytes_sha256(workload_bytes),
        source_report_sha256=evidence.report_sha256,
        input_identity_sha256=input_identity,
        camera_artifact_sha256=tuple(
            cast(Sha256Digest, item["sha256"]) for item in evidence.camera_projection
        ),
    )


def write_external_paired_workload(
    workload: ExternalPairedWorkloadManifest,
    output_path: Path,
) -> Sha256Digest:
    """Atomically write canonical workload bytes and return their exact digest."""

    if not isinstance(workload, ExternalPairedWorkloadManifest):
        raise TypeError("workload must be ExternalPairedWorkloadManifest")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    payload = canonical_json_bytes(workload.model_dump(mode="json")) + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except FileExistsError as error:
        raise ExternalPairedWorkloadBuilderError(
            f"temporary workload output path is already occupied: {temporary}"
        ) from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return exact_bytes_sha256(payload)


def _load_target(path: Path, label: str) -> ExternalPairedTargetConfig:
    raw, _document = _load_exact_json_object(path, label)
    try:
        return ExternalPairedTargetConfig.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise ExternalPairedWorkloadBuilderError(f"{label} is invalid") from error


def _load_exact_json_object(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ExternalPairedWorkloadBuilderError(f"cannot read {label}: {path}") from error
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ExternalPairedWorkloadBuilderError(f"{label} must not contain a UTF-8 BOM")
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ExternalPairedWorkloadBuilderError(f"{label} is not strict JSON") from error
    if not isinstance(document, dict):
        raise ExternalPairedWorkloadBuilderError(f"{label} root must be a JSON object")
    return raw, document


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _verify_frozen_report(
    *,
    report_path: Path,
    report: LocalRealModelE2EReport,
    report_raw: bytes,
) -> _FrozenReportEvidence:
    if report.status != "SUCCEEDED":
        raise ExternalPairedWorkloadBuilderError("source report must be SUCCEEDED")
    if report.execution_class != "LOCAL_QUALIFICATION":
        raise ExternalPairedWorkloadBuilderError("source report must be LOCAL_QUALIFICATION")
    if report.production_eligible or report.canonical_authority:
        raise ExternalPairedWorkloadBuilderError(
            "source report must remain non-production and non-authoritative"
        )
    if report.model.input_image_count != len(CAMERA_IDS):
        raise ExternalPairedWorkloadBuilderError("source report must contain six model inputs")
    if report.participation_coverage is not E2EParticipationCoverage.COMPLETE:
        raise ExternalPairedWorkloadBuilderError(
            "source report participation coverage must be COMPLETE"
        )
    participation_path = _resolve_report_path(report.participation_manifest_path)
    try:
        participation_raw = participation_path.read_bytes()
    except OSError as error:
        raise ExternalPairedWorkloadBuilderError(
            f"participation manifest is not readable: {participation_path}"
        ) from error
    if exact_bytes_sha256(participation_raw) != report.participation_manifest_sha256:
        raise ExternalPairedWorkloadBuilderError(
            "participation manifest digest does not match report"
        )
    participation_raw, _participation_document = _load_exact_json_object(
        participation_path, "participation manifest"
    )
    try:
        participation = E2EParticipationManifest.model_validate_json(participation_raw, strict=True)
        validate_e2e_participation_manifest_against_fragment(participation, report.trace)
    except (ValidationError, TypeError, ValueError) as error:
        raise ExternalPairedWorkloadBuilderError(
            "participation manifest does not bind to the report trace"
        ) from error
    if participation.coverage is not E2EParticipationCoverage.COMPLETE:
        raise ExternalPairedWorkloadBuilderError("participation manifest coverage must be COMPLETE")
    if len(report.camera_artifacts) != len(CAMERA_IDS):
        raise ExternalPairedWorkloadBuilderError("source report must contain six camera artifacts")

    source_path = _resolve_report_path(report.source_path)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise ExternalPairedWorkloadBuilderError(
            f"source MCAP is not readable: {source_path}"
        ) from error
    if len(source_bytes) != report.source_size_bytes:
        raise ExternalPairedWorkloadBuilderError("source MCAP byte count does not match report")
    if exact_bytes_sha256(source_bytes) != report.source_sha256:
        raise ExternalPairedWorkloadBuilderError("source MCAP digest does not match report")

    object_root = _resolve_report_path(report.storage.object_store_root)
    if not object_root.is_dir():
        raise ExternalPairedWorkloadBuilderError(
            f"local camera artifact root is not a directory: {object_root}"
        )
    projection: list[dict[str, object]] = []
    seen_camera_ids: set[str] = set()
    seen_digests: set[str] = set()
    for expected_camera, artifact in zip(CAMERA_IDS, report.camera_artifacts, strict=True):
        if artifact.camera_id != expected_camera.value:
            raise ExternalPairedWorkloadBuilderError(
                "source camera artifacts must use canonical camera order"
            )
        if artifact.camera_id in seen_camera_ids or artifact.sha256 in seen_digests:
            raise ExternalPairedWorkloadBuilderError(
                "source camera artifact identities must be unique"
            )
        seen_camera_ids.add(artifact.camera_id)
        seen_digests.add(artifact.sha256)
        artifact_path = _file_uri_path(artifact.uri)
        if not _is_relative_to(artifact_path, object_root):
            raise ExternalPairedWorkloadBuilderError(
                f"camera artifact escapes the declared local object root: {artifact.uri}"
            )
        try:
            artifact_bytes = artifact_path.read_bytes()
        except OSError as error:
            raise ExternalPairedWorkloadBuilderError(
                f"camera artifact is not readable: {artifact_path}"
            ) from error
        if len(artifact_bytes) != artifact.byte_count:
            raise ExternalPairedWorkloadBuilderError(
                f"camera artifact byte count does not match report: {artifact.camera_id}"
            )
        if exact_bytes_sha256(artifact_bytes) != artifact.sha256:
            raise ExternalPairedWorkloadBuilderError(
                f"camera artifact digest does not match report: {artifact.camera_id}"
            )
        projection.append(
            {
                "camera_id": artifact.camera_id,
                "topic": artifact.topic,
                "source_timestamp_ns": artifact.source_timestamp_ns,
                "width": artifact.width,
                "height": artifact.height,
                "media_type": artifact.media_type,
                "sha256": artifact.sha256,
                "byte_count": artifact.byte_count,
                "uri": artifact.uri,
            }
        )
    return _FrozenReportEvidence(
        report=report,
        report_sha256=exact_bytes_sha256(report_raw),
        camera_projection=tuple(projection),
    )


def _validate_targets(
    evidence: _FrozenReportEvidence,
    control: ExternalPairedTargetConfig,
    candidate: ExternalPairedTargetConfig,
) -> None:
    if control.deployment_id == candidate.deployment_id:
        raise ExternalPairedWorkloadBuilderError("control and candidate deployment IDs must differ")
    if control.source != candidate.source:
        raise ExternalPairedWorkloadBuilderError(
            "control and candidate source bindings must be byte-identical"
        )
    if control.policy.task is not candidate.policy.task:
        raise ExternalPairedWorkloadBuilderError("paired targets must use the same task")
    if control.policy.task is not control.input_plan.subject.task:
        raise ExternalPairedWorkloadBuilderError("control policy and input plan task differ")
    if candidate.policy.task is not candidate.input_plan.subject.task:
        raise ExternalPairedWorkloadBuilderError("candidate policy and input plan task differ")
    if control.policy.task is not control.input_plan.request_catalog.task:
        raise ExternalPairedWorkloadBuilderError("control input catalog task differs")
    if candidate.policy.task is not candidate.input_plan.request_catalog.task:
        raise ExternalPairedWorkloadBuilderError("candidate input catalog task differs")
    _validate_policy_plan_compatibility(control.policy, control.input_plan, "control")
    _validate_policy_plan_compatibility(candidate.policy, candidate.input_plan, "candidate")
    _validate_same_source_input(evidence, control.input_plan, candidate.input_plan)


def _validate_policy_plan_compatibility(
    policy: InferencePolicy,
    plan: InferenceInputPlan,
    role: str,
) -> None:
    target = plan.target
    checks = (
        ("provider", policy.provider, target.provider),
        ("model_name", policy.model_name, target.model_name),
        ("model_version", policy.model_version, target.model_version),
        ("adapter_version", policy.adapter_version, target.adapter_version),
    )
    for field, left, right in checks:
        if left != right:
            raise ExternalPairedWorkloadBuilderError(
                f"{role} policy/input plan {field} binding differs"
            )
    prompt = plan.prompt_output
    if policy.prompt_version != prompt.prompt_version:
        raise ExternalPairedWorkloadBuilderError(f"{role} policy prompt_version differs")
    if policy.prompt_sha256 != prompt.prompt_sha256:
        raise ExternalPairedWorkloadBuilderError(f"{role} policy prompt_sha256 differs")
    if policy.output_schema.sha256 != prompt.provider_response_schema_sha256:
        raise ExternalPairedWorkloadBuilderError(
            f"{role} policy output schema is not the input plan response schema"
        )
    if policy.required_input_mode is not InputMode.MULTI_IMAGE:
        raise ExternalPairedWorkloadBuilderError(
            f"{role} policy required_input_mode must be MULTI_IMAGE for six frozen frames"
        )


def _validate_same_source_input(
    evidence: _FrozenReportEvidence,
    control_plan: InferenceInputPlan,
    candidate_plan: InferenceInputPlan,
) -> None:
    control_items = _input_item_projection(control_plan)
    candidate_items = _input_item_projection(candidate_plan)
    if control_items != candidate_items:
        raise ExternalPairedWorkloadBuilderError(
            "control and candidate input plans do not refer to identical rendered input"
        )
    for role, plan in (("control", control_plan), ("candidate", candidate_plan)):
        _validate_plan_against_report(evidence, plan, role)
    if control_plan.subject.packages != candidate_plan.subject.packages:
        raise ExternalPairedWorkloadBuilderError("control and candidate package identities differ")
    if (
        control_plan.request_catalog.semantic_sha256
        != candidate_plan.request_catalog.semantic_sha256
    ):
        raise ExternalPairedWorkloadBuilderError(
            "control and candidate request catalog identities differ"
        )


def _validate_plan_against_report(
    evidence: _FrozenReportEvidence,
    plan: InferenceInputPlan,
    role: str,
) -> None:
    items = tuple(sorted(plan.rendered_items, key=lambda item: item.provider_item_ordinal))
    if len(items) != len(CAMERA_IDS):
        raise ExternalPairedWorkloadBuilderError(
            f"{role} input plan must contain exactly six rendered camera items"
        )
    if tuple(item.camera_id.value for item in items) != tuple(
        camera.value for camera in CAMERA_IDS
    ):
        raise ExternalPairedWorkloadBuilderError(
            f"{role} input plan rendered camera order is not canonical"
        )
    for expected, item, camera in zip(evidence.camera_projection, items, CAMERA_IDS, strict=True):
        if item.camera_id.value != expected["camera_id"] or camera.value != expected["camera_id"]:
            raise ExternalPairedWorkloadBuilderError(f"{role} camera identity differs from report")
        if item.source_artifact_sha256 != expected["sha256"]:
            raise ExternalPairedWorkloadBuilderError(
                f"{role} source artifact digest differs from report: {camera.value}"
            )
        artifact = item.artifact
        if (
            artifact.sha256 != expected["sha256"]
            or artifact.uri != expected["uri"]
            or artifact.byte_count != expected["byte_count"]
            or artifact.width != expected["width"]
            or artifact.height != expected["height"]
            or artifact.media_type != expected["media_type"]
            or item.source_timestamp_ns != expected["source_timestamp_ns"]
        ):
            raise ExternalPairedWorkloadBuilderError(
                f"{role} rendered artifact differs from frozen report: {camera.value}"
            )
        if item.transform.operation is not TransformOperation.NONE:
            raise ExternalPairedWorkloadBuilderError(
                f"{role} input plan applies a transform to frozen local artifacts"
            )


def _input_item_projection(plan: InferenceInputPlan) -> tuple[dict[str, object], ...]:
    """Project only source/rendering facts, not plan-local row identities."""

    return tuple(
        {
            "package_ordinal": item.package_ordinal,
            "camera_id": item.camera_id.value,
            "camera_ordinal": item.camera_ordinal,
            "frame_ordinal": item.frame_ordinal,
            "aligned_timestamp_ns": item.aligned_timestamp_ns,
            "source_timestamp_ns": item.source_timestamp_ns,
            "source_artifact_sha256": item.source_artifact_sha256,
            "artifact": {
                "uri": item.artifact.uri,
                "sha256": item.artifact.sha256,
                "byte_count": item.artifact.byte_count,
                "media_type": item.artifact.media_type,
                "encoding": item.artifact.encoding,
                "width": item.artifact.width,
                "height": item.artifact.height,
            },
            "transform": item.transform.model_dump(mode="json"),
        }
        for item in sorted(plan.rendered_items, key=lambda value: value.provider_item_ordinal)
    )


def _package_inputs(plan: InferenceInputPlan) -> tuple[PackageInput, ...]:
    return tuple(
        PackageInput(
            package_id=package.package_id,
            package_semantic_content_sha256=package.semantic_content_sha256,
            package_manifest_sha256=package.manifest_bytes_sha256,
            role="primary",
            ordinal=package.ordinal,
        )
        for package in plan.request_catalog.packages
    )


def _input_identity(
    report: LocalRealModelE2EReport,
    camera_projection: tuple[dict[str, object], ...],
) -> Sha256Digest:
    # URI and topic are locators, not semantic input identity.  Keep the
    # source/content digests and exact frame geometry/timestamps only.
    camera_identity = tuple(
        {
            key: str(value) if key == "source_timestamp_ns" else value
            for key, value in artifact.items()
            if key not in {"topic", "uri"}
        }
        for artifact in camera_projection
    )
    return semantic_sha256(
        {
            "version": "robata-external-paired-input-identity-v1",
            "source_sha256": report.source_sha256,
            "source_size_bytes": report.source_size_bytes,
            "camera_artifacts": camera_identity,
        }
    )


def _resolve_report_path(value: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve()
    if not resolved.is_file() and not resolved.is_dir():
        raise ExternalPairedWorkloadBuilderError(f"reported path is not present: {resolved}")
    return resolved


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.query or parsed.fragment:
        raise ExternalPairedWorkloadBuilderError(
            f"camera artifact URI must be a local file URI without query/fragment: {uri}"
        )
    if parsed.netloc not in ("", "localhost"):
        raise ExternalPairedWorkloadBuilderError(
            f"camera artifact URI must not name a remote host: {uri}"
        )
    path = unquote(parsed.path)
    if os.name == "nt" and path.startswith("/") and len(path) >= 3 and path[2] == ":":
        path = path[1:]
    if not path:
        raise ExternalPairedWorkloadBuilderError(f"camera artifact URI has no path: {uri}")
    return Path(path).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "TARGET_CONFIG_VERSION",
    "ExternalPairedTargetConfig",
    "ExternalPairedWorkloadBuildResult",
    "ExternalPairedWorkloadBuilderError",
    "ExternalPairedWorkloadSourceConfig",
    "build_external_paired_workload",
    "write_external_paired_workload",
]
