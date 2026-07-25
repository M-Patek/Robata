"""Local canonical runtime for exact supplemental frame QA evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

if TYPE_CHECKING:
    from robata.application.canonical.mcap_source import CanonicalMcapSourceBundle

from robata.application.canonical.media_quality_binding import (
    derive_local_media_quality_binding_document,
)
from robata.application.canonical.media_quality_source_binding import (
    bind_registered_media_quality_source,
)
from robata.application.canonical.media_quality_supplemental import (
    freeze_registered_media_quality_targets,
)
from robata.application.canonical.supplemental_qa_evidence import (
    load_registered_local_supplemental_qa_evidence_document,
    parse_local_supplemental_qa_evidence_document,
    publish_registered_local_supplemental_qa_evidence_document,
    registered_local_supplemental_qa_evidence_document,
)
from robata.contracts.cameras import CameraId
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.qa_pipeline.supplemental import DeterministicSupplementalQaDenseConsumer
from robata.qa_pipeline.supplemental_wire import LocalSupplementalQaEvidence
from robata.sampling.materializer import MaterializedArtifactManifest
from robata.sampling.supplemental import (
    SUPPLEMENTAL_DEDUPE_POLICY_VERSION as LOCAL_SUPPLEMENTAL_DEDUPE_POLICY_VERSION,
)
from robata.sampling.supplemental import (
    SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION as LOCAL_SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION,
)
from robata.sampling.supplemental import (
    ExplicitTargetPackageMaterializer,
    MaterializedSupplementalPackage,
)

LOCAL_SUPPLEMENTAL_QA_RUNTIME_VERSION: Final = "canonical-local-supplemental-qa-v1"
LOCAL_SUPPLEMENTAL_SELECTION_TOLERANCE_NS: Final = 300_000_000

LOCAL_SUPPLEMENTAL_QA_EVIDENCE_FILENAME: Final = "local-supplemental-qa-evidence.json"


def build_and_publish_local_supplemental_qa_evidence(
    *,
    bundle: CanonicalMcapSourceBundle,
    state_dir: Path,
    registry: SchemaRegistry,
    created_at: str,
) -> LocalSupplementalQaEvidence | None:
    """Build the bounded explicit-target branch before primary completion."""

    # Keep JSON fixture composition independent of optional PyAV/MCAP imports.
    # The concrete runtime check runs only on the MCAP supplemental branch.
    from robata.application.canonical.mcap_source import CanonicalMcapSourceBundle

    if not isinstance(bundle, CanonicalMcapSourceBundle):
        raise TypeError("bundle must be a CanonicalMcapSourceBundle")
    root = _state_root(state_dir)
    evidence_path = root / LOCAL_SUPPLEMENTAL_QA_EVIDENCE_FILENAME
    context = bundle.admitted_context
    source_binding = bind_registered_media_quality_source(
        bundle.media_quality_report,
        registry=registry,
        source_content_sha256=context.source_content_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
    )
    plan = freeze_registered_media_quality_targets(
        bundle.media_quality_report,
        registry=registry,
        source_binding=source_binding,
        selection_tolerance_ns=LOCAL_SUPPLEMENTAL_SELECTION_TOLERANCE_NS,
        tie_break_policy_version=LOCAL_SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION,
        dedupe_policy_version=LOCAL_SUPPLEMENTAL_DEDUPE_POLICY_VERSION,
    )
    if plan is None:
        _require_absent(evidence_path)
        return None

    materialized = ExplicitTargetPackageMaterializer().materialize(
        plan=plan,
        frame_index=bundle.frame_index,
        artifact_resolver=bundle.resolve_artifact,
        created_at=created_at,
    )
    consumer = DeterministicSupplementalQaDenseConsumer()
    input_plan = consumer.prepare(materialized)
    result = consumer.consume(
        materialized,
        input_plan,
        artifact_bytes_resolver=_artifact_bytes_resolver(root / "frames"),
    )
    document = registered_local_supplemental_qa_evidence_document(
        plan,
        materialized,
        input_plan,
        result,
        registry,
    )
    publish_registered_local_supplemental_qa_evidence_document(
        evidence_path,
        document,
        registry,
    )
    return parse_local_supplemental_qa_evidence_document(document, registry)


def load_and_verify_local_supplemental_qa_evidence(
    *,
    media_quality_document: Mapping[str, object],
    state_dir: Path,
    expected_source_content_sha256: str,
    registry: SchemaRegistry,
) -> LocalSupplementalQaEvidence | None:
    """Replay the registered envelope and re-read every selected artifact."""

    root = _state_root(state_dir)
    evidence_path = root / LOCAL_SUPPLEMENTAL_QA_EVIDENCE_FILENAME
    report_targets = _report_target_coordinates(media_quality_document)
    if not report_targets:
        _require_absent(evidence_path)
        return None

    document = load_registered_local_supplemental_qa_evidence_document(
        evidence_path,
        registry,
    )
    evidence = parse_local_supplemental_qa_evidence_document(document, registry)
    _validate_report_binding(
        evidence,
        media_quality_document=media_quality_document,
        report_targets=report_targets,
        expected_source_content_sha256=expected_source_content_sha256,
        registry=registry,
    )

    materialized = MaterializedSupplementalPackage(
        package=evidence.package,
        manifest_bytes=canonical_json_bytes(evidence.package),
        manifest_sha256=evidence.package_manifest_sha256,
    )
    consumer = DeterministicSupplementalQaDenseConsumer()
    replayed_input = consumer.prepare(materialized)
    if replayed_input != evidence.input_plan:
        raise ValueError("persisted supplemental QA input plan does not replay exactly")
    replayed_result = consumer.consume(
        materialized,
        replayed_input,
        artifact_bytes_resolver=_artifact_bytes_resolver(root / "frames"),
    )
    if replayed_result != evidence.result:
        raise ValueError("persisted supplemental QA result does not replay exactly")
    return evidence


def _validate_report_binding(
    evidence: LocalSupplementalQaEvidence,
    *,
    media_quality_document: Mapping[str, object],
    report_targets: tuple[tuple[CameraId, int], ...],
    expected_source_content_sha256: str,
    registry: SchemaRegistry,
) -> None:
    report_ref = SchemaRef.model_validate(
        media_quality_document.get("schema_ref"),
        strict=True,
    )
    supplemental = _mapping(
        media_quality_document.get("supplemental_targets"),
        "supplemental_targets",
    )
    quality_binding = derive_local_media_quality_binding_document(
        media_quality_document,
        registry,
    )
    source_binding = evidence.frozen_plan.source_binding
    if (
        source_binding.report_schema_ref != report_ref
        or source_binding.report_semantic_sha256
        != _string(media_quality_document.get("semantic_sha256"), "semantic_sha256")
        or source_binding.supplemental_target_plan_semantic_sha256
        != _string(supplemental.get("semantic_sha256"), "supplemental_targets.semantic_sha256")
        or source_binding.media_quality_binding_semantic_sha256 != quality_binding.semantic_sha256
        or source_binding.source_content_sha256 != expected_source_content_sha256
    ):
        raise ValueError("supplemental QA evidence does not bind the persisted source report")
    if (
        tuple((target.camera_id, target.target_ns) for target in evidence.frozen_plan.targets)
        != report_targets
    ):
        raise ValueError("supplemental QA targets differ from the persisted source report")
    if evidence.frozen_plan.target_policy_version != _string(
        supplemental.get("policy_version"),
        "supplemental_targets.policy_version",
    ):
        raise ValueError("supplemental QA target policy differs from the source report")
    interval = _mapping(
        supplemental.get("interval"),
        "supplemental_targets.interval",
    )
    if evidence.frozen_plan.effective_interval.model_dump(mode="json") != dict(interval):
        raise ValueError("supplemental QA interval differs from the source report")


def _report_target_coordinates(
    document: Mapping[str, object],
) -> tuple[tuple[CameraId, int], ...]:
    supplemental = _mapping(document.get("supplemental_targets"), "supplemental_targets")
    targets = _sequence(supplemental.get("targets"), "supplemental_targets.targets")
    coordinates: list[tuple[CameraId, int]] = []
    for ordinal, raw in enumerate(targets):
        target = _mapping(raw, f"supplemental_targets.targets[{ordinal}]")
        camera_id = CameraId(
            _string(target.get("camera_id"), f"supplemental_targets.targets[{ordinal}].camera_id")
        )
        target_ns = int(
            _string(target.get("target_ns"), f"supplemental_targets.targets[{ordinal}].target_ns")
        )
        coordinates.append((camera_id, target_ns))
    return tuple(coordinates)


def _artifact_bytes_resolver(
    artifact_root: Path,
) -> Callable[[MaterializedArtifactManifest], bytes]:
    root = artifact_root.resolve()

    def resolve(artifact: MaterializedArtifactManifest) -> bytes:
        split = urlsplit(artifact.uri)
        if (
            split.scheme.lower() != "file"
            or split.netloc not in {"", "localhost"}
            or split.query
            or split.fragment
        ):
            raise ValueError("supplemental QA artifacts must use local file URIs")
        candidate = Path(url2pathname(unquote(split.path)))
        if not candidate.is_absolute() or candidate.is_symlink():
            raise ValueError("supplemental QA artifact path must be an absolute regular file")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("supplemental QA artifact path escapes the canonical frame root")
        return resolved.read_bytes()

    return resolve


def _state_root(state_dir: Path) -> Path:
    if not isinstance(state_dir, Path):
        raise TypeError("state_dir must be pathlib.Path")
    root = state_dir.resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("supplemental QA state root must be a regular directory")
    return root


def _require_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("supplemental QA evidence exists for an empty target plan")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


__all__ = [
    "LOCAL_SUPPLEMENTAL_DEDUPE_POLICY_VERSION",
    "LOCAL_SUPPLEMENTAL_QA_EVIDENCE_FILENAME",
    "LOCAL_SUPPLEMENTAL_QA_RUNTIME_VERSION",
    "LOCAL_SUPPLEMENTAL_SELECTION_TOLERANCE_NS",
    "LOCAL_SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION",
    "build_and_publish_local_supplemental_qa_evidence",
    "load_and_verify_local_supplemental_qa_evidence",
]
