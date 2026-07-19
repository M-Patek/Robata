"""Run the complete local MCAP-to-action path with the deterministic fake model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from robata.adapters import (
    EXPORT_CONFIG,
    EXPORT_PROFILE_ID,
    EXPORT_PROFILE_VERSION,
    EXPORTER_NAME,
    EXPORTER_VERSION,
    DeterministicFakeVisionModelAdapter,
    LocalArtifactRegistry,
    OfficialMcapInspector,
    PyAvFrameMaterializer,
    PyAvH264Mp4Exporter,
)
from robata.application import (
    LocalVideoExportRequest,
    RegisteredSixCameraVideoExportService,
    VideoExporterDescriptor,
    VideoExportRunError,
)
from robata.application.mainline import (
    LocalMainlineConfig,
    LocalMainlinePipeline,
    MainlineRunError,
)
from robata.contracts import SchemaRegistryError, semantic_sha256
from robata.contracts.video_export import VideoExporterMode
from robata.ingestion import ExactTopicMappingPolicy, TopicMappingProfile
from robata.ports import ArtifactRegistryError, IngestionError, VideoExportError
from robata.runtime.execution import (
    ExecutionEvidenceError,
    PublishedExecutionEvidence,
    write_execution_evidence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_CONFIG = REPOSITORY_ROOT / "config" / "genrobot-observed-v0.json"


class CliArgumentError(ValueError):
    """A command-line validation failure that is safe to serialize."""

    code = "INVALID_ARGUMENT"


class CliResultError(RuntimeError):
    """A completed component returned accounting that violates the CLI contract."""

    code = "INVALID_PIPELINE_RESULT"


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise CliArgumentError(message)


@dataclass(frozen=True, slots=True)
class _Rate:
    numerator: int
    denominator: int


def _positive_rate(value: str) -> _Rate:
    parts = value.split("/", maxsplit=1)
    if len(parts) == 1:
        parts.append("1")
    try:
        numerator, denominator = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("rate must be NUM or NUM/DEN") from error
    if numerator <= 0 or denominator <= 0:
        raise argparse.ArgumentTypeError("rate numerator and denominator must be positive")
    return _Rate(numerator=numerator, denominator=denominator)


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(
        description=(
            "Export six camera videos, materialize sampled frames, and run the complete "
            "local action-event mainline with a deterministic fake model."
        )
    )
    parser.add_argument("source", type=Path, help="local MCAP source")
    parser.add_argument("output", type=Path, help="new output root")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=DEFAULT_MAPPING_CONFIG,
        help="exact topic mapping profile",
    )
    parser.add_argument(
        "--namespace",
        default="robata",
        help="recording-identity namespace; source paths are excluded from identity",
    )
    parser.add_argument(
        "--allow-unapproved",
        "--allow-unapproved-profile",
        dest="allow_unapproved_profile",
        action="store_true",
        help="explicitly permit the local unapproved mapping development override",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="shared local artifact registry; defaults beside the output root",
    )
    parser.add_argument(
        "--no-event",
        action="store_true",
        help="run the deterministic fake adapter in its no-event mode",
    )
    parser.add_argument(
        "--coarse-rate",
        type=_positive_rate,
        default=_Rate(1, 1),
        metavar="NUM[/DEN]",
        help="coarse sampling rate in frames per second (default: 1)",
    )
    parser.add_argument(
        "--dense-rate",
        type=_positive_rate,
        default=_Rate(2, 1),
        metavar="NUM[/DEN]",
        help="dense sampling rate in frames per second (default: 2)",
    )
    return parser


def _error_payload(error: Exception, *, stage: str) -> dict[str, Any]:
    code = getattr(error, "code", "UNEXPECTED_ERROR")
    code_value = getattr(code, "value", code)
    return {
        "ok": False,
        "stage": stage,
        "error": {
            "code": str(code_value),
            "message": str(error),
        },
        "provider_requests": 0,
    }


def _video_descriptor() -> VideoExporterDescriptor:
    return VideoExporterDescriptor(
        name=EXPORTER_NAME,
        version=EXPORTER_VERSION,
        mode=VideoExporterMode.REMUX,
        export_profile_id=EXPORT_PROFILE_ID,
        profile_version=EXPORT_PROFILE_VERSION,
        canonical_config_sha256=semantic_sha256(EXPORT_CONFIG),
    )


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _validate_output_paths(
    output_root: Path,
    explicit_registry_root: Path | None,
) -> Path:
    if output_root.exists() or output_root.is_symlink():
        raise CliArgumentError("output root must be absent and must not be a symlink")

    if explicit_registry_root is None:
        return output_root.parent / ".robata-artifacts"

    registry_root = _absolute_path(explicit_registry_root)
    canonical_output = output_root.resolve(strict=False)
    canonical_registry = registry_root.resolve(strict=False)
    if canonical_registry.is_relative_to(canonical_output):
        raise CliArgumentError("registry root must not be inside the output root")
    return registry_root


def _validate_staged_directory(
    *,
    label: str,
    actual: object,
    expected: Path,
) -> None:
    if not isinstance(actual, Path):
        raise CliResultError(f"{label} result has no Path output_directory")
    if _absolute_path(actual) != expected:
        raise CliResultError(f"{label} result points outside its staging directory")
    if actual.is_symlink() or not actual.is_dir():
        raise CliResultError(f"{label} result did not publish a regular directory")


def _create_staging_root(output_root: Path) -> Path:
    """Create a unique sibling with inherited ACLs for nested component staging."""

    for _ in range(32):
        candidate = output_root.parent / f".{output_root.name}.partial-{uuid.uuid4().hex[:16]}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise OSError("could not allocate a unique staging root")


def _success_payload(
    *,
    video: Any,
    analysis: Any,
    staging_root: Path,
    output_root: Path,
    registry_root: Path,
    provider_requests: int,
    evidence: PublishedExecutionEvidence | None = None,
) -> dict[str, Any]:
    _validate_staged_directory(
        label="video",
        actual=getattr(video, "output_directory", None),
        expected=staging_root / "video",
    )
    _validate_staged_directory(
        label="analysis",
        actual=getattr(analysis, "output_directory", None),
        expected=staging_root / "analysis",
    )
    report = analysis.bundle.report
    expected_attempts = 5 if report.event_count else 2
    if report.fake_inference_attempt_count != expected_attempts:
        raise CliResultError(
            "complete local run must contain five fake attempts with an event or two "
            "without an event"
        )
    if provider_requests != 0 or report.real_provider_request_count != 0:
        raise CliResultError("local fake run attempted a real provider request")
    payload = {
        "ok": True,
        "execution_mode": "LOCAL_DEVELOPMENT_FAKE_MODEL",
        "run_status": report.status.value,
        "fake_inference_attempt_count": report.fake_inference_attempt_count,
        "event_count": report.event_count,
        "provider_requests": 0,
        "video": {
            "output_directory": str(output_root / "video"),
            "manifest_sha256": video.manifest_sha256,
            "manifest_artifact_id": video.manifest_artifact_id,
            "schema_version": video.manifest.schema_version,
            "derivation_reused": video.derivation_reused,
            "materialized_view_reused": video.materialized_view_reused,
        },
        "analysis": {
            "output_directory": str(output_root / "analysis"),
            "bundle_sha256": analysis.bundle_sha256,
            "run_id": report.run_id,
            "pipeline_version": report.pipeline_version,
            "events": [
                {
                    "event_id": event.event_id,
                    "action_type": event.action_type,
                    "start_ns": str(event.interval.start_ns),
                    "end_ns": str(event.interval.end_ns),
                    "status": event.status.value,
                    "production_eligible": event.production_eligible,
                }
                for event in analysis.bundle.events
            ],
        },
        "registry_root": str(registry_root.resolve()),
    }
    if evidence is not None:
        payload["execution"] = {
            "manifest_sha256": evidence.manifest_sha256,
            "manifest_semantic_sha256": evidence.manifest_semantic_sha256,
            "audit_sha256": evidence.audit_sha256,
            "artifact_count": evidence.artifact_count,
            "manifest_path": str(output_root / "execution-manifest.json"),
            "audit_path": str(output_root / "execution-audit.ndjson"),
        }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    stage = "arguments"
    staging_root: Path | None = None
    published = False
    try:
        args = _parser().parse_args(argv)
        output_root = _absolute_path(args.output)
        registry_root = _validate_output_paths(
            output_root,
            args.registry_root,
        )

        stage = "mapping_authorization"
        profile = TopicMappingProfile.load(args.mapping_config)
        # Authorization intentionally precedes source inspection and output creation.
        mapping_policy = ExactTopicMappingPolicy.from_profile(
            profile,
            allow_unapproved=args.allow_unapproved_profile,
        )

        stage = "source_inspection"
        inspection = OfficialMcapInspector().inspect(args.source)
        channels = mapping_policy.resolve(inspection)

        stage = "staging"
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists() or output_root.is_symlink():
            raise CliArgumentError("output root appeared before staging")
        staging_root = _create_staging_root(output_root)
        video_output = staging_root / "video"
        analysis_output = staging_root / "analysis"

        stage = "video_export"
        video = RegisteredSixCameraVideoExportService(
            PyAvH264Mp4Exporter(),
            LocalArtifactRegistry(registry_root),
        ).export_local(
            LocalVideoExportRequest(
                source=args.source,
                output_directory=video_output,
                namespace=args.namespace,
                inspection=inspection,
                channels=channels,
                mapping_profile=profile,
                mapping_profile_digest=profile.semantic_digest,
                exporter=_video_descriptor(),
            )
        )

        stage = "analysis"
        model = DeterministicFakeVisionModelAdapter(no_event=args.no_event)
        analysis = LocalMainlinePipeline(
            PyAvFrameMaterializer(),
            model,
            config=LocalMainlineConfig(
                coarse_rate_num=args.coarse_rate.numerator,
                coarse_rate_den=args.coarse_rate.denominator,
                dense_rate_num=args.dense_rate.numerator,
                dense_rate_den=args.dense_rate.denominator,
            ),
        ).run(video, analysis_output)

        stage = "execution_evidence"
        evidence = write_execution_evidence(
            staging_root,
            report=analysis.bundle.report,
            video=video,
            model=model,
            provider_requests=model.external_provider_requests,
        )
        payload = _success_payload(
            video=video,
            analysis=analysis,
            staging_root=staging_root,
            output_root=output_root,
            registry_root=registry_root,
            provider_requests=model.external_provider_requests,
            evidence=evidence,
        )
        payload_json = json.dumps(payload, indent=2, sort_keys=True)

        stage = "publication"
        if output_root.exists() or output_root.is_symlink():
            raise CliResultError("output root appeared before atomic publication")
        staging_root.rename(output_root)
        published = True
        print(payload_json)
        return 0
    except (
        ArtifactRegistryError,
        ExecutionEvidenceError,
        CliArgumentError,
        CliResultError,
        IngestionError,
        MainlineRunError,
        SchemaRegistryError,
        VideoExportError,
        VideoExportRunError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps(_error_payload(error, stage=stage), indent=2, sort_keys=True))
        return 2
    finally:
        if staging_root is not None and not published:
            with suppress(OSError):
                shutil.rmtree(staging_root)


if __name__ == "__main__":
    raise SystemExit(main())
