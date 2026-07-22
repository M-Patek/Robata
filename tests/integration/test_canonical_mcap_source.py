from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import pytest

from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository
from robata.application.canonical import local_composition as local_composition_module
from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    run_local_canonical_mcap,
)
from robata.application.canonical.mcap_source import (
    CanonicalMcapSourceError,
    authorize_mcap_mapping,
    load_canonical_mcap_source,
)
from robata.application.canonical.media_quality import (
    registered_local_media_quality_report_document,
    validate_registered_local_media_quality_report_document,
)
from robata.application.canonical.primary_completion import PrimaryCompletionEvidenceRole
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRegistry
from robata.review.routing import ReviewRoutingDisposition
from tests.support.six_camera_mcap import (
    SIX_CAMERA_MCAP_SHA256,
    SIX_CAMERA_TOPICS,
    write_six_camera_mcap,
)


def _write_mapping(path: Path) -> Path:
    path.write_bytes(
        canonical_json_bytes(
            {
                "profile_id": "canonical-mcap-integration-v1",
                "version": "canonical-mcap-integration-v1",
                "profile_kind": "TEST_FIXTURE",
                "approval_status": "UNAPPROVED",
                "approved": False,
                "mapping_policy": "EXACT_TOPIC",
                "required_schema": "foxglove.CompressedImage",
                "topics": {
                    camera_id.value: topic
                    for camera_id, topic in zip(
                        CAMERA_IDS,
                        SIX_CAMERA_TOPICS,
                        strict=True,
                    )
                },
            }
        )
    )
    return path


def test_real_mcap_builds_admitted_canonical_source_bundle(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )

    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert bundle.source_content_sha256 == SIX_CAMERA_MCAP_SHA256
    assert bundle.admitted_context.source_content_sha256 == SIX_CAMERA_MCAP_SHA256
    assert bundle.admitted_context.alignment_manifest.method.value == "mcap_log_time"
    assert tuple(bundle.frame_index.cameras.keys()) == CAMERA_IDS
    assert all(bundle.frame_index.cameras[camera_id].frames for camera_id in CAMERA_IDS)
    resolved = tuple(
        bundle.resolve_artifact(camera_id, bundle.frame_index.cameras[camera_id].frames[0])
        for camera_id in CAMERA_IDS
    )
    assert all(artifact is not None for artifact in resolved)
    assert all(artifact.artifact.media_type == "image/png" for artifact in resolved if artifact)
    report_path = tmp_path / "media-state" / "media-quality-report.json"
    registry = SchemaRegistry()
    report_document = registered_local_media_quality_report_document(
        bundle.media_quality_report,
        registry,
    )
    validate_registered_local_media_quality_report_document(report_document, registry)
    assert report_path.read_bytes() == canonical_json_bytes(report_document)
    supplemental = report_document["supplemental_targets"]
    assert isinstance(supplemental, dict)
    assert supplemental["candidate_count"] > 0
    assert supplemental["targets"]
    assert all(target["provenance"] for target in supplemental["targets"])
    assert "dropped_by_per_camera_budget" in supplemental
    assert "dropped_by_total_budget" in supplemental


def test_cache_miss_lazily_materializes_exact_verified_frame(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    camera_id = CAMERA_IDS[1]
    frame = bundle.frame_index.cameras[camera_id].frames[-1]
    key = (camera_id, frame.source_frame_id)
    expected = bundle.resolve_artifact(camera_id, frame)
    assert expected is not None

    bundle._artifact_resolver.artifacts.pop(key)
    actual = bundle.resolve_artifact(camera_id, frame)

    assert actual == expected
    assert bundle._artifacts[key] == actual
    assert bundle.resolve_artifact(camera_id, frame) is actual
    tampered = frame.model_copy(update={"source_timestamp_ns": frame.source_timestamp_ns + 1})
    with pytest.raises(CanonicalMcapSourceError, match="canonical frame index"):
        bundle.resolve_artifact(camera_id, tampered)


def test_max_duration_clamps_half_open_source_and_quality_window(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        max_duration_ns=250_000_000,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert bundle.requested_interval.start_ns == 0
    assert bundle.requested_interval.end_ns == 250_000_000
    assert bundle.media_quality_report.window_limited is True
    assert bundle.media_quality_report.requested_max_duration_ns == 250_000_000
    assert all(
        len(ledger.decoded_observations) == 1
        for ledger in bundle.media_quality_report.camera_ledgers
    )
    for camera_id in CAMERA_IDS:
        outside = bundle.frame_index.cameras[camera_id].frames[-1]
        assert outside.alignment_projection.aligned_timestamp_ns >= bundle.requested_interval.end_ns
        assert bundle.resolve_artifact(camera_id, outside) is None


def test_existing_quality_report_bytes_fail_closed(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    mapping = _write_mapping(tmp_path / "mapping.json")
    state_dir = tmp_path / "media-state"
    authorization = authorize_mcap_mapping(
        mapping,
        allow_unapproved_profile=True,
    )
    load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    (state_dir / "media-quality-report.json").write_bytes(b"{}")

    with pytest.raises(
        CanonicalMcapSourceError,
        match="existing media quality report bytes are inconsistent",
    ):
        load_canonical_mcap_source(
            source,
            authorization=authorization,
            state_dir=state_dir,
            expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
            clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        )


def test_real_mcap_command_commits_and_exactly_replays(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    mapping = _write_mapping(tmp_path / "mapping.json")
    state_dir = tmp_path / "canonical-state"

    first = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key="real-mcap-replay",
        allow_unapproved_profile=True,
    )
    replay = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key="real-mcap-replay",
        allow_unapproved_profile=True,
    )

    assert first.replayed is False
    assert first.fixture_inference_calls > 0
    assert first.event_ids
    assert first.revision_ids
    assert first.outbox_ids
    assert first.production_eligible is False
    assert first.network_call_count == 0
    assert first.media_quality_binding is not None
    assert first.media_quality_binding.requires_review is True
    assert first.supplemental_qa_evidence is not None
    assert first.supplemental_qa_evidence.schema_version == "2.0"
    assert first.supplemental_qa_evidence.production_eligible is False
    assert first.review_routing.disposition is ReviewRoutingDisposition.ENQUEUED
    assert replay.replayed is True
    assert replay.fixture_inference_calls == 0
    assert replay.run_id == first.run_id
    assert replay.command_sha256 == first.command_sha256
    assert replay.completion_semantic_sha256 == first.completion_semantic_sha256
    assert replay.event_ids == first.event_ids
    assert replay.revision_ids == first.revision_ids
    assert replay.outbox_ids == first.outbox_ids
    assert replay.media_quality_binding == first.media_quality_binding
    assert replay.supplemental_qa_evidence == first.supplemental_qa_evidence
    assert replay.review_routing.disposition is ReviewRoutingDisposition.ALREADY_ENQUEUED
    committed = SQLitePrimaryCompletionRepository(
        state_dir / "primary-completion.sqlite3",
        registry=SchemaRegistry(),
    ).get(first.run_id)
    assert committed is not None
    assert tuple(reference.role for reference in committed.evidence_references) == (
        PrimaryCompletionEvidenceRole.MEDIA_QUALITY_REPORT,
        PrimaryCompletionEvidenceRole.SUPPLEMENTAL_QA_EVIDENCE,
    )
    assert all(reference.byte_count > 0 for reference in committed.evidence_references)


def test_recovered_mcap_completion_rejects_tampered_quality_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    mapping = _write_mapping(tmp_path / "mapping.json")
    state_dir = tmp_path / "canonical-state"
    run_key = "tampered-quality-recovery"

    first = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key=run_key,
        allow_unapproved_profile=True,
    )
    assert first.media_quality_binding is not None
    reports = tuple((state_dir / "mcap").glob("*/media-quality-report.json"))
    assert len(reports) == 1
    report_bytes = reports[0].read_bytes()
    reports[0].write_bytes(report_bytes + b"\n")
    reconciliation_calls: list[str] = []
    original_reconcile = local_composition_module._reconcile_local_outbox

    def observed_reconcile(*args: object, **kwargs: object):
        reconciliation_calls.append("outbox")
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        local_composition_module,
        "_reconcile_local_outbox",
        observed_reconcile,
    )

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_mcap(
            source,
            mapping,
            state_dir,
            run_key=run_key,
            allow_unapproved_profile=True,
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    assert "exact canonical JSON" in str(caught.value)
    assert reconciliation_calls == ["outbox"]

    reports[0].write_bytes(report_bytes)
    assert first.supplemental_qa_evidence is not None
    selected = next(
        outcome.selected_artifact
        for outcome in first.supplemental_qa_evidence.package.outcomes
        if outcome.selected_artifact is not None
    )
    artifact_path = Path(url2pathname(unquote(urlsplit(selected.artifact.uri).path)))
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(CanonicalLocalCompositionError) as caught_artifact:
        run_local_canonical_mcap(
            source,
            mapping,
            state_dir,
            run_key=run_key,
            allow_unapproved_profile=True,
        )

    assert caught_artifact.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    assert "artifact byte count is inconsistent" in str(caught_artifact.value)


def test_corrupt_mcap_fails_before_completion(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.mcap"
    source.write_bytes(write_six_camera_mcap(tmp_path / "valid.mcap").read_bytes()[:256])

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_mcap(
            source,
            _write_mapping(tmp_path / "mapping.json"),
            tmp_path / "canonical-state",
            run_key="corrupt-input",
            allow_unapproved_profile=True,
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.SOURCE_INVALID
