from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    run_local_canonical_mcap,
)
from robata.application.canonical.mcap_source import (
    authorize_mcap_mapping,
    load_canonical_mcap_source,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes
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
    assert replay.replayed is True
    assert replay.fixture_inference_calls == 0
    assert replay.run_id == first.run_id
    assert replay.command_sha256 == first.command_sha256
    assert replay.completion_semantic_sha256 == first.completion_semantic_sha256
    assert replay.event_ids == first.event_ids
    assert replay.revision_ids == first.revision_ids
    assert replay.outbox_ids == first.outbox_ids


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
