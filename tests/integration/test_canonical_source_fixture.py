from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from robata.application.canonical.models import (
    CANONICAL_OFFLINE_PIPELINE_VERSION,
    CanonicalOfflineRunStatus,
)
from robata.application.canonical.source_fixture import (
    CanonicalSourceBundle,
    CanonicalSourceFixtureError,
    load_canonical_source_fixture,
)
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.cameras import CAMERA_IDS
from tests.integration.test_canonical_offline import (
    NOW,
    NOW_TEXT,
    _claim_bytes,
    _harness,
    _uuid,
)

SOURCE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical" / "source-recording.json"


def _run_bundle(
    bundle: CanonicalSourceBundle,
    *,
    logical_registry_root: Path,
    run_id: str,
):
    harness = _harness(_claim_bytes, logical_registry_root=logical_registry_root)
    processing_run = CanonicalProcessingRunContext.fresh(
        run_id=run_id,
        recording_identity=bundle.admitted_context.recording_identity,
        mcap_id=bundle.admitted_context.ready_manifest.mcap_id,
        pipeline_version=CANONICAL_OFFLINE_PIPELINE_VERSION,
        config_sha256=harness.execution_policy.semantic_sha256,
        started_at=NOW_TEXT,
    )

    first = asyncio.run(
        harness.pipeline.run(
            processing_run=processing_run,
            admitted_context=bundle.admitted_context,
            requested_interval=bundle.requested_interval,
            sampling_plan=bundle.sampling_plan,
            frame_index=bundle.frame_index,
            artifact_resolver=bundle.resolve_artifact,
        )
    )
    replay = asyncio.run(
        harness.pipeline.run(
            processing_run=processing_run,
            admitted_context=bundle.admitted_context,
            requested_interval=bundle.requested_interval,
            sampling_plan=bundle.sampling_plan,
            frame_index=bundle.frame_index,
            artifact_resolver=bundle.resolve_artifact,
        )
    )
    return harness, first, replay


def test_source_fixture_runs_canonical_path_and_replays_without_side_effects(
    tmp_path: Path,
) -> None:
    bundle = load_canonical_source_fixture(SOURCE_FIXTURE, clock=lambda: NOW)

    harness, first, replay = _run_bundle(
        bundle,
        logical_registry_root=tmp_path,
        run_id=_uuid(45_001),
    )

    assert tuple(bundle.frame_index.cameras.keys()) == CAMERA_IDS
    assert all(
        bundle.resolve_artifact(camera_id, frame) is not None
        for camera_id in CAMERA_IDS
        for frame in bundle.frame_index.cameras[camera_id].frames
    )
    assert first.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert first.output_decision is not None
    assert first.output_decision.production_eligible is False
    assert first.identity_result is None
    assert replay.status is CanonicalOfflineRunStatus.SUCCEEDED
    assert replay.processing_run == first.processing_run
    assert replay.run_memberships == first.run_memberships
    assert replay.part_results == first.part_results
    assert replay.output_decision == first.output_decision
    assert replay.hypotheses == first.hypotheses
    assert replay.adapter_infer_calls == 0
    assert harness.pipeline.adapter.infer_calls == first.adapter_infer_calls


def test_source_fixture_rejects_undecodable_png_before_pipeline(
    tmp_path: Path,
) -> None:
    document = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    document["cameras"]["cam_01"]["frames"][0]["payload_base64"] = "iVBORw0KGgo="
    invalid_source = tmp_path / "invalid-source.json"
    invalid_source.write_text(
        json.dumps(document, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(CanonicalSourceFixtureError, match="cannot be decoded"):
        load_canonical_source_fixture(invalid_source, clock=lambda: NOW)
