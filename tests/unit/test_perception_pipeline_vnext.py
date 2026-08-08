"""End-to-end unit coverage for the stream-oriented perception graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from robata.contracts.cameras import CameraId
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.perception_stream import (
    MageObservation,
    PerceptionContextManifest,
    RefineReason,
    RefineTargetField,
)
from robata.perception.fusion import (
    PerceptionFusionEngine,
    PerceptionFusionPolicy,
)
from robata.perception.pipeline import (
    LocalPerceptionArtifactStore,
    PerceptionStage,
    StreamPerceptionPipeline,
)
from robata.perception.projectors import EventProjector, EvidenceProjector, QaProjector
from robata.perception.tracking import EventTrackPolicy, EventTrackReconciler, EventTrackState
from robata.runtime.observability import RuntimeProfileRecorder
from tests.support.perception_stream import make_context, make_media_health, make_observation


@dataclass
class _FakeMageProvider:
    observations: dict[str, MageObservation]
    calls: list[str] = field(default_factory=list)

    def observe(self, context: PerceptionContextManifest) -> MageObservation:
        digest = context.context_manifest_semantic_sha256
        self.calls.append(digest)
        return self.observations[digest]


def _pipeline(
    provider: _FakeMageProvider,
    tmp_path,
    *,
    runtime_observer: RuntimeProfileRecorder | None = None,
) -> StreamPerceptionPipeline:
    return StreamPerceptionPipeline(
        provider=provider,
        qa_projector=QaProjector(),
        event_projector=EventProjector(),
        evidence_projector=EvidenceProjector(),
        reconciler=EventTrackReconciler(
            EventTrackPolicy(version="event-track-policy-vnext-v1", max_merge_gap_ns=1)
        ),
        fusion_engine=PerceptionFusionEngine(
            PerceptionFusionPolicy(
                version="perception-fusion-policy-v1",
                minimum_observable_cameras=2,
                minimum_supporting_cameras=2,
                final_confidence_threshold=0.5,
                boundary_refine_threshold=0.65,
            )
        ),
        refine_policy_version="perception-refine-policy-v1",
        refine_prompt_version="perception-refine-prompt-v1",
        artifact_sink=LocalPerceptionArtifactStore(tmp_path / "perception-cas"),
        runtime_observer=runtime_observer,
    )


def test_one_observation_per_segment_drives_all_logical_products_and_one_track(tmp_path) -> None:
    selected = (CameraId.CAM_01, CameraId.CAM_02)
    contexts = (
        make_context(
            start_ns=0,
            end_ns=8_000_000_000,
            segment_ordinal=0,
            selected_cameras=selected,
        ),
        make_context(
            start_ns=8_000_000_000,
            end_ns=16_000_000_000,
            segment_ordinal=1,
            selected_cameras=selected,
        ),
        make_context(
            start_ns=16_000_000_000,
            end_ns=24_000_000_000,
            segment_ordinal=2,
            selected_cameras=selected,
        ),
    )
    observations = (
        make_observation(
            context=contexts[0],
            local_ref="o0",
            action_start_ns=7_000_000_000,
            action_end_ns=8_000_000_000,
            continues_after_context=True,
            inference_artifact_seed="artifact-0",
        ),
        make_observation(
            context=contexts[1],
            local_ref="o1",
            action_start_ns=8_000_000_000,
            action_end_ns=16_000_000_000,
            started_before_context=True,
            continues_after_context=True,
            inference_artifact_seed="artifact-1",
        ),
        make_observation(
            context=contexts[2],
            local_ref="o2",
            action_start_ns=16_000_000_000,
            action_end_ns=17_000_000_000,
            started_before_context=True,
            continues_after_context=False,
            inference_artifact_seed="artifact-2",
        ),
    )
    provider = _FakeMageProvider(
        {
            context.context_manifest_semantic_sha256: observation
            for context, observation in zip(contexts, observations, strict=True)
        }
    )

    result = _pipeline(provider, tmp_path).run(
        contexts=contexts,
        media_health=tuple(make_media_health(context) for context in contexts),
    )

    assert provider.calls == [item.context_manifest_semantic_sha256 for item in contexts]
    assert result.normal_model_call_count == 3
    assert result.refinement_model_call_count == 0
    assert result.total_model_call_count == 3
    assert len(result.contexts) == 3
    assert all(item.qa_projection for item in result.contexts)
    assert all(item.event_projection for item in result.contexts)
    assert all(item.evidence_projection for item in result.contexts)

    assert len(result.event_tracks) == 1
    track = result.event_tracks[0]
    assert track.state is EventTrackState.CLOSED
    assert len(track.source_hypotheses) == 3
    assert track.interval.start_ns == 7_000_000_000
    assert track.interval.end_ns == 17_000_000_000

    assert len(result.fusion_decisions) == 1
    decision = result.fusion_decisions[0]
    assert decision.selected_camera_count == 2
    assert decision.observable_camera_count == 2
    assert decision.supporting_camera_count == 2
    assert decision.confidence == pytest.approx(1.0)
    assert decision.resolved
    assert not result.refine_requests

    stages = {item.stage: item for item in result.stage_measurements}
    assert stages[PerceptionStage.MEDIA_SCAN].invocation_count == 3
    assert stages[PerceptionStage.PERCEPTION_OBSERVE].invocation_count == 3
    assert stages[PerceptionStage.OBSERVATION_PROJECT].invocation_count == 3
    assert stages[PerceptionStage.TEMPORAL_RECONCILE].invocation_count == 3
    assert stages[PerceptionStage.FUSION].invocation_count == 1
    assert stages[PerceptionStage.PERCEPTION_REFINE].invocation_count == 0

    terminal = result.terminal_artifacts
    assert terminal is not None
    assert len(terminal.event_tracks) == 1
    assert len(terminal.fusion_decisions) == 1
    assert terminal.refine_requests == ()

    store = LocalPerceptionArtifactStore(tmp_path / "perception-cas")
    assert store.read(
        kind="event-track", logical_key=track.event_track_key
    ) == canonical_json_bytes(track)
    assert store.read(
        kind="fusion-decision", logical_key=decision.fusion_key
    ) == canonical_json_bytes(decision)
    terminal_document = json.loads(
        store.read(
            kind="perception-terminal-manifest",
            logical_key=terminal.terminal_manifest.logical_key,
        )
    )
    assert terminal_document["normal_model_call_count"] == 3
    assert terminal_document["refinement_model_call_count"] == 0
    assert len(terminal_document["contexts"]) == 3
    assert terminal_document["event_tracks"] == [
        {
            "exact_sha256": terminal.event_tracks[0].exact_sha256,
            "kind": "event-track",
            "logical_key": track.event_track_key,
        }
    ]

    persisted = tuple((tmp_path / "perception-cas").rglob("*.json"))
    assert len(persisted) == 24
    assert len(tuple((tmp_path / "perception-cas" / "_logical").rglob("*.ref"))) == 24


def test_projection_must_be_temporally_reconciled_before_finalization(tmp_path) -> None:
    context = make_context(selected_cameras=(CameraId.CAM_01, CameraId.CAM_02))
    observation = make_observation(context=context)
    provider = _FakeMageProvider({context.context_manifest_semantic_sha256: observation})
    session = _pipeline(provider, tmp_path).open_session()
    health = session.scan_media(lambda: make_media_health(context))

    projected = session.project_precomputed(
        context=context,
        media_health=health,
        observation=observation,
        observation_elapsed_seconds=0.0,
    )
    with pytest.raises(RuntimeError, match="pending temporal reconciliation"):
        session.finalize()

    outcome = session.reconcile_projected(projected)
    assert outcome.context.context_manifest_key == context.context_manifest_key
    assert session.finalize().normal_model_call_count == 1


def test_shadow_gate_does_not_suppress_normal_perception(tmp_path) -> None:
    context = make_context(selected_cameras=(CameraId.CAM_01, CameraId.CAM_02))
    observation = make_observation(context=context)
    assert observation.cognition_gate.would_admit is False
    provider = _FakeMageProvider({context.context_manifest_semantic_sha256: observation})

    result = _pipeline(provider, tmp_path).run(
        contexts=(context,),
        media_health=(make_media_health(context),),
    )

    assert len(provider.calls) == 1
    assert len(result.contexts[0].event_projection.hypotheses) == 1
    assert len(result.fusion_decisions) == 1


def test_overlapping_focus_segments_fail_before_any_model_call(tmp_path) -> None:
    first = make_context(start_ns=0, end_ns=8_000_000_000, segment_ordinal=0)
    second = make_context(
        start_ns=7_000_000_000,
        end_ns=15_000_000_000,
        segment_ordinal=1,
    )
    provider = _FakeMageProvider(
        {
            first.context_manifest_semantic_sha256: make_observation(context=first),
            second.context_manifest_semantic_sha256: make_observation(
                context=second,
                action_start_ns=8_000_000_000,
                action_end_ns=9_000_000_000,
            ),
        }
    )

    with pytest.raises(ValueError, match="must not overlap"):
        _pipeline(provider, tmp_path).run(
            contexts=(first, second),
            media_health=(make_media_health(first), make_media_health(second)),
        )

    assert not provider.calls


def test_pipeline_emits_provider_neutral_hotspot_spans(tmp_path) -> None:
    context = make_context(selected_cameras=(CameraId.CAM_01, CameraId.CAM_02))
    provider = _FakeMageProvider(
        {context.context_manifest_semantic_sha256: make_observation(context=context)}
    )
    recorder = RuntimeProfileRecorder()

    _pipeline(provider, tmp_path, runtime_observer=recorder).run(
        contexts=(context,),
        media_health=(make_media_health(context),),
    )

    names = tuple(span.name for span in recorder.snapshot().spans)
    assert names.count("perception.observe") == 1
    assert names.count("perception.project") == 1
    assert names.count("perception.temporal_reconcile") == 1
    assert names.count("perception.fusion") == 1
    assert names.count("perception.finalize") == 1
    assert "perception.refine" not in names


def test_pipeline_rejects_cross_recording_or_gapped_contexts_before_model_calls(
    tmp_path,
) -> None:
    first = make_context(start_ns=0, end_ns=8_000_000_000, segment_ordinal=0)
    other_recording = make_context(
        start_ns=8_000_000_000,
        end_ns=16_000_000_000,
        segment_ordinal=1,
        source_recording_key="recording:other",
        source_recording_seed="other",
    )
    provider = _FakeMageProvider({})
    with pytest.raises(ValueError, match="one recording and policy"):
        _pipeline(provider, tmp_path).run(
            contexts=(first, other_recording),
            media_health=(make_media_health(first), make_media_health(other_recording)),
        )
    assert provider.calls == []

    gapped = make_context(
        start_ns=9_000_000_000,
        end_ns=17_000_000_000,
        segment_ordinal=1,
    )
    with pytest.raises(ValueError, match="contiguous partition"):
        _pipeline(provider, tmp_path).run(
            contexts=(first, gapped),
            media_health=(make_media_health(first), make_media_health(gapped)),
        )
    assert provider.calls == []


def test_each_refinement_reason_becomes_one_bounded_single_purpose_request(tmp_path) -> None:
    selected = (CameraId.CAM_01, CameraId.CAM_02)
    context = make_context(selected_cameras=selected)
    observation = make_observation(
        context=context,
        start_confidence=0.2,
        end_confidence=0.3,
        contradicting_cameras=(CameraId.CAM_02,),
        unusable_qa_cameras=(CameraId.CAM_01,),
    )
    provider = _FakeMageProvider({context.context_manifest_semantic_sha256: observation})

    result = _pipeline(provider, tmp_path).run(
        contexts=(context,),
        media_health=(make_media_health(context),),
    )

    by_reason = {request.reason: request for request in result.refine_requests}
    assert set(by_reason) == {
        RefineReason.BOUNDARY,
        RefineReason.CONFLICT,
        RefineReason.QA,
    }
    assert by_reason[RefineReason.BOUNDARY].target_fields == (
        RefineTargetField.END_BOUNDARY,
        RefineTargetField.START_BOUNDARY,
    )
    assert by_reason[RefineReason.CONFLICT].target_fields == (RefineTargetField.CAMERA_RELATION,)
    assert by_reason[RefineReason.QA].target_fields == (RefineTargetField.SEMANTIC_QA,)
    assert len({request.refine_request_key for request in result.refine_requests}) == 3
