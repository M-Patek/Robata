from __future__ import annotations

import pytest

from robata.retrieval import (
    ClipArtifact,
    ClipManifest,
    EventIndex,
    EventIndexError,
    RetrievalCapabilityError,
    RetrievalQuery,
    RetrievalService,
    SearchFilter,
)


def _revision(
    revision_id: str,
    *,
    event_id: str = "event-1",
    action_type: str = "grasp",
    label: str = "cup",
    start_ns: int = 10,
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_revision_id": revision_id,
        "mcap_id": "mcap-1",
        "start_ns": start_ns,
        "end_ns": start_ns + 10,
        "action_type": action_type,
        "active_hand": "RIGHT",
        "object_class_id": f"class-{label}",
        "object_label": label,
        "confidence_value": confidence,
        "camera_statuses": {
            "cam_01": "SUPPORTING",
            "cam_02": "SUPPORTING",
            "cam_03": "PARTIAL",
        },
        "usable_camera_count": 3,
        "text": f"right hand {action_type} {label}",
    }


def test_revisions_are_append_only_and_selection_is_separate() -> None:
    index = EventIndex()
    first = _revision("revision-1")
    second = _revision("revision-2", confidence=0.95)

    index.update_index(first, select=True)
    index.update_index(first, select=True)
    assert index.revision_count == 1
    assert len(index.selection_history("event-1")) == 2

    index.update_index(second)
    index.select_revision(
        event_id="event-1",
        revision_id="revision-2",
        selection_decision_id="decision-3",
        sequence=3,
    )
    assert index.current_revision("event-1")["event_revision_id"] == "revision-2"

    changed = dict(first)
    changed["confidence_value"] = 0.1
    with pytest.raises(EventIndexError, match="cannot be mutated"):
        index.update_index(changed)


def test_structured_filters_precede_lexical_semantic_ranking() -> None:
    index = EventIndex()
    index.update_index(_revision("revision-cup"), select=True)
    index.update_index(
        _revision(
            "revision-bottle",
            event_id="event-2",
            label="bottle",
            start_ns=30,
        ),
        select=True,
    )

    query = RetrievalQuery(
        filters=SearchFilter(
            action_type="grasp",
            active_hand="RIGHT",
            object_label="cup",
            min_confidence=0.8,
            min_usable_camera_count=2,
        ),
        semantic_query="right grasp cup",
        limit=10,
        offset=0,
    )
    result = index.query_index(query)

    assert result.total == 1
    assert result.items[0].event_revision_id == "revision-cup"
    assert result.items[0].semantic_score == 1.0


def test_current_selection_hides_superseded_revision() -> None:
    index = EventIndex()
    index.build_index(
        {
            "event_revisions": [
                _revision("revision-1"),
                _revision("revision-2", confidence=0.95),
            ],
            "current_selections": [
                {
                    "event_id": "event-1",
                    "selected_revision_id": "revision-2",
                    "selection_decision_id": "decision-1",
                    "selection_sequence": 1,
                }
            ],
        }
    )

    current = index.query_index(RetrievalQuery())
    all_revisions = index.query_index(
        RetrievalQuery(filters=SearchFilter(require_current_revision=False))
    )
    assert [item.event_revision_id for item in current.items] == ["revision-2"]
    assert {item.event_revision_id for item in all_revisions.items} == {
        "revision-1",
        "revision-2",
    }


def _clip_manifest() -> ClipManifest:
    return ClipManifest(
        clip_manifest_id="clip-manifest-1",
        event_id="event-1",
        event_revision_id="revision-1",
        mcap_id="mcap-1",
        alignment_id="alignment-1",
        camera_mapping_run_id="mapping-1",
        start_ns=10,
        end_ns=20,
        source_stream_artifacts=("source-1",),
        source_manifest_digest="b" * 64,
        clip_artifacts=(
            ClipArtifact(
                artifact_id="clip-1",
                camera_id="cam_01",
                uri="file:///clips/cam_01.mp4",
                sha256="a" * 64,
                bytes=100,
                media_type="video/mp4",
                format="mp4",
                trim_policy_version="1.0",
                effective_start_ns=10,
                effective_end_ns=20,
            ),
        ),
        extractor_version="1.0",
        created_at="2026-07-19T12:00:00Z",
    )


def test_optional_capabilities_fail_closed_and_registered_data_is_reused() -> None:
    service = RetrievalService()
    with pytest.raises(RetrievalCapabilityError, match="embedding reranking"):
        service.semantic_search(RetrievalQuery(), embedding_vector=[0.1, 0.2])
    with pytest.raises(RetrievalCapabilityError, match="clip extraction"):
        service.extract_clip("event-1")

    manifest = _clip_manifest()
    service.register_clip_manifest(manifest, camera_mask=["cam_01"])
    assert service.extract_clip("event-1", ["cam_01"]) == manifest

    service.register_provenance("event-1", {"source": "mcap-1", "frames": ["frame-1"]})
    assert service.resolve_provenance("event-1")["source"] == "mcap-1"
    with pytest.raises(EventIndexError, match="immutable"):
        service.register_provenance("event-1", {"source": "different"})


def test_selection_rejects_wrong_ownership_and_sequence() -> None:
    index = EventIndex()
    index.update_index(_revision("revision-1"))
    with pytest.raises(EventIndexError, match="owned revision"):
        index.select_revision(
            event_id="event-2",
            revision_id="revision-1",
            selection_decision_id="decision-1",
            sequence=1,
        )
    with pytest.raises(EventIndexError, match="selection_sequence"):
        index.select_revision(
            event_id="event-1",
            revision_id="revision-1",
            selection_decision_id="decision-1",
            sequence=2,
        )
