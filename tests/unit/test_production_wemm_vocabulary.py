from __future__ import annotations

import pytest

from robata.benchmark.production_wemm_shadow import _ontology_profile
from robata.benchmark.production_wemm_vocabulary import (
    ProductionWemmVocabularyError,
    load_production_vocabulary,
    rank_production_vocabulary,
    run_production_wemm_vocabulary_shadow,
)


def _vocabulary() -> dict[str, object]:
    return {
        "format": "robata-production-coarse-vocabulary-owner-approval-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "OWNER_APPROVED_SCOPED_NON_GOLD",
        "owner_approved": True,
        "production_eligible": False,
        "official_gold": False,
        "accepted_as_gold": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "source": {"media_path": "data/source/sample-medium.mcap"},
        "vocabulary": {
            "verb_noun_pairs": [
                {
                    "verb": "pick up",
                    "verb_code": "pick_up",
                    "noun": "garment",
                    "canonical_label": "pick up garment",
                },
                {
                    "verb": "fold",
                    "verb_code": "fold",
                    "noun": "garment",
                    "canonical_label": "fold garment",
                },
            ]
        },
    }


def test_load_production_vocabulary_keeps_local_ids_and_text_variants() -> None:
    labels, provenance = load_production_vocabulary(_vocabulary())

    assert [label.label_id for label in labels] == ["pick_up", "fold"]
    assert labels[0].text_for("canonical") == "pick up garment"
    assert labels[0].text_for("natural") == "a person is picking up a garment"
    assert provenance["pair_count"] == 2
    assert provenance["production_eligible"] is False


def test_owner_canonical_surface_is_used_for_canonical_prototype() -> None:
    payload = _vocabulary()
    pairs = payload["vocabulary"]["verb_noun_pairs"]  # type: ignore[index]
    pairs[0]["canonical_label"] = "pick up the garment"  # type: ignore[index]
    labels, _ = load_production_vocabulary(payload)
    assert labels[0].text_for("canonical") == "pick up the garment"
    assert labels[0].text_for("verb_noun") == "verb: pick up; noun: garment"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_approved", False),
        ("production_eligible", True),
        ("official_gold", True),
        ("accepted_as_gold", True),
    ],
)
def test_load_rejects_unsafe_or_unapproved_vocabulary(field: str, value: object) -> None:
    payload = _vocabulary()
    payload[field] = value
    with pytest.raises(ProductionWemmVocabularyError):
        load_production_vocabulary(payload)


def test_load_rejects_epic_or_duplicate_catalog() -> None:
    payload = _vocabulary()
    payload["format"] = "robata-wemm-action-pair-catalog-v1"
    with pytest.raises(ProductionWemmVocabularyError, match="format"):
        load_production_vocabulary(payload)

    duplicate = _vocabulary()
    pairs = duplicate["vocabulary"]["verb_noun_pairs"]  # type: ignore[index]
    pairs.append(dict(pairs[0]))  # type: ignore[union-attr]
    with pytest.raises(ProductionWemmVocabularyError, match="duplicate"):
        load_production_vocabulary(duplicate)


def test_rank_is_deterministic_and_requires_every_label_embedding() -> None:
    labels, _ = load_production_vocabulary(_vocabulary())
    ranked = rank_production_vocabulary(
        labels,
        query_embedding=(1.0, 0.0),
        label_embeddings={"pick_up": (1.0, 0.0), "fold": (0.0, 1.0)},
        top_k=2,
    )
    assert [item.label_id for item in ranked] == ["pick_up", "fold"]
    assert ranked[0].rank == 1
    assert ranked[0].visual_score == pytest.approx(1.0)

    with pytest.raises(ProductionWemmVocabularyError, match="missing label embedding"):
        rank_production_vocabulary(
            labels,
            query_embedding=(1.0, 0.0),
            label_embeddings={"pick_up": (1.0, 0.0)},
        )


def test_epic_profile_is_not_used_for_declared_production_catalog() -> None:
    assert _ontology_profile({"format": "robata-production-provisional-vocabulary-v1"}) == (
        "PRODUCTION_VOCABULARY_FOR_SHADOW_ONLY"
    )
    assert _ontology_profile({"format": "robata-wemm-action-pair-catalog-v1"}) == (
        "PROVISIONAL_EPIC_ONTOLOGY_FOR_SHADOW_ONLY"
    )


def test_run_uses_production_label_ids_and_preserves_non_gold_controls(monkeypatch) -> None:
    import robata.benchmark.production_wemm_vocabulary as route

    class Group:
        frames = ("frame-0", "frame-1")

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {"camera_id": "cam_01", "window_id": "w00", "frame_count": 2}

    monkeypatch.setattr(
        route,
        "decode_production_windows",
        lambda manifest, **kwargs: {
            "cam_01": {"w00": Group()},
            "cam_02": {"w00": Group()},
        },
    )

    class Observation:
        def to_dict(self):
            return {"modality": "video", "frame_count": 2}

    class FakeBackend:
        def __init__(self, **kwargs):
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            # pick_up is aligned with the query; fold is orthogonal.
            return tuple((1.0, 0.0) if "pick up" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            self.observations.append(Observation())
            return ((1.0, 0.0),)

        def observation_payload(self):
            return [item.to_dict() for item in self.observations]

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FakeBackend)
    manifest = {
        "format": "robata-production-shaped-cohort-v1",
        "source": {"path": "data/source/sample-medium.mcap"},
        "windows": [{"ordinal": 0, "window_id": "w00", "start_seconds": 0.0, "end_seconds": 1.0}],
    }
    report = run_production_wemm_vocabulary_shadow(
        manifest,
        vocabulary=_vocabulary(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
    )
    window = report["windows"][0]
    predictions = window["model"]["predictions"]
    assert predictions[0]["label_id"] == "pick_up"
    assert predictions[0]["verb_code"] == "pick_up"
    assert predictions[0]["label_text"] == "pick up garment"
    assert "verb_id" not in predictions[0]
    assert "noun_id" not in predictions[0]
    assert report["vocabulary"]["epic_ontology_used"] is False
    assert report["controls"]["existing_mapper_invoked"] is False
    assert report["controls"]["predictions_are_gold"] is False
