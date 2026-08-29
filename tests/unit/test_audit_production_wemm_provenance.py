from __future__ import annotations

from pathlib import Path

from scripts.audit_production_wemm_provenance import classify


def _sidecar(*, epic: bool = False, mapper: bool = False) -> dict[str, object]:
    return {
        "format": "robata-production-wemm-vocabulary-shadow-v1",
        "production_eligible": False,
        "source": {"window_count": 1, "camera_count": 6},
        "model": {"label_variant": "canonical"},
        "vocabulary": {
            "profile": "PRODUCTION_OWNER_APPROVED_COARSE_VOCABULARY",
            "epic_ontology_used": epic,
            "mapper_used": mapper,
        },
    }


def test_classify_accepts_explicit_terra_production_sidecar() -> None:
    row = classify(Path("production.json"), _sidecar())
    assert row["classification"] == "PRODUCTION_TERRA_VOCABULARY"
    assert row["epic_ontology_used"] is False
    assert row["mapper_used"] is False


def test_classify_quarantines_sidecar_with_epic_or_mapper_marker() -> None:
    assert classify(Path("production.json"), _sidecar(epic=True))["classification"] == "QUARANTINED"
    assert (
        classify(Path("production.json"), _sidecar(mapper=True))["classification"] == "QUARANTINED"
    )


def test_classify_separates_legacy_epic_and_legacy_production_shapes() -> None:
    epic = {
        "format": "robata-production-wemm-shadow-v1",
        "ontology": {
            "format": "robata-wemm-action-pair-catalog-v1",
            "source": "EPIC-KITCHENS action ontology graph",
            "profile": "PROVISIONAL_EPIC_ONTOLOGY_FOR_SHADOW_ONLY",
        },
    }
    legacy_production = {
        "format": "robata-production-wemm-shadow-v1",
        "ontology": {
            "format": "robata-production-provisional-vocabulary-v1",
            "source": "agent-surrogate-production-vocabulary",
            "profile": "PROVISIONAL_EPIC_ONTOLOGY_FOR_SHADOW_ONLY",
        },
    }
    assert classify(Path("epic.json"), epic)["classification"] == "EPIC_DERIVED_QUARANTINED"
    assert (
        classify(Path("legacy.json"), legacy_production)["classification"]
        == "LEGACY_PRODUCTION_VOCABULARY_QUARANTINED"
    )
