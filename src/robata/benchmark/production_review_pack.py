"""Create a blank human-review pack for production-shaped windows.

The pack is a review *input*, not a label generator.  Every action segment and
structured field is blank until a reviewer enters it.  Model predictions are
represented by separate status slots so they cannot accidentally become gold
labels through a JSON merge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ProductionReviewPackError(ValueError):
    """Raised when a cohort manifest cannot be converted to a review pack."""


_REQUIRED_LABEL_FIELDS = ("verb", "noun", "attributes", "location", "hand")
_MODEL_NAMES = ("wemm", "qwen", "mage")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionReviewPackError(f"{field} must be a non-empty string")
    return value.strip()


def build_review_pack(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build a blank, serialisable review pack from a cohort manifest."""

    if not isinstance(manifest, Mapping):
        raise ProductionReviewPackError("manifest must be an object")
    source = manifest.get("source")
    windows = manifest.get("windows")
    if not isinstance(source, Mapping):
        raise ProductionReviewPackError("manifest.source must be an object")
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes, bytearray)):
        raise ProductionReviewPackError("manifest.windows must be an array")
    source_path = _required_text(source.get("path"), "manifest.source.path")
    items: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise ProductionReviewPackError(f"window {index} must be an object")
        window_id = _required_text(window.get("window_id"), f"window {index}.window_id")
        camera_ids = window.get("camera_ids")
        if not isinstance(camera_ids, Sequence) or isinstance(camera_ids, (str, bytes, bytearray)):
            raise ProductionReviewPackError(f"window {window_id}.camera_ids must be an array")
        item = {
            "ordinal": window.get("ordinal", index),
            "window_id": window_id,
            "source_path": source_path,
            "start_seconds": window.get("start_seconds"),
            "end_seconds": window.get("end_seconds"),
            "camera_ids": [str(camera_id) for camera_id in camera_ids],
            "qa": {
                "status": "PENDING",
                "clip_marks": [],
                "per_camera_notes": {},
            },
            "gold": {
                "status": "PENDING_HUMAN_REVIEW",
                "segments": [],
                "label_fields": list(_REQUIRED_LABEL_FIELDS),
                "provenance": {
                    "source": None,
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "adjudication_status": "PENDING",
                },
            },
            "model_outputs": {
                model: {
                    "status": "NOT_RUN",
                    "predictions": [],
                    "artifact_reference": None,
                }
                for model in _MODEL_NAMES
            },
            "adjudication": {
                "status": "PENDING",
                "reviewer_a": None,
                "reviewer_b": None,
                "decision": None,
                "disagreement_notes": None,
            },
        }
        items.append(item)
    return {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_manifest_format": manifest.get("format"),
        "source": dict(source),
        "review_contract": {
            "boundary_fields": ["start_seconds", "end_seconds"],
            "structured_label_fields": list(_REQUIRED_LABEL_FIELDS),
            "observable_only": True,
            "one_action_per_segment_when_possible": True,
            "split_when_visible_action_changes": True,
            "model_outputs_are_not_gold": True,
            "allowed_gold_status": ["PENDING_HUMAN_REVIEW", "ACCEPTED", "REJECTED", "ABSTAIN"],
        },
        "items": items,
        "controls": {
            "labels_inferred": False,
            "model_predictions_copied": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "sha_or_digest_computed": False,
        },
    }


__all__ = ["ProductionReviewPackError", "build_review_pack"]
