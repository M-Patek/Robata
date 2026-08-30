from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_wemm_epic_retrieval as runner  # noqa: E402


def _args(**overrides):
    values = {
        "frame_count": 4,
        "top_k": 10,
        "text_batch_size": 16,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_runner_rejects_unbounded_frames_and_truncated_metric_rankings() -> None:
    with pytest.raises(ValueError, match="between 2 and 64"):
        runner._validate_run_args(_args(frame_count=65))
    with pytest.raises(ValueError, match="at least 10"):
        runner._validate_run_args(_args(top_k=9))


def test_manifest_keys_preserve_zero_and_reject_duplicates() -> None:
    rows = [{"uid": 0}, {"uid": 1}]
    assert runner._validate_unique_row_keys(rows) == ("0", "1")
    with pytest.raises(ValueError, match="duplicate manifest row key"):
        runner._validate_unique_row_keys([{"uid": "same"}, {"uid": "same"}])


def test_ordinal_alignment_takes_precedence_over_repeated_video_aliases() -> None:
    rows = [
        {"uid": None, "video_id": "clip"},
        {"uid": None, "video_id": "clip"},
        {"uid": None, "video_id": "other"},
    ]
    predictions = [
        {"video_id": "clip", "ordinal": 0, "prediction": {"verb": "open"}},
        {"video_id": "clip", "ordinal": 1, "prediction": {"verb": "close"}},
        {"video_id": "other", "ordinal": 2, "prediction": {"verb": "wash"}},
    ]
    aligned, provenance = runner._align_text_predictions(
        rows,
        predictions,
        {
            "provided": True,
            "alignment": None,
            "alignment_explicit": False,
            "label_blind_declared": True,
            "label_bearing_fields_present": False,
            "source": "independent-qwen-export",
            "split": "train-derived",
        },
    )
    assert [item["prediction"]["verb"] for item in aligned] == [
        "open",
        "close",
        "wash",
    ]
    assert provenance["alignment"] == "ordinal"
    assert provenance["ordinal_consistent"] is True
    assert provenance["quality_valid"] is True


def test_explicit_ordinal_alignment_allows_ordered_rows_without_ordinal_column() -> None:
    rows = [{"uid": None}, {"uid": None}]
    predictions = [
        {"prediction": {"verb": "open"}},
        {"prediction": {"verb": "close"}},
    ]
    aligned, provenance = runner._align_text_predictions(
        rows,
        predictions,
        {
            "provided": True,
            "alignment": "ordinal",
            "alignment_explicit": True,
            "label_blind_declared": True,
            "label_bearing_fields_present": False,
            "source": "independent-export",
            "split": "validation",
        },
    )
    assert [item["prediction"]["verb"] for item in aligned] == ["open", "close"]
    assert provenance["quality_valid"] is True


def test_text_report_target_fields_are_not_consumed_as_prediction() -> None:
    rows = [{"uid": "a"}]
    with pytest.raises(ValueError, match="target fields"):
        runner._align_text_predictions(
            rows,
            [{"uid": "a", "prediction": {"verb": "open", "verb_class": 1}}],
            {
                "provided": True,
                "alignment": None,
                "alignment_explicit": False,
                "label_blind_declared": True,
                "label_bearing_fields_present": False,
                "source": "independent",
                "split": "train",
            },
        )


def test_pair_catalog_provenance_marks_bare_lists_unverified(monkeypatch) -> None:
    payload = {
        "action_pairs": [[0, 0]],
        "provenance": {"source": "train", "split": "train", "label_blind": True},
    }
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: json.dumps(payload),
    )
    pairs, verified = runner._pair_file_payload(Path("pairs.json"))
    assert pairs == ((0, 0),)
    assert verified is True

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: json.dumps([[0, 0]]),
    )
    _pairs, verified = runner._pair_file_payload(Path("pairs.json"))
    assert verified is False


def test_pair_catalog_provenance_accepts_participant_disjoint_train_split(monkeypatch) -> None:
    payload = {
        "action_pairs": [[0, 0]],
        "provenance": {
            "source": "EPIC-KITCHENS validation train projection",
            "split": "train_disjoint_from_dev27",
            "label_blind": True,
        },
    }
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: json.dumps(payload),
    )
    pairs, verified = runner._pair_file_payload(Path("pairs.json"))
    assert pairs == ((0, 0),)
    assert verified is True


@pytest.mark.parametrize("label_blind", [None, False, "true"])
def test_pair_catalog_requires_explicit_boolean_label_blind(monkeypatch, label_blind) -> None:
    provenance = {"source": "EPIC train", "split": "train"}
    if label_blind is not None:
        provenance["label_blind"] = label_blind
    payload = {"action_pairs": [[0, 0]], "provenance": provenance}
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: json.dumps(payload),
    )
    pairs, verified = runner._pair_file_payload(Path("pairs.json"))
    assert pairs == ((0, 0),)
    assert verified is False


def test_pair_file_provenance_preserves_declared_fields_without_digest(monkeypatch) -> None:
    payload = {
        "format": "robata-wemm-action-pair-catalog-v1",
        "action_pairs": [[0, 0]],
        "provenance": {
            "source": "EPIC ontology",
            "split": "ontology",
            "label_blind": True,
        },
    }
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: json.dumps(payload),
    )
    metadata = runner._pair_file_provenance(Path("pairs.json"))
    assert metadata["format"] == "robata-wemm-action-pair-catalog-v1"
    assert metadata["source"] == "EPIC ontology"
    assert metadata["split"] == "ontology"
    assert metadata["label_blind"] is True
    assert metadata["verified"] is True
    assert "sha" not in metadata and "digest" not in metadata


def test_manifest_rejects_escape_and_invalid_intervals(monkeypatch) -> None:
    current: dict[str, object] = {}
    original_read_text = Path.read_text

    def read_manifest(self, *args, **kwargs):
        if self.name == "manifest.jsonl":
            return json.dumps(current) + "\n"
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_manifest)
    root = Path(".")
    manifest = Path("manifest.jsonl")

    def load(row):
        current.clear()
        current.update(row)
        return runner._load_manifest(manifest, root, None)

    with pytest.raises(ValueError, match="escapes dataset root"):
        load({"video_relpath": "../outside.mp4", "start_seconds": 0, "end_seconds": 1})
    with pytest.raises(ValueError, match="must be finite"):
        load(
            {
                "video_relpath": "scripts/run_wemm_epic_retrieval.py",
                "start_seconds": "NaN",
                "end_seconds": 1,
            }
        )
    with pytest.raises(ValueError, match="non-negative"):
        load(
            {
                "video_relpath": "scripts/run_wemm_epic_retrieval.py",
                "start_seconds": -1,
                "end_seconds": 1,
            }
        )
    with pytest.raises(ValueError, match="greater than"):
        load(
            {
                "video_relpath": "scripts/run_wemm_epic_retrieval.py",
                "start_seconds": 2,
                "end_seconds": 2,
            }
        )


def test_manifest_canonicalizes_interval_and_resolved_path(monkeypatch) -> None:
    row = {
        "uid": "row-a",
        "video_relpath": "scripts\\run_wemm_epic_retrieval.py",
        "start_seconds": "0.25",
        "end_seconds": 1,
    }
    original_read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: (
            json.dumps(row) + "\n"
            if self.name == "manifest.jsonl"
            else original_read_text(self, *args, **kwargs)
        ),
    )
    rows = runner._load_manifest(Path("manifest.jsonl"), Path("."), None)
    assert rows[0]["start_seconds"] == 0.25
    assert rows[0]["end_seconds"] == 1.0
    assert Path(rows[0]["video_path"]) == Path("scripts/run_wemm_epic_retrieval.py").resolve()


def test_catalog_modes_are_mutually_exclusive_and_fallback_is_flagged() -> None:
    rows = [{"ground_truth": {"verb_class": 0, "noun_class": 0}}]
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner._catalog_pairs(
            rows,
            {0: "open"},
            {0: "door"},
            pair_file=Path("pairs.json"),
            full_cartesian=True,
        )
    pairs, source, uses_labels = runner._catalog_pairs(
        rows,
        {0: "open"},
        {0: "door"},
        pair_file=None,
        full_cartesian=False,
    )
    assert pairs == ((0, 0),)
    assert source == "development_manifest_pairs"
    assert uses_labels is True


def test_catalog_target_coverage_marks_unavailable_eval_action_pairs() -> None:
    labels = runner.build_joint_action_catalog(
        verb_table_or_entries={0: "open"},
        noun_table_or_entries={0: "door"},
        action_pairs=[(0, 0)],
    )
    coverage = runner._catalog_ground_truth_coverage(
        [
            {"ground_truth": {"verb_class": 0, "noun_class": 0}},
            {"ground_truth": {"verb_class": 1, "noun_class": 0}},
        ],
        labels,
    )
    assert coverage == {
        "unique_manifest_target_pairs": 2,
        "catalog_pairs": 1,
        "covered_target_pairs": 1,
        "missing_target_pairs": [[1, 0]],
        "complete": False,
    }


def test_runner_streams_decode_and_closes_backend_on_success(monkeypatch) -> None:
    events: list[str] = []

    class FakeBackend:
        variant = "2B"
        supported_dimensions = (2,)
        identity = "fake-wemm"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.closed = False
            self.video_calls = 0

        def encode_video_frames(self, groups, *, metadata_groups=None):
            group = list(groups)
            events.append(f"encode{self.video_calls}")
            self.video_calls += 1
            assert len(group) == 1
            return ((1.0, 0.0),)

        def encode_texts(self, texts, *, batch_size):
            events.append("text")
            return tuple((1.0, 0.0) for _ in texts)

        def observation_payload(self):
            return []

        def close(self):
            self.closed = True
            events.append("close")

    rows = [
        {"uid": "a", "video_path": "a.mp4"},
        {"uid": "b", "video_path": "b.mp4"},
    ]
    monkeypatch.setattr(runner, "WemmEmbeddingBackend", FakeBackend)
    monkeypatch.setattr(runner, "_load_manifest", lambda *args, **kwargs: rows)
    monkeypatch.setattr(runner, "_read_class_table", lambda path: {0: "open"})
    monkeypatch.setattr(
        runner,
        "_catalog_pairs",
        lambda *args, **kwargs: (((0, 0),), "full_cartesian", False),
    )
    monkeypatch.setattr(
        runner,
        "_load_and_align_text_predictions",
        lambda *args, **kwargs: ([], {"provided": False, "quality_valid": False}),
    )

    def decode(*args, **kwargs):
        index = len([item for item in events if item.startswith("decode")])
        events.append(f"decode{index}")
        return ([f"frame-{index}-0", f"frame-{index}-1"], {"fps": 1.0})

    monkeypatch.setattr(runner, "_decode_interval", decode)
    output = Path(".tmp_wemm_runner_stream_report.json")
    args = _args(
        manifest=Path("manifest.jsonl"),
        dataset_root=Path("."),
        model_dir=Path("model"),
        verb_classes=Path("verbs.csv"),
        noun_classes=Path("nouns.csv"),
        output=output,
        text_report=None,
        catalog_pairs=None,
        full_cartesian=True,
        max_cases=None,
        dimension=2,
        intervention="normal",
        device="cpu",
        min_score=0.0,
        min_margin=0.0,
        visual_weight=1.0,
        text_weight=0.0,
    )
    try:
        report = runner.run(args)
    finally:
        output.unlink(missing_ok=True)
    assert report["input"]["catalog_provenance_verified"] is True
    assert report["input"]["catalog_provenance"]["source"] == "verb and noun class tables"
    assert report["quality_validity"]["mode_metrics_valid"] == {
        "visual": True,
        "text": False,
        "hybrid": False,
    }
    assert report["quality_validity"]["overall_metrics_valid"] is False
    assert report["controls"]["quality_metrics_valid"] is False
    assert len(report["input"]["row_input_audit"]) == 2
    assert report["input"]["row_input_audit"][0]["row_key"] == "a"
    assert report["input"]["row_input_audit"][0]["fps"] == 1.0
    assert report["input"]["video_interval_audit"] == report["input"]["row_input_audit"]
    assert events[:4] == ["decode0", "encode0", "decode1", "encode1"]
    assert events[-1] == "close"


def test_runner_closes_backend_when_decode_fails(monkeypatch) -> None:
    closed: list[bool] = []

    class FakeBackend:
        variant = "2B"
        supported_dimensions = (2,)
        identity = "fake-wemm"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self):
            closed.append(True)

    rows = [{"uid": "a", "video_path": "a.mp4"}]
    monkeypatch.setattr(runner, "WemmEmbeddingBackend", FakeBackend)
    monkeypatch.setattr(runner, "_load_manifest", lambda *args, **kwargs: rows)
    monkeypatch.setattr(runner, "_read_class_table", lambda path: {0: "open"})
    monkeypatch.setattr(
        runner,
        "_catalog_pairs",
        lambda *args, **kwargs: (((0, 0),), "full_cartesian", False),
    )
    monkeypatch.setattr(
        runner,
        "_load_and_align_text_predictions",
        lambda *args, **kwargs: ([], {"provided": False, "quality_valid": False}),
    )
    monkeypatch.setattr(
        runner,
        "_decode_interval",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    args = _args(
        manifest=Path("manifest.jsonl"),
        dataset_root=Path("."),
        model_dir=Path("model"),
        verb_classes=Path("verbs.csv"),
        noun_classes=Path("nouns.csv"),
        output=Path(".tmp_wemm_runner_failure_report.json"),
        text_report=None,
        catalog_pairs=None,
        full_cartesian=True,
        max_cases=None,
        dimension=2,
        intervention="normal",
        device="cpu",
        min_score=0.0,
        min_margin=0.0,
        visual_weight=1.0,
        text_weight=0.0,
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        runner.run(args)
    assert closed == [True]
