from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robata.benchmark.production_wemm_candidate_order_diagnostic import (
    AMBIGUITY_SELECTION_FORMAT,
    ProductionWemmCandidateOrderDiagnosticError,
    build_candidate_order_diagnostic,
    order_top_k,
)
from scripts.build_production_wemm_candidate_order_diagnostic import main


def _selection() -> dict[str, object]:
    candidates = [
        {"rank": 1, "label_text": "fold garment", "score": 0.81},
        {"rank": 2, "label_text": "flatten garment", "score": 0.80},
        {"rank": 3, "label_text": "pick up garment", "score": 0.79},
    ]
    return {
        "format": AMBIGUITY_SELECTION_FORMAT,
        "status": "SELECTED",
        "quality_claim": False,
        "controls": {"model_invoked": False, "hash_or_digest_computed": False},
        "summary": {"selected_window_count": 9},
        "windows": [
            {
                "recording_id": "recording-a",
                "window_id": "recording-a-w0001",
                "ordinal": 1,
                "source_context_is_action_boundary": False,
                "declared_camera_ids": ["cam_01", "cam_02"],
                "source_interval": {
                    "start_seconds": 8.0,
                    "end_seconds": 16.0,
                    "status": "WINDOW_CONTEXT_ONLY",
                },
                "raw_candidates": copy.deepcopy(candidates),
                "proposal_diagnostics": [
                    {
                        "proposal_id": "proposal-1",
                        "top1_label": "fold garment",
                        "top2_label": "flatten garment",
                        "margin": 0.01,
                        "top_k": copy.deepcopy(candidates),
                    }
                ],
                "source_ref": {
                    "archive_member": "file/recording-a.mcap",
                    "source": {"camera_ids": ["cam_01", "cam_02"]},
                },
            }
        ],
    }


def _signatures(rows: list[dict[str, object]]) -> list[tuple[object, object, object]]:
    return [(row.get("rank"), row.get("label_text"), row.get("score")) for row in rows]


def test_reverse_preserves_rank_label_score_and_does_not_mutate_input() -> None:
    candidates = _selection()["windows"][0]["proposal_diagnostics"][0]["top_k"]  # type: ignore[index]
    original = copy.deepcopy(candidates)
    ordered, metadata = order_top_k(candidates, mode="reverse", window_id="w01")  # type: ignore[arg-type]
    assert [row["rank"] for row in ordered] == [3, 2, 1]
    assert sorted(_signatures(ordered)) == sorted(_signatures(original))
    assert candidates == original
    assert metadata["rank_label_score_preserved"] is True
    assert metadata["order_changed"] is True


def test_shuffle_is_deterministic_and_keeps_source_rank_fields() -> None:
    candidates = _selection()["windows"][0]["proposal_diagnostics"][0]["top_k"]  # type: ignore[index]
    first, first_meta = order_top_k(candidates, mode="shuffle", seed="test", window_id="w01")  # type: ignore[arg-type]
    second, second_meta = order_top_k(candidates, mode="shuffle", seed="test", window_id="w01")  # type: ignore[arg-type]
    assert first == second
    assert first_meta["after"] == second_meta["after"]
    assert sorted(int(row["rank"]) for row in first) == [1, 2, 3]


def test_build_one_window_reverse_and_restrict_camera() -> None:
    source = _selection()
    report = build_candidate_order_diagnostic(
        source,
        mode="reverse",
        camera_id="cam_02",
        source_path="pilot24.json",
    )
    row = report["windows"][0]  # type: ignore[index]
    assert report["status"] == "CANDIDATE_ORDER_DIAGNOSTIC"
    assert row["declared_camera_ids"] == ["cam_02"]
    assert [item["rank"] for item in row["proposal_diagnostics"][0]["top_k"]] == [3, 2, 1]  # type: ignore[index]
    assert row["raw_candidates"] == source["windows"][0]["raw_candidates"]  # type: ignore[index]
    assert report["diagnostic"]["order"]["rank_label_score_preserved"] is True  # type: ignore[index]
    assert report["diagnostic"]["camera_scope"]["requested_camera_id"] == "cam_02"  # type: ignore[index]
    assert report["summary"]["selected_window_count"] == 1  # type: ignore[index]


def test_unknown_camera_and_bad_mode_fail() -> None:
    with pytest.raises(ProductionWemmCandidateOrderDiagnosticError, match="not declared"):
        build_candidate_order_diagnostic(_selection(), mode="reverse", camera_id="cam_99")
    with pytest.raises(ProductionWemmCandidateOrderDiagnosticError, match="mode"):
        order_top_k([], mode="randomized")  # type: ignore[arg-type]


def test_cli_writes_variants_without_model_invocation(tmp_path: Path) -> None:
    input_path = tmp_path / "selection.json"
    input_path.write_text(json.dumps(_selection()), encoding="utf-8")
    output_dir = tmp_path / "out"
    status = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "as_is",
            "--mode",
            "reverse",
            "--camera-id",
            "cam_01",
        ]
    )
    assert status == 0
    files = sorted(output_dir.glob("*.json"))
    assert len(files) == 3  # two selections plus the pack manifest
    manifest_path = output_dir / "candidate-order-recording-a-w0001-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_invoked"] is False
    reverse_path = output_dir / "candidate-order-recording-a-w0001-reverse.json"
    reverse = json.loads(reverse_path.read_text())
    ranks = [item["rank"] for item in reverse["windows"][0]["proposal_diagnostics"][0]["top_k"]]
    assert ranks == [3, 2, 1]
