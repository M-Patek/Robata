from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.benchmark.production_wemm_batch_plan import (
    ProductionWemmBatchPlanError,
    build_production_wemm_batch_plan,
    load_qa_statuses,
    render_markdown,
)


def _inventory() -> dict[str, object]:
    return {
        "archive_path": "D:/input/source.zip",
        "mcap_entries": [
            {"name": "file/medium.mcap", "size_bytes": 10, "ordinal": 1},
            {"name": "file/small.mcap", "size_bytes": 4, "ordinal": 0},
            {"name": "file/large.mcap", "size_bytes": 20, "ordinal": 2},
        ],
    }


def test_plan_keeps_pending_out_of_execution_and_allows_open_label_space() -> None:
    plan = build_production_wemm_batch_plan(
        _inventory(),
        max_batch_bytes=12,
        max_items_per_batch=2,
        qa_status_by_member={
            "file/small.mcap": "pass",
            "file/medium.mcap": "warning",
            "file/large.mcap": "fail",
        },
    )

    assert plan["format"] == "robata-production-wemm-batch-plan-v1"
    assert plan["label_space"]["kind"] == "OPEN_PROVISIONAL_PHRASES"  # type: ignore[index]
    assert plan["label_space"]["epic_ontology_used"] is False  # type: ignore[index]
    assert plan["summary"]["recording_count"] == 3  # type: ignore[index]
    assert plan["summary"]["scheduled_recording_count"] == 2  # type: ignore[index]
    assert plan["summary"]["fail_count"] == 1  # type: ignore[index]
    assert plan["summary"]["pending_qa_count"] == 0  # type: ignore[index]
    assert len(plan["batches"]) == 2  # type: ignore[arg-type]
    assert all(item["route"] == "wemm_video_preannotation" for item in plan["batches"])  # type: ignore[index]
    assert plan["controls"]["wemm_invoked"] is False  # type: ignore[index]


def test_pending_can_be_included_for_dry_run_but_fail_never_is() -> None:
    plan = build_production_wemm_batch_plan(
        _inventory(),
        max_batch_bytes=100,
        max_items_per_batch=10,
        include_pending_qa=True,
        qa_status_by_member={"file/large.mcap": "FAIL"},
    )
    # All but the explicitly failed row are pending and are schedulable only
    # because the dry-run override was requested.
    assert plan["summary"]["scheduled_recording_count"] == 2  # type: ignore[index]
    assert plan["summary"]["pending_qa_count"] == 2  # type: ignore[index]


def test_priority_order_is_explicit_and_unknown_ordinals_rejected() -> None:
    plan = build_production_wemm_batch_plan(
        _inventory(),
        priority_ordinals=[2, 0],
        qa_status_by_member={
            "file/small.mcap": "PASS",
            "file/medium.mcap": "PASS",
            "file/large.mcap": "WARNING",
        },
    )
    assert [item["ordinal"] for item in plan["items"]] == [2, 0, 1]  # type: ignore[index]
    with pytest.raises(ProductionWemmBatchPlanError, match="unknown"):
        build_production_wemm_batch_plan(_inventory(), priority_ordinals=[99])


def test_oversize_recording_is_singleton_not_dropped() -> None:
    plan = build_production_wemm_batch_plan(
        _inventory(),
        max_batch_bytes=5,
        qa_status_by_member={
            "file/small.mcap": "PASS",
            "file/medium.mcap": "PASS",
            "file/large.mcap": "PASS",
        },
    )
    oversize = [batch for batch in plan["batches"] if batch["oversize_item"]]  # type: ignore[index]
    assert len(oversize) == 2
    assert all(batch["item_count"] == 1 for batch in oversize)


def test_zip_input_delegates_to_central_directory_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    # The archive reader itself is covered by production_corpus_audit tests;
    # this seam proves the planner delegates to it and does not implement a
    # second extraction path.
    import robata.benchmark.production_wemm_batch_plan as module

    called: list[Path] = []

    def fake_audit(path: Path):
        called.append(path)
        return {
            "archive_path": str(path),
            "mcap_entries": [
                {"name": "file/a.mcap", "size_bytes": 1},
                {"name": "file/b.mcap", "size_bytes": 2},
            ],
        }

    monkeypatch.setattr(module, "audit_zip_archive", fake_audit)
    plan = build_production_wemm_batch_plan(
        Path("source.zip"),
        qa_status_by_member={"file/a.mcap": "PASS", "file/b.mcap": "WARNING"},
    )
    assert called == [Path("source.zip").resolve()]
    assert plan["source"]["inspection_mode"] == "central_directory_only"  # type: ignore[index]
    assert plan["summary"]["scheduled_recording_count"] == 2  # type: ignore[index]


def test_audit_inventory_and_markdown_are_accepted() -> None:
    plan = build_production_wemm_batch_plan(
        {"mcap_entries": [{"name": "sample.mcap", "size_bytes": 4}]},
        qa_status_by_member={"sample.mcap": "PASS"},
    )
    markdown = render_markdown(plan)
    assert "PLANNED_NONPRODUCTION" in markdown
    assert "open/provisional" in markdown
    json.dumps(plan)


def test_qa_status_loader_accepts_wrapper_and_rejects_non_mapping() -> None:
    assert load_qa_statuses({"statuses": {"file/a.mcap": "WARN"}})["file/a.mcap"] == "WARN"
    with pytest.raises(ProductionWemmBatchPlanError):
        load_qa_statuses([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        {"mcap_entries": [{"name": "../a.mcap", "size_bytes": 1}]},
        {"mcap_entries": [{"name": "a.mp4", "size_bytes": 1}]},
        {"mcap_entries": [{"name": "a.mcap", "size_bytes": -1}]},
    ],
)
def test_invalid_inventory_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ProductionWemmBatchPlanError):
        build_production_wemm_batch_plan(bad)
