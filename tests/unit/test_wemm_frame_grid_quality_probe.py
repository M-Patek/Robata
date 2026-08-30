from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robata.benchmark.wemm_frame_grid_quality_probe import (
    WemmFrameGridQualityProbeError,
    analyze_wemm_frame_grid_quality,
)
from scripts.analyze_wemm_frame_grid_quality import main


def _runtime_report(
    *,
    source_path: str = "fixture.mcap",
    official_quality_status: str = "NOT_MEASURED",
    official_gold_status: str = "NOT_ESTABLISHED",
    include_batch4: bool = True,
    modal_labels: tuple[str, str] = ("open cupboard", "open drawer"),
) -> dict[str, object]:
    camera_ids = ["cam_01", "cam_02"]
    window_ids = ["w00", "w01"]
    input_order = [
        {
            "row_index": row_index,
            "window_id": window_id,
            "camera_id": camera_id,
            "frame_count": 4,
        }
        for row_index, (window_id, camera_id) in enumerate(
            (window, camera) for window in window_ids for camera in camera_ids
        )
    ]

    per_window = [
        {
            "window_id": window_ids[0],
            "camera_count_observed": 2,
            "modal_top1_count": 2,
            "modal_top1_fraction": 1.0,
            "top1_labels_not_gold": {modal_labels[0]: 2},
        },
        {
            "window_id": window_ids[1],
            "camera_count_observed": 2,
            "modal_top1_count": 1,
            "modal_top1_fraction": 0.5,
            "top1_labels_not_gold": {modal_labels[1]: 1, modal_labels[0]: 1},
        },
    ]
    rank_diagnostic = {
        "phrase_catalog_top1_count": 4,
        "phrase_catalog_top1_fraction": 1.0,
        "top_label_counts_not_gold": {modal_labels[0]: 3, modal_labels[1]: 1},
        "top1_top2_margin_not_calibrated": {
            "count": 4,
            "mean": 0.12,
            "min": 0.04,
            "max": 0.2,
        },
        "camera_consistency_not_gold": {
            "window_count": 2,
            "mean_modal_top1_fraction": 0.75,
            "all_camera_same_fraction": 0.5,
            "per_window": per_window,
        },
    }
    observations = [
        {
            "modality": "video",
            "item_count": 2,
            "frame_count": 4,
            "video_grid_thw": [[2, 14, 16], [2, 14, 16]],
        }
    ]
    arms: list[dict[str, object]] = [
        {
            "arm_id": "serial",
            "batch_size": 1,
            "frame_count": 4,
            "video_max_pixels": 262144,
            "input_count": 4,
            "decode_seconds_shared": 1.5,
            "inference_seconds": 2.0,
            "estimated_e2e_seconds": 3.5,
            "source_camera_normalized_realtime": 1.1,
            "observations": copy.deepcopy(observations),
            "rank_diagnostic": copy.deepcopy(rank_diagnostic),
        }
    ]
    if include_batch4:
        arms.append(
            {
                "arm_id": "batch4",
                "batch_size": 4,
                "frame_count": 4,
                "video_max_pixels": 262144,
                "input_count": 4,
                "decode_seconds_shared": 1.5,
                "inference_seconds": 1.0,
                "estimated_e2e_seconds": 2.5,
                "source_camera_normalized_realtime": 1.6,
                "observations": copy.deepcopy(observations),
                "rank_diagnostic": copy.deepcopy(rank_diagnostic),
            }
        )

    return {
        "format": "robata-wemm-cohort-runtime-benchmark-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "MEASURED_NONPRODUCTION",
        "production_eligible": False,
        "official_quality_status": official_quality_status,
        "official_gold_status": official_gold_status,
        "source": {
            "path": source_path,
            "camera_count": 2,
            "window_count": 2,
            "input_order": input_order,
        },
        "arms": arms,
    }


def _entries(*reports: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "arm_id": f"arm-{index}",
            "source_artifact": f"fixture-{index}.json",
            "runtime_report": report,
        }
        for index, report in enumerate(reports)
    ]


def test_probe_compares_same_cohort_and_reports_modal_agreement() -> None:
    reference = _runtime_report()
    candidate = _runtime_report(modal_labels=("open cupboard", "open cupboard"))

    report = analyze_wemm_frame_grid_quality(
        _entries(reference, candidate),
        reference_arm_id="arm-0",
    )

    assert report["status"] == "MEASURED_NONPRODUCTION_ARTIFACT_ONLY"
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["official_gold_status"] == "NOT_ESTABLISHED"
    assert report["quality_claim"] is False
    assert report["production_eligible"] is False
    assert report["scope"] == {
        "source_path": "fixture.mcap",
        "window_ids": ["w00", "w01"],
        "camera_count": 2,
        "window_count": 2,
        "runtime_arm_id": "batch4",
        "reference_arm_id": "arm-0",
    }
    first, second = report["arms"]  # type: ignore[misc]
    assert first["observed_grids"] == [[2, 14, 16]]
    assert first["observed_frame_counts"] == [4]
    assert first["agreement_vs_reference"] == {"reference": "arm-0", "self": True}
    agreement = second["agreement_vs_reference"]
    assert agreement["common_window_count"] == 2
    assert agreement["modal_top1_agreement_fraction"] == 1.0
    assert agreement["disagreements"] == []
    assert all(value is False for value in report["controls"].values())  # type: ignore[union-attr]
    json.dumps(report)


def test_probe_reports_disagreement_for_changed_window_modal() -> None:
    reference = _runtime_report()
    candidate = _runtime_report(modal_labels=("close tap", "close tap"))

    report = analyze_wemm_frame_grid_quality(_entries(reference, candidate))
    agreement = report["arms"][1]["agreement_vs_reference"]  # type: ignore[index]

    assert agreement["common_window_count"] == 2
    assert agreement["modal_top1_agreement_fraction"] == 0.0
    assert [row["window_id"] for row in agreement["disagreements"]] == ["w00", "w01"]


def test_probe_does_not_count_missing_modal_labels_as_agreement() -> None:
    reference = _runtime_report()
    candidate = _runtime_report()
    candidate_arm = candidate["arms"][1]  # type: ignore[index]
    candidate_camera = candidate_arm["rank_diagnostic"]["camera_consistency_not_gold"]  # type: ignore[index]
    candidate_camera["per_window"][0]["top1_labels_not_gold"] = {}  # type: ignore[index]

    report = analyze_wemm_frame_grid_quality(_entries(reference, candidate))
    agreement = report["arms"][1]["agreement_vs_reference"]  # type: ignore[index]

    assert agreement["modal_top1_agreement_fraction"] == 0.5
    assert agreement["disagreements"][0]["candidate_modal_top1"] is None


def test_probe_rejects_different_cohort() -> None:
    with pytest.raises(WemmFrameGridQualityProbeError, match="same cohort"):
        analyze_wemm_frame_grid_quality(
            _entries(_runtime_report(), _runtime_report(source_path="other.mcap"))
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("official_quality_status", "MEASURED", "official_quality_status"),
        ("official_gold_status", "ESTABLISHED", "official_gold_status"),
    ],
)
def test_probe_rejects_quality_or_gold_claims(field: str, value: str, message: str) -> None:
    kwargs = {field: value}
    with pytest.raises(WemmFrameGridQualityProbeError, match=message):
        analyze_wemm_frame_grid_quality(_entries(_runtime_report(**kwargs)))  # type: ignore[arg-type]


def test_probe_rejects_missing_runtime_arm() -> None:
    with pytest.raises(WemmFrameGridQualityProbeError, match="does not contain runtime arm"):
        analyze_wemm_frame_grid_quality(
            _entries(_runtime_report(include_batch4=False)), runtime_arm_id="batch4"
        )


def test_probe_requires_at_least_one_entry() -> None:
    with pytest.raises(WemmFrameGridQualityProbeError, match="must not be empty"):
        analyze_wemm_frame_grid_quality([])


def test_cli_reads_artifacts_and_writes_posthoc_report() -> None:
    root = Path(__file__).resolve().parents[2] / ".agent_tmp"
    root.mkdir(parents=True, exist_ok=True)
    reference_path = root / "wemm_frame_grid_quality_probe_test_reference.json"
    candidate_path = root / "wemm_frame_grid_quality_probe_test_candidate.json"
    output_path = root / "wemm_frame_grid_quality_probe_test_report.json"
    try:
        reference_path.write_text(json.dumps(_runtime_report()), encoding="utf-8")
        candidate_path.write_text(
            json.dumps(_runtime_report(modal_labels=("close tap", "close tap"))),
            encoding="utf-8",
        )

        status = main(
            [
                "--input",
                f"reference={reference_path}",
                "--input",
                f"candidate={candidate_path}",
                "--reference",
                "reference",
                "--output",
                str(output_path),
            ]
        )

        assert status == 0
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["quality_claim"] is False
        assert report["scope"]["reference_arm_id"] == "reference"
        assert report["arms"][1]["agreement_vs_reference"]["modal_top1_agreement_fraction"] == 0.0
    finally:
        for path in (reference_path, candidate_path, output_path):
            path.unlink(missing_ok=True)
