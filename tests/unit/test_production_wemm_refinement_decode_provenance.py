from __future__ import annotations

import json

from robata.benchmark.production_wemm_open_runner import (
    _summarize_refinement_decode_provenance,
)


def test_refinement_decode_provenance_summarizes_padding_by_window() -> None:
    summary = _summarize_refinement_decode_provenance(
        {
            "raw_model_output": {
                "windows": [
                    {
                        "window_id": "fine-000",
                        "input_observations": [
                            {
                                "camera_id": "cam_01",
                                "frame_count_requested": 4,
                                "frame_count_observed": 3,
                                "frame_padding_used": True,
                                "frame_padding_indices": [0],
                            },
                            {
                                "camera_id": "cam_02",
                                "frame_count_requested": 4,
                                "frame_count_observed": 3,
                                "frame_padding_used": True,
                                "frame_padding_indices": [0],
                            },
                        ],
                    },
                    {
                        "window_id": "fine-001",
                        "input_observations": [
                            {
                                "camera_id": "cam_01",
                                "frame_count_requested": 4,
                                "frame_count_observed": 4,
                                "frame_padding_used": False,
                                "frame_padding_indices": [],
                            }
                        ],
                    },
                ]
            }
        }
    )

    assert summary == {
        "format": "robata-production-wemm-temporal-decode-provenance-v1",
        "available": True,
        "camera_window_count": 3,
        "padding_used": True,
        "padding_group_count": 2,
        "padding_group_fraction": 2 / 3,
        "padding_index_counts": {"0": 2},
        "observed_frame_count_counts": {"3": 2, "4": 1},
        "requested_frame_count_counts": {"4": 3},
        "windows": [
            {
                "window_id": "fine-000",
                "camera_window_count": 2,
                "padding_group_count": 2,
                "padding_indices": [0],
                "observed_frame_count_counts": {"3": 2},
                "requested_frame_count_counts": {"4": 2},
            },
            {
                "window_id": "fine-001",
                "camera_window_count": 1,
                "padding_group_count": 0,
                "padding_indices": [],
                "observed_frame_count_counts": {"4": 1},
                "requested_frame_count_counts": {"4": 1},
            },
        ],
    }
    json.dumps(summary)


def test_refinement_decode_provenance_is_empty_when_raw_observations_are_absent() -> None:
    assert _summarize_refinement_decode_provenance({"raw_model_output": {}}) == {
        "format": "robata-production-wemm-temporal-decode-provenance-v1",
        "available": False,
        "camera_window_count": 0,
        "padding_used": False,
        "padding_group_count": 0,
        "padding_group_fraction": None,
        "padding_index_counts": {},
        "observed_frame_count_counts": {},
        "requested_frame_count_counts": {},
        "windows": [],
    }
