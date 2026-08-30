from __future__ import annotations

import json
from pathlib import Path

from robata.benchmark.production_structured_annotation import (
    normalize_structured_annotation_envelope,
)
from scripts.build_production_structured_annotation import main


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_builds_sidecar_without_model_or_media_work(tmp_path: Path) -> None:
    wemm_path = tmp_path / "wemm.json"
    qwen_path = tmp_path / "qwen.json"
    output_path = tmp_path / "nested" / "structured.json"
    _write(
        wemm_path,
        {
            "format": "robata-production-wemm-shadow-v1",
            "source": {"path": "sample.mcap", "camera_count": 6},
            "windows": [
                {
                    "window_id": "w00",
                    "ordinal": 0,
                    "start_seconds": 0,
                    "end_seconds": 4,
                    "model": {
                        "status": "SUCCEEDED",
                        "predictions": [{"rank": 1, "verb": "open", "noun": "drawer"}],
                    },
                }
            ],
        },
    )
    _write(
        qwen_path,
        {
            "format": "robata-production-qwen-shadow-v1",
            "source": {"manifest": "cohort.json"},
            "camera_ids": ["cam_01"],
            "windows": [
                {
                    "window_id": "w00",
                    "interval": [0, 4.1],
                    "camera_id": "cam_01",
                    "status": "SUCCEEDED",
                    "raw_text": '{"verb":"opens","noun":"drawer"}',
                }
            ],
        },
    )

    assert (
        main(
            [
                "--wemm",
                str(wemm_path),
                "--qwen",
                str(qwen_path),
                "--source-path",
                "sample.mcap",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    envelope = json.loads(output_path.read_text(encoding="utf-8"))
    assert normalize_structured_annotation_envelope(envelope) == envelope
    assert envelope["source"] == {
        "path": "sample.mcap",
        "window_count": 1,
        "camera_count": 6,
    }
    assert envelope["windows"][0]["models"]["mage"]["status"] == "BLOCKED"
    assert envelope["controls"]["gold_included"] is False
