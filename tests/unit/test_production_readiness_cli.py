from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from robata.benchmark.production_cohort import DEFAULT_CAMERA_TOPICS
from robata.benchmark.production_model_output import build_model_output_sidecar
from scripts import assess_production_readiness as readiness_cli


def _manifest() -> dict[str, object]:
    return {
        "format": "robata-production-shaped-cohort-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {"path": "sample.mcap"},
        "windows": [
            {
                "ordinal": 0,
                "window_id": "sample-w00",
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "camera_ids": list(DEFAULT_CAMERA_TOPICS),
            }
        ],
    }


def _paths(prefix: str) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2] / ".agent_tmp"
    root.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex
    return (
        root / f"{prefix}-{suffix}-manifest.json",
        root / f"{prefix}-{suffix}-report.json",
    )


def test_run_writes_json_report_and_main_returns_blocked_exit() -> None:
    manifest_path, output_path = _paths("production-readiness-cli")
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    try:
        report = readiness_cli.run(
            manifest_path=manifest_path,
            output_path=output_path,
        )

        assert report["inference_readiness"] == "BLOCKED"
        assert json.loads(output_path.read_text(encoding="utf-8")) == report
        assert readiness_cli.main([str(manifest_path)]) == 1
    finally:
        manifest_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def test_main_returns_zero_for_structurally_invocation_ready_inputs() -> None:
    manifest = _manifest()
    root = Path(__file__).resolve().parents[2] / ".agent_tmp"
    root.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex
    paths = {
        kind: root / f"production-readiness-cli-{suffix}-{kind}.json"
        for kind in ("manifest", "sidecar", "ontology", "mapping")
    }
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    paths["sidecar"].write_text(json.dumps(build_model_output_sidecar(manifest)), encoding="utf-8")
    paths["ontology"].write_text(
        json.dumps({"approved": True, "actions": [{"id": 1}]}), encoding="utf-8"
    )
    paths["mapping"].write_text(
        json.dumps({"approved": True, "mappings": [{"camera": "cam_01"}]}),
        encoding="utf-8",
    )
    try:
        # Human review is intentionally omitted, so this is invocation-ready
        # but remains quality-NOT_MEASURED.
        assert (
            readiness_cli.main(
                [
                    str(paths["manifest"]),
                    "--sidecar",
                    str(paths["sidecar"]),
                    "--ontology",
                    str(paths["ontology"]),
                    "--mapping",
                    str(paths["mapping"]),
                ]
            )
            == 0
        )
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)
