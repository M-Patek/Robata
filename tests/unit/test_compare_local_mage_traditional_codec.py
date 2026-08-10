from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import robata.benchmark.mage_traditional_codec as mage_traditional_codec
from robata.benchmark.mage_25x import load_provider_v2_local_baseline
from robata.benchmark.mage_traditional_codec import (
    MageTraditionalEvidenceError,
    build_traditional_local_qualification_report,
    load_host_measurement,
    load_traditional_receipt,
    verify_receipt_sources,
)
from robata.contracts.hashing import canonical_json_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPOSITORY_ROOT / "docs" / "mage-dcvc-provider-v2-local-qualification-2026-08-09.json"
RECEIPT = REPOSITORY_ROOT / "docs" / "mage-traditional-codec-container-receipt-2026-08-09.json"
SINGLE = REPOSITORY_ROOT / "docs" / "mage-traditional-codec-single-receipt-2026-08-09.json"
HOST = REPOSITORY_ROOT / "docs" / "mage-traditional-codec-host-measurement-2026-08-09.json"
REPORT = REPOSITORY_ROOT / "docs" / "mage-traditional-codec-local-qualification-2026-08-09.json"
BASELINE_EXACT = "7298d21fb05f0ecbc4bc1e11481f67abf2c82b4b13380227177edfbbbaa24287"
BASELINE_SEMANTIC = "ea659e3e78243e43e4c1f921ff0898c64f18c4e68993c9c219d2425c8a25b0d8"
IMAGE_DIGEST = "857ad103f01c1594500f6b6ba300c084d891f9ec6106f7f25de583403ec86cbf"
REPORT_EXACT = "7fa7c99d3c6335f84a1e8186cca3d4cd0710458772040247f170e35bafa0f56d"
REPORT_SEMANTIC = "ed23f3fe17d4bbc302b77cc071b7379020cefeae758a44afdb937ce74d600816"


def _tracked_control_source_verifier(path: Path) -> tuple[str, int]:
    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    variants = document["variants"]["provider_v2_bounded"]
    selected = [item for item in variants if item["variant_id"] == "provider-v2-max-side-448"]
    assert len(selected) == 1
    jobs = selected[0]["preparation"]["per_segment"]
    matches = [item for item in jobs if item["source_path"] == str(path)]
    assert len(matches) == 1
    return matches[0]["source_content_sha256"], matches[0]["source_byte_count"]


def _inputs():
    baseline = load_provider_v2_local_baseline(
        path=BASELINE,
        expected_exact_sha256=BASELINE_EXACT,
        expected_semantic_sha256=BASELINE_SEMANTIC,
        source_reference="docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json",
    )
    receipt = load_traditional_receipt(
        path=RECEIPT,
        source_reference="docs/mage-traditional-codec-container-receipt-2026-08-09.json",
    )
    single = load_traditional_receipt(
        path=SINGLE,
        source_reference="docs/mage-traditional-codec-single-receipt-2026-08-09.json",
    )
    host = load_host_measurement(
        path=HOST,
        expected_image_digest=IMAGE_DIGEST,
        source_reference="docs/mage-traditional-codec-host-measurement-2026-08-09.json",
    )
    return baseline, receipt, single, host


def test_tracked_traditional_qualification_report_is_reproducible() -> None:
    baseline, receipt, single, host = _inputs()
    report = build_traditional_local_qualification_report(
        baseline=baseline,
        baseline_report_path=BASELINE,
        receipt=receipt,
        single_receipt=single,
        host_measurement=host,
        control_source_verifier=_tracked_control_source_verifier,
    )

    assert REPORT.read_bytes() == canonical_json_bytes(report) + b"\n"
    import hashlib

    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_EXACT
    assert report["semantic_sha256"] == REPORT_SEMANTIC
    assert report["production_eligible"] is False
    assert report["decision"] == {
        "state": "HOLD_TRADITIONAL",
        "preparation_performance_gate": "PASS",
        "reason": (
            "Traditional H.264 preparation is faster than the retained DCVC control and no "
            "longer dominates the retained decoder timing, but real Mage generation, business "
            "quality, segment-ready integration, and representative salience remain unmeasured."
        ),
    }
    preparation = report["traditional_preparation"]
    assert preparation["service_rates"]["provider_job_sum_seconds"] == pytest.approx(
        5.68156921300033
    )
    assert preparation["repeatability"]["loader_payload_equal"] is True
    assert preparation["repeatability"]["normalized_loader_meta_equal"] is True
    assert preparation["repeatability"]["raw_asset_set_equal"] is False
    assert report["comparison"]["dcvc_worker_job_sum_to_traditional_job_sum_speedup"] > 6.5
    assert report["comparison"]["codec_bottleneck_transferred_to_decoder"] is True
    assert report["quality"]["selected_block_count"] == 0


def test_receipt_rejects_content_hash_tamper(tmp_path: Path) -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload["policy"]["target_canvas"] = 9
    changed = tmp_path / "changed-receipt.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MageTraditionalEvidenceError, match="content identity"):
        load_traditional_receipt(path=changed)


def test_host_measurement_rejects_wrong_image_digest() -> None:
    with pytest.raises(MageTraditionalEvidenceError, match="image digest differs"):
        load_host_measurement(path=HOST, expected_image_digest="0" * 64)


def test_report_rejects_single_probe_payload_drift() -> None:
    baseline, receipt, single, host = _inputs()
    changed_job = replace(single.jobs[0], loader_payload_sha256="0" * 64)
    changed_single = replace(single, jobs=(changed_job,))

    with pytest.raises(MageTraditionalEvidenceError, match="loader payload is not repeatable"):
        build_traditional_local_qualification_report(
            baseline=baseline,
            baseline_report_path=BASELINE,
            receipt=receipt,
            single_receipt=changed_single,
            host_measurement=host,
            control_source_verifier=_tracked_control_source_verifier,
        )


def test_report_rejects_control_source_attestation_mismatch() -> None:
    baseline, receipt, single, host = _inputs()

    with pytest.raises(MageTraditionalEvidenceError, match="source bytes differ"):
        build_traditional_local_qualification_report(
            baseline=baseline,
            baseline_report_path=BASELINE,
            receipt=receipt,
            single_receipt=single,
            host_measurement=host,
            control_source_verifier=lambda _path: ("0" * 64, 1),
        )


def test_source_verifier_defaults_to_live_exact_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt, _, _ = _inputs()
    observed: list[Path] = []

    def mismatched_exact_file(path: Path) -> tuple[str, int]:
        observed.append(path)
        return "0" * 64, 1

    monkeypatch.setattr(mage_traditional_codec, "_exact_file", mismatched_exact_file)
    with pytest.raises(MageTraditionalEvidenceError, match="source bytes differ"):
        verify_receipt_sources(receipt=receipt, baseline_report_path=BASELINE)

    assert len(observed) == 1
