from __future__ import annotations

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import semantic_sha256
from robata.runtime.media_adaptive_profile import (
    MediaAdaptiveMeasurement,
    MediaAdaptivePolicy,
    build_media_adaptive_profile_report,
)

_HOUR_NS = 3_600_000_000_000


def _rows() -> tuple[MediaAdaptiveMeasurement, ...]:
    workload = semantic_sha256({"media-adaptive-test": "workload"})
    values = (
        (MediaAdaptivePolicy.BASELINE, 1_000_000_000, 120, 150, 12, 0, 0, 10_000, 20_000, 30_000),
        (
            MediaAdaptivePolicy.SENTINEL_ONLY,
            700_000_000,
            48,
            58,
            5,
            0,
            0,
            7_000,
            12_000,
            24_000,
        ),
        (
            MediaAdaptivePolicy.SELECTIVE_GEOMETRY,
            800_000_000,
            52,
            68,
            6,
            15,
            2,
            8_000,
            13_000,
            25_000,
        ),
    )
    return tuple(
        MediaAdaptiveMeasurement(
            policy=policy,
            workload_fingerprint=workload,
            recording_count=2,
            camera_count=6,
            recording_duration_ns=_HOUR_NS,
            wall_time_ns=wall_time_ns,
            decoded_frames=240,
            selected_images=selected_images,
            provider_images=provider_images,
            provider_calls=provider_calls,
            geometry_images=geometry_images,
            geometry_calls=geometry_calls,
            process_read_bytes=read_bytes,
            process_write_bytes=write_bytes,
            peak_rss_bytes=rss_bytes,
        )
        for (
            policy,
            wall_time_ns,
            selected_images,
            provider_images,
            provider_calls,
            geometry_images,
            geometry_calls,
            read_bytes,
            write_bytes,
            rss_bytes,
        ) in values
    )


def test_profile_reports_units_resources_amplification_and_frontier() -> None:
    report = build_media_adaptive_profile_report(reversed(_rows()))

    assert report.baseline.recording_hours == pytest.approx(2.0)
    assert report.baseline.camera_hours == pytest.approx(12.0)
    assert report.baseline.recording_hours_per_wall_hour == pytest.approx(7_200.0)
    assert report.sentinel_only.provider_image_amplification == pytest.approx(58 / 48)
    assert report.selective_geometry.geometry_selection_fraction == pytest.approx(15 / 52)
    assert report.comparisons[0].wall_time_ratio == pytest.approx(0.7)
    assert report.comparisons[0].provider_images_ratio == pytest.approx(58 / 150)
    assert report.pareto_policy_ids == (MediaAdaptivePolicy.SENTINEL_ONLY,)
    assert report.production_eligible is False
    assert report.evidence_note == "LOCAL_ONLY_NOT_PRODUCTION_QUALIFIED"
    assert report.as_dict()["evidence_class"] == "LOCAL_CONFORMANCE"
    markdown = report.render_markdown()
    assert "Recording hours" in markdown
    assert "Camera hours" in markdown
    assert "Provider calls" in markdown
    assert "Peak RSS" in markdown


def test_profile_digest_and_policy_order_reject_tampering() -> None:
    report = build_media_adaptive_profile_report(_rows())
    payload = report.model_dump(mode="python")
    payload["profile_sha256"] = semantic_sha256({"tampered": True})
    with pytest.raises(ValidationError, match="profile_sha256"):
        type(report).model_validate(payload, strict=True)

    with pytest.raises(ValueError, match="each P2 policy"):
        build_media_adaptive_profile_report(_rows()[:2])


def test_profile_preserves_unavailable_io_and_requires_quality_pairing() -> None:
    rows = _rows()
    missing_io = tuple(row.model_copy(update={"process_read_bytes": None}) for row in rows)
    report = build_media_adaptive_profile_report(missing_io)
    assert report.baseline.process_read_bytes is None
    assert report.comparisons[0].process_read_bytes_ratio is None

    with pytest.raises(ValidationError, match="quality_score"):
        type(rows[0]).model_validate(
            {
                **rows[0].model_dump(mode="python"),
                "quality_measurement_status": "LOCAL_PROXY",
            },
            strict=True,
        )


def test_no_provider_mode_cannot_hide_provider_work() -> None:
    row = _rows()[0]
    with pytest.raises(ValidationError, match="NO_PROVIDER_CALLS"):
        type(row).model_validate(
            {
                **row.model_dump(mode="python"),
                "provider_mode": "NO_PROVIDER_CALLS",
                "provider_images": 1,
            },
            strict=True,
        )
