"""Run a deterministic, provider-free acceptance check for REQUIREMENTS.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.annotation import AnnotationPipeline  # noqa: E402
from robata.capacity import (  # noqa: E402
    CapacityPlanner,
    MeasurementStatus,
    SLAStage,
    ThroughputLedger,
    calibrate_capacity_scenarios,
)
from robata.frame_cache import FramePayload, SharedFrameCache  # noqa: E402
from robata.qa import ClipMark, QAClassifier, QAIssue, QAStatus  # noqa: E402
from robata.qa_validation import validate_issue_matrix  # noqa: E402
from robata.runtime.integration_validation import (  # noqa: E402
    run_frame_cache_stress,
    run_worker_requirements_integration,
)
from robata.runtime.synthetic_benchmark import (  # noqa: E402
    build_synthetic_fixtures,
    run_synthetic_benchmark,
)
from robata.search import ClipSearchIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    classifier = QAClassifier()
    passed = classifier.assess("pass", 10.0)
    warning = classifier.assess(
        "warning",
        10.0,
        [
            ClipMark(start_sec=1.0, end_sec=2.0, issue=QAIssue.BLACK_SCREEN, confidence=0.95),
            ClipMark(start_sec=5.0, end_sec=5.5, issue=QAIssue.HAIR_BLOCKING_VIEW, confidence=0.8),
        ],
    )
    failed = classifier.assess(
        "fail",
        10.0,
        [ClipMark(start_sec=0.0, end_sec=10.0, issue=QAIssue.BLACK_SCREEN, confidence=1.0)],
    )
    if (
        passed.status is not QAStatus.PASS
        or warning.status is not QAStatus.WARNING
        or failed.status is not QAStatus.FAIL
    ):
        raise RuntimeError("QA pass/warning/fail acceptance failed")
    annotations = AnnotationPipeline().run((passed, warning, failed))
    if "fail" in annotations.accepted_video_ids or "fail" not in annotations.skipped_fail_video_ids:
        raise RuntimeError("fail video was not excluded from annotation")
    index = ClipSearchIndex(annotations.drafts)
    hits = index.search("interact object")
    if not hits or not all(hit.start_sec < hit.end_sec for hit in hits):
        raise RuntimeError("clip search acceptance failed")
    cache_root = ROOT / "tmp" / "requirements-acceptance-frame-cache"
    cache = SharedFrameCache(cache_root)
    cache.clear_video("acceptance")
    cache.feed_once("acceptance", "local://acceptance", lambda: [FramePayload(0.0, b"frame")])
    cache.feed_once(
        "acceptance",
        "local://acceptance",
        lambda: (_ for _ in ()).throw(RuntimeError("must not decode twice")),
    )
    matrix = validate_issue_matrix()
    if not matrix.passed or matrix.issue_count != 21:
        raise RuntimeError("QA 21-issue matrix acceptance failed")
    cache_stress = run_frame_cache_stress(video_count=3, callers=9, frames_per_video=2)
    if not cache_stress.passed:
        raise RuntimeError("frame cache concurrency acceptance failed")
    worker_integration = run_worker_requirements_integration()
    if not worker_integration.passed:
        raise RuntimeError("worker QA->annotation->search acceptance failed")
    synthetic_benchmark = run_synthetic_benchmark(
        build_synthetic_fixtures(3), iterations=1, warmups=0
    )
    if not synthetic_benchmark.output_hash_equal:
        raise RuntimeError("serial/parallel synthetic benchmark hash mismatch")
    capacity_matrix = calibrate_capacity_scenarios()
    planner = CapacityPlanner()
    ledger = ThroughputLedger(planner=planner)
    ledger.record(SLAStage.QA, 1.0, 1.0, measurement_status=MeasurementStatus.NOT_MEASURED)
    report = ledger.report()
    payload = {
        "qa": {
            "pass": passed.status.value,
            "warning": warning.status.value,
            "fail": failed.status.value,
        },
        "annotation": {
            "draft_count": annotations.draft_count,
            "skipped_fail": annotations.skipped_fail_video_ids,
        },
        "search": {"result_count": len(hits), "zero_gpu": True},
        "frame_cache": {"decode_attempts": cache.stats().decode_attempts, "feed_once": True},
        "capacity": {
            "target_recording_hours_per_day": report.target_hours_per_day,
            "planned_gpu_hours_per_day": planner.planned_gpu_hours_per_day,
            "measurement_status": MeasurementStatus.NOT_MEASURED.value,
            "production_eligible": False,
        },
        "qa_matrix": {
            "issue_count": matrix.issue_count,
            "passed": matrix.passed,
        },
        "frame_cache_stress": cache_stress.as_dict(),
        "worker_integration": worker_integration.as_dict(),
        "synthetic_benchmark": {
            "output_hash_equal": synthetic_benchmark.output_hash_equal,
            "measurement_status": synthetic_benchmark.measurement_status,
            "certifying": synthetic_benchmark.certifying,
        },
        "capacity_scenarios": [scenario.as_dict() for scenario in capacity_matrix],
        "provider_requests": 0,
        "execution_mode": "LOCAL_DEVELOPMENT_FAKE_MODEL",
        "production_eligible": False,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
