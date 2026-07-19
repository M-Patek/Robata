"""Run every local workstream and emit one non-certifying evidence report."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.capacity import calibrate_capacity_scenarios  # noqa: E402
from robata.qa_validation import validate_sample_mcap  # noqa: E402
from robata.runtime.integration_validation import (  # noqa: E402
    run_frame_cache_stress,
    run_worker_requirements_integration,
)
from robata.runtime.process_pool_poc import compare_png_reuse, run_spawn_probe  # noqa: E402
from robata.runtime.synthetic_benchmark import (  # noqa: E402
    build_synthetic_fixtures,
    run_synthetic_benchmark,
)


def _png_probe() -> dict[str, object]:
    try:
        import av
    except ModuleNotFoundError:
        return {"supported": False, "error": "PyAV is not installed"}
    frames = []
    for value in (0, 1, 2):
        frame = av.VideoFrame(2, 2, "rgb24")
        frame.pts = 0
        frame.time_base = Fraction(1, 1)
        frame.planes[0].update(bytes([value]) * frame.planes[0].buffer_size)
        frames.append(frame)
    return asdict(compare_png_reuse(frames))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(ROOT / "data" / "source" / "sample-medium.mcap"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "local-workstreams-2026-07-19.json"),
    )
    args = parser.parse_args()
    sample = validate_sample_mcap(args.source)
    cache = run_frame_cache_stress(video_count=4, callers=12, frames_per_video=3)
    worker = run_worker_requirements_integration()
    benchmark = run_synthetic_benchmark(build_synthetic_fixtures(4), iterations=2, warmups=1)
    spawn = run_spawn_probe(iterations=4, max_workers=2)
    payload = {
        "date": "2026-07-19",
        "measurement_status": "NOT_MEASURED",
        "production_eligible": False,
        "provider_requests": 0,
        "execution_mode": "LOCAL_DEVELOPMENT_FAKE_MODEL",
        "qa_sample": sample.as_dict(),
        "frame_cache_stress": cache.as_dict(),
        "worker_integration": worker.as_dict(),
        "synthetic_benchmark": benchmark.as_dict(),
        "process_pool_spawn": asdict(spawn),
        "png_reuse": _png_probe(),
        "capacity_scenarios": [scenario.as_dict() for scenario in calibrate_capacity_scenarios()],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    checks = [sample.passed, cache.passed, worker.passed, benchmark.output_hash_equal]
    return 0 if all(checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
