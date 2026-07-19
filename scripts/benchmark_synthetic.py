"""Run the deterministic synthetic serial-vs-parallel benchmark (non-certifying)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.runtime.synthetic_benchmark import run_synthetic_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    from robata.runtime.synthetic_benchmark import build_synthetic_fixtures

    report = run_synthetic_benchmark(
        build_synthetic_fixtures(args.fixtures),
        iterations=args.iterations,
        parallel_workers=args.workers,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.output_hash_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
