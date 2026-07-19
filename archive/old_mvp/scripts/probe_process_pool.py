"""Run local ProcessPool/PNG-reuse engineering probes (non-certifying)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.runtime.process_pool_poc import run_spawn_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    report = run_spawn_probe(args.iterations, max_workers=args.workers)
    print(json.dumps({"measurement_status": "NOT_MEASURED", **asdict(report)}, indent=2))
    return 0 if report.supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
