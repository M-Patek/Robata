"""Run InMemoryTaskQueue + PipelineWorker QA -> annotation -> search integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robata.runtime.integration_validation import run_worker_requirements_integration  # noqa: E402


def main() -> int:
    report = run_worker_requirements_integration()
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
