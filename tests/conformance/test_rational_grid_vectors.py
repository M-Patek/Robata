from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.verify_rational_grid_vectors import DEFAULT_VECTOR_PATH, verify_vectors

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CASE_IDS = (
    "half-even-ties",
    "negative-k-and-rounded-target-dedupe",
    "clipping-preserves-grid-phase",
    "tolerance-and-nearest-frame-tie-break",
    "one-source-frame-dedupe",
)


def test_python_runtime_matches_checked_in_cross_language_vectors() -> None:
    assert verify_vectors() == EXPECTED_CASE_IDS


def test_node_bigint_runner_matches_checked_in_cross_language_vectors() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for cross-language conformance")

    completed = subprocess.run(
        [node, "scripts/verify_rational_grid_vectors.mjs", str(DEFAULT_VECTOR_PATH)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert json.loads(completed.stdout) == {
        "implementation": "node-bigint-independent-v1",
        "suite": "robata-rational-grid-canonicalization@1.0.0",
        "verified_vectors": len(EXPECTED_CASE_IDS),
    }
