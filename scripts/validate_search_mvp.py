"""Exercise zero-GPU structured-label search MVP with deterministic fake annotations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.annotation import StructuredLabels  # noqa: E402
from robata.search import ClipIndexEntry, ClipSearchIndex, VerbNormalizer  # noqa: E402


def main() -> int:
    entries = [
        ClipIndexEntry(
            clip_id="clip-clean-1",
            video_id="video-1",
            start_sec=1.0,
            end_sec=3.5,
            structured_labels=StructuredLabels(
                verb="wipe", noun="table", attributes=("red",), location="left", hand="right"
            ),
        ),
        ClipIndexEntry(
            clip_id="clip-clean-2",
            video_id="video-2",
            start_sec=4.0,
            end_sec=7.0,
            structured_labels=StructuredLabels(
                verb="scrub", noun="table", attributes=("blue",), location="right", hand="left"
            ),
        ),
        ClipIndexEntry(
            clip_id="clip-cut-1",
            video_id="video-3",
            start_sec=2.0,
            end_sec=5.0,
            structured_labels=StructuredLabels(
                verb="slice", noun="apple", attributes=("red",), location="center", hand="both"
            ),
        ),
    ]
    index = ClipSearchIndex(entries)
    family = VerbNormalizer().normalize("wash")
    hits = index.filter(verb_family="wash", noun="table", location="left", hand="right")
    facet_hits = index.filter(verb_family="scrub", noun="table", location="right")
    passed = (
        family == "clean"
        and len(hits) == 1
        and hits[0].clip_id == "clip-clean-1"
        and hits[0].playback_target.endswith("?start=1&end=3.5")
        and len(facet_hits) == 1
        and facet_hits[0].clip_id == "clip-clean-2"
    )
    payload = {
        "measurement_status": "NOT_MEASURED",
        "zero_gpu": True,
        "provider_requests": 0,
        "verb_family_wash": family,
        "query_hits": [hit.model_dump(mode="json") for hit in hits],
        "facet_hits": [hit.model_dump(mode="json") for hit in facet_hits],
        "passed": passed,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
