"""Verify the language-neutral rational-grid and canonicalization golden vectors."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.alignment.rational_time import round_half_even  # noqa: E402
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256  # noqa: E402
from robata.sampling.grid import (  # noqa: E402
    FrameCandidate,
    SamplingGrid,
    SamplingRate,
    TargetSelection,
)

DEFAULT_VECTOR_PATH = REPOSITORY_ROOT / "conformance" / "rational-grid-canonicalization-v1.json"
_CANONICAL_DECIMAL = re.compile(r"^(?:0|-[1-9][0-9]*|[1-9][0-9]*)$")


class VectorVerificationError(ValueError):
    """One checked-in conformance vector is malformed or disagrees with the runtime."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VectorVerificationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise VectorVerificationError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise VectorVerificationError(f"{label} must be a JSON array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VectorVerificationError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str) -> int:
    text = _text(value, label)
    if _CANONICAL_DECIMAL.fullmatch(text) is None:
        raise VectorVerificationError(f"{label} must use canonical base-10 integer text")
    return int(text)


def _frame(value: Any, label: str) -> FrameCandidate:
    fields = _object(value, label)
    locator_hex = _text(fields.get("source_locator_hex"), f"{label}.source_locator_hex")
    if locator_hex.lower() != locator_hex or len(locator_hex) % 2 != 0:
        raise VectorVerificationError(f"{label}.source_locator_hex must be lowercase octets")
    try:
        locator = bytes.fromhex(locator_hex)
    except ValueError as exc:
        raise VectorVerificationError(
            f"{label}.source_locator_hex must be lowercase hexadecimal"
        ) from exc
    decodable = fields.get("decodable")
    if not isinstance(decodable, bool):
        raise VectorVerificationError(f"{label}.decodable must be a boolean")
    return FrameCandidate(
        aligned_timestamp_ns=_integer(
            fields.get("aligned_timestamp_ns"), f"{label}.aligned_timestamp_ns"
        ),
        source_timestamp_ns=_integer(
            fields.get("source_timestamp_ns"), f"{label}.source_timestamp_ns"
        ),
        source_locator_bytes=locator,
        decodable=decodable,
    )


def _target_projection(grid: SamplingGrid, start_ns: int, end_ns: int) -> list[dict[str, str]]:
    return [
        {"k": str(target.k), "target_ns": str(target.target_ns)}
        for target in grid.enumerate_targets(start_ns, end_ns)
    ]


def _selection_projection(selection: TargetSelection) -> dict[str, Any]:
    frame = selection.frame
    frame_projection = (
        None
        if frame is None
        else {
            "aligned_timestamp_ns": str(frame.aligned_timestamp_ns),
            "source_timestamp_ns": str(frame.source_timestamp_ns),
            "source_locator_hex": frame.source_locator_bytes.hex(),
            "decodable": frame.decodable,
        }
    )
    return {
        "k": str(selection.k),
        "target_ns": str(selection.target_ns),
        "status": selection.status.value,
        "frame": frame_projection,
        "delta_to_target_ns": (
            None if selection.delta_to_target_ns is None else str(selection.delta_to_target_ns)
        ),
    }


def _rounding_outcome(vector_input: Mapping[str, Any], label: str) -> dict[str, Any]:
    fractions = _array(vector_input.get("fractions"), f"{label}.fractions")
    results: list[str] = []
    for index, value in enumerate(fractions):
        fraction = _object(value, f"{label}.fractions[{index}]")
        numerator = _integer(fraction.get("numerator"), f"{label}.fractions[{index}].numerator")
        denominator = _integer(
            fraction.get("denominator"), f"{label}.fractions[{index}].denominator"
        )
        results.append(str(round_half_even(numerator, denominator)))
    return {"rounding_results": results}


def _grid_outcome(vector_input: Mapping[str, Any], label: str) -> dict[str, Any]:
    grid_fields = _object(vector_input.get("grid"), f"{label}.grid")
    interval = _object(vector_input.get("interval"), f"{label}.interval")
    grid = SamplingGrid(
        grid_origin_ns=_integer(grid_fields.get("grid_origin_ns"), f"{label}.grid.grid_origin_ns"),
        rate=SamplingRate(
            numerator=_integer(grid_fields.get("rate_num"), f"{label}.grid.rate_num"),
            denominator=_integer(grid_fields.get("rate_den"), f"{label}.grid.rate_den"),
        ),
    )
    start_ns = _integer(interval.get("start_ns"), f"{label}.interval.start_ns")
    end_ns = _integer(interval.get("end_ns"), f"{label}.interval.end_ns")
    tolerance_ns = _integer(
        vector_input.get("selection_tolerance_ns"), f"{label}.selection_tolerance_ns"
    )
    frames = tuple(
        _frame(value, f"{label}.frames[{index}]")
        for index, value in enumerate(_array(vector_input.get("frames"), f"{label}.frames"))
    )
    selections = grid.select_frames(frames, start_ns, end_ns, tolerance_ns)
    return {
        "period": {
            "period_num_ns": str(grid.period_num_ns),
            "period_den": str(grid.period_den),
        },
        "targets": _target_projection(grid, start_ns, end_ns),
        "selections": [_selection_projection(selection) for selection in selections],
    }


def verify_vectors(path: Path = DEFAULT_VECTOR_PATH) -> tuple[str, ...]:
    """Verify canonical input pins and runtime outcomes, returning checked case IDs."""

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VectorVerificationError(f"cannot read vector document {path}: {exc}") from exc
    root = _object(document, "document")
    if root.get("suite_id") != "robata-rational-grid-canonicalization":
        raise VectorVerificationError("unexpected suite_id")
    if root.get("suite_version") != "1.0.0":
        raise VectorVerificationError("unexpected suite_version")
    if root.get("canonicalization") != "RFC8785":
        raise VectorVerificationError("canonicalization must be RFC8785")
    if root.get("integer_encoding") != "canonical-base10-string":
        raise VectorVerificationError("integer_encoding must be canonical-base10-string")

    checked: list[str] = []
    for index, value in enumerate(_array(root.get("vectors"), "vectors")):
        vector = _object(value, f"vectors[{index}]")
        case_id = _text(vector.get("case_id"), f"vectors[{index}].case_id")
        if case_id in checked:
            raise VectorVerificationError(f"duplicate case_id: {case_id}")
        label = f"vector {case_id!r}"
        vector_input = _object(vector.get("input"), f"{label}.input")
        expected = _object(vector.get("expected"), f"{label}.expected")

        canonical_input = canonical_json_bytes(vector_input)
        if expected.get("canonical_input_bytes_hex") != canonical_input.hex():
            raise VectorVerificationError(f"{label} canonical input bytes do not match")
        if expected.get("canonical_input_sha256") != exact_bytes_sha256(canonical_input):
            raise VectorVerificationError(f"{label} canonical input SHA-256 does not match")

        operation = vector.get("operation")
        if operation == "ROUND_HALF_EVEN":
            actual_outcome = _rounding_outcome(vector_input, label)
        elif operation == "GRID_SELECTION":
            actual_outcome = _grid_outcome(vector_input, label)
        else:
            raise VectorVerificationError(f"{label} has unsupported operation {operation!r}")
        if expected.get("outcome") != actual_outcome:
            raise VectorVerificationError(f"{label} runtime outcome does not match")
        checked.append(case_id)
    if not checked:
        raise VectorVerificationError("vector suite must not be empty")
    return tuple(checked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_VECTOR_PATH)
    args = parser.parse_args(argv)
    checked = verify_vectors(args.path)
    print(
        json.dumps(
            {
                "implementation": "python-authoritative-runtime-v1",
                "suite": "robata-rational-grid-canonicalization@1.0.0",
                "verified_vectors": len(checked),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
