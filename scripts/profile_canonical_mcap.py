"""Profile one explicitly mapped local MCAP through the canonical composition."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Literal, Protocol, TextIO, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.application.canonical.local_composition import (  # noqa: E402
    CanonicalLocalCompositionError,
    CanonicalLocalRunReceipt,
    run_local_canonical_mcap,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402
from robata.runtime.canonical_profile import (  # noqa: E402
    CanonicalProfileError,
    CanonicalProfileReport,
    CanonicalProfileRunError,
    build_canonical_profile_manifest,
    build_canonical_profile_measurements,
    build_profile_capacity,
    build_profile_reconciliation,
    compare_canonical_profile_reports,
    discover_canonical_profile_durations,
    snapshot_state_tree,
    snapshot_work_queue,
    unique_runtime_counter_value,
)
from robata.runtime.observability import RuntimeProfileRecorder  # noqa: E402

DEFAULT_MAX_DURATION_SECONDS = 180
_ExecutionMode = Literal["FRESH", "REPLAY", "UNKNOWN"]


class _ProfiledCanonicalMcapRunner(Protocol):
    def __call__(
        self,
        *,
        source_path: Path,
        mapping_config: Path,
        state_dir: Path,
        run_key: str,
        allow_unapproved_profile: bool,
        max_duration_ns: int,
        runtime_observer: RuntimeProfileRecorder,
    ) -> CanonicalLocalRunReceipt: ...


def _positive_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the local canonical pipeline from one real six-camera MCAP."
    )
    parser.add_argument("source", metavar="SOURCE", type=Path, help="local MCAP source")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        required=True,
        help="exact six-camera topic mapping profile",
    )
    parser.add_argument(
        "--allow-unapproved-profile",
        action="store_true",
        help="explicitly authorize a development UNAPPROVED mapping profile",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="directory for durable local canonical state",
    )
    parser.add_argument(
        "--run-key",
        default="primary",
        help="stable key for replaying one canonical run",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=_positive_seconds,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help="analyze at most this many seconds from the recording start",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="atomically written machine-readable profile report",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="existing v3 profile report used as the comparison baseline",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="atomically written fresh/replay or worker-scaling comparison report",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="reject a profile candidate when the repository worktree is dirty",
    )
    return parser


def _write_json(payload: object, *, stream: TextIO | None = None) -> None:
    print(canonical_json_bytes(payload).decode("utf-8"), file=stream or sys.stdout)


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = path.resolve()
    if destination.exists() and destination.is_dir():
        raise CanonicalProfileError("profile output must not be a directory")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    except OSError as error:
        raise CanonicalProfileError(f"cannot prepare profile output: {error}") from error
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except OSError as error:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise CanonicalProfileError(f"cannot publish profile output: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    max_duration_ns = args.max_duration_seconds * 1_000_000_000
    output = args.output.resolve()
    comparison_output = None if args.comparison_output is None else args.comparison_output.resolve()
    compare_with = None if args.compare_with is None else args.compare_with.resolve()
    try:
        if (compare_with is None) != (comparison_output is None):
            raise CanonicalProfileError(
                "--compare-with and --comparison-output must be supplied together"
            )
        if comparison_output is not None and comparison_output == output:
            raise CanonicalProfileError("comparison output must differ from profile output")
        if output.exists() and output.is_dir():
            raise CanonicalProfileError("profile output must not be a directory")
        if (
            comparison_output is not None
            and comparison_output.exists()
            and comparison_output.is_dir()
        ):
            raise CanonicalProfileError("comparison output must not be a directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        if comparison_output is not None:
            comparison_output.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_canonical_profile_manifest(
            repository_root=REPOSITORY_ROOT,
            source_path=args.source,
            mapping_config=args.mapping_config,
            run_key=args.run_key,
            max_duration_ns=max_duration_ns,
            allow_unapproved_profile=args.allow_unapproved_profile,
        )
        if args.require_clean and manifest.git.dirty:
            raise CanonicalProfileError(
                "profile requires a clean git worktree; commit or remove untracked changes"
            )
        state_before = snapshot_state_tree(
            args.state_dir,
            excluded_paths=tuple(path for path in (output, comparison_output) if path is not None),
            externally_owned_paths=(args.source,),
        )
    except CanonicalProfileError as error:
        _write_json(
            {
                "ok": False,
                "code": "PROFILE_PRECONDITION_FAILED",
                "detail": str(error),
            },
            stream=sys.stderr,
        )
        return 2

    recorder = RuntimeProfileRecorder()
    receipt = None
    run_error = None
    try:
        profiled_run = cast(_ProfiledCanonicalMcapRunner, run_local_canonical_mcap)
        receipt = profiled_run(
            source_path=args.source,
            mapping_config=args.mapping_config,
            state_dir=args.state_dir,
            run_key=args.run_key,
            allow_unapproved_profile=args.allow_unapproved_profile,
            max_duration_ns=max_duration_ns,
            runtime_observer=recorder,
        )
    except CanonicalLocalCompositionError as error:
        code = getattr(error.code, "value", error.code)
        run_error = CanonicalProfileRunError(
            code=str(code),
            error_type=type(error).__name__,
            detail=str(error),
        )

    observer = recorder.snapshot()
    try:
        state_after = snapshot_state_tree(
            args.state_dir,
            excluded_paths=tuple(path for path in (output, comparison_output) if path is not None),
            externally_owned_paths=(args.source,),
        )
        work_queue_after = snapshot_work_queue(args.state_dir)
        persisted_recording_ns, persisted_requested_ns = discover_canonical_profile_durations(
            args.state_dir,
            source_sha256=manifest.source.sha256,
        )
        source_span_duration_ns = unique_runtime_counter_value(
            observer,
            "source.span_duration_ns",
        )
        recording_duration_ns = unique_runtime_counter_value(
            observer,
            "source.recording_duration_ns",
        )
        requested_duration_ns = unique_runtime_counter_value(
            observer,
            "source.requested_duration_ns",
        )
        recording_duration_ns = recording_duration_ns or persisted_recording_ns
        requested_duration_ns = requested_duration_ns or persisted_requested_ns
        execution_mode: _ExecutionMode = (
            "UNKNOWN" if receipt is None else ("REPLAY" if receipt.replayed else "FRESH")
        )
        reconciliation = build_profile_reconciliation(
            observer=observer,
            state_after=state_after,
            manifest=manifest,
            execution_mode=execution_mode,
        )
        measurements = build_canonical_profile_measurements(
            observer=observer,
            state_before=state_before,
            state_after=state_after,
            manifest=manifest,
            receipt=receipt,
        )
        capacity = build_profile_capacity(
            observer=observer,
            manifest=manifest,
            receipt=receipt,
            execution_mode=execution_mode,
            recording_duration_ns=recording_duration_ns,
            requested_duration_ns=requested_duration_ns,
            measurements=measurements,
        )
        report = CanonicalProfileReport(
            schema_version="1.0",
            model_version="canonical-profile-report-v3",
            manifest=manifest,
            manifest_sha256=manifest.manifest_sha256,
            observer=observer,
            state_before=state_before,
            state_after=state_after,
            state_file_count_delta=state_after.file_count - state_before.file_count,
            state_byte_count_delta=state_after.byte_count - state_before.byte_count,
            work_queue_after=work_queue_after,
            receipt=receipt,
            error=run_error,
            execution_mode=execution_mode,
            source_span_duration_ns=source_span_duration_ns,
            recording_duration_ns=recording_duration_ns,
            requested_duration_ns=requested_duration_ns,
            reconciliation=reconciliation,
            measurements=measurements,
            capacity=capacity,
        )
        comparison_payload = None
        if compare_with is not None and comparison_output is not None:
            try:
                baseline = CanonicalProfileReport.model_validate_json(compare_with.read_bytes())
            except OSError as error:
                raise CanonicalProfileError(f"cannot read comparison baseline: {error}") from error
            comparison = compare_canonical_profile_reports(baseline, report)
            comparison_payload = canonical_json_bytes(comparison.model_dump(mode="json"))
        payload = canonical_json_bytes(report.model_dump(mode="json"))
        _atomic_write(output, payload)
        if comparison_output is not None and comparison_payload is not None:
            _atomic_write(comparison_output, comparison_payload)
    except (CanonicalProfileError, OSError, TypeError, ValueError) as error:
        _write_json(
            {
                "ok": False,
                "code": "PROFILE_PUBLICATION_FAILED",
                "detail": str(error),
            },
            stream=sys.stderr,
        )
        return 2

    _write_json(report.model_dump(mode="json"))
    return 0 if receipt is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
