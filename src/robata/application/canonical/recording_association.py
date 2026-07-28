"""Durable, nonblocking publication for recording-association reports.

The recording reduction remains the authority for primary completion.  This
module is deliberately a separate, pull/worker-facing persistence boundary:
it accepts only a fully derived association report, writes exact canonical
bytes once, and can replay those bytes after a process restart.  It has no
dependency on the completion outbox or on released recording-result versions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes
from robata.event_pipeline.recording_association import (
    CompletedRecordingAssociationBinding,
    RecordingAssociationBridgeEvidence,
    RecordingAssociationEngine,
    RecordingAssociationInput,
    RecordingAssociationReport,
    verify_recording_association_report,
)
from robata.tempfiles import make_temp_file


class RecordingAssociationReportError(RuntimeError):
    """Base error for the detached recording-association report boundary."""


class RecordingAssociationReportConflict(RecordingAssociationReportError):
    """A content-addressed report path already contains different bytes."""


class RecordingAssociationReportStorageError(RecordingAssociationReportError):
    """A stored report is malformed, noncanonical, or cannot be persisted."""


class RecordingAssociationPublicationStatus(StrEnum):
    """Result of one detached report-publication attempt."""

    PUBLISHED = "PUBLISHED"
    REPLAYED = "REPLAYED"
    NO_ACCEPTED_EVIDENCE = "NO_ACCEPTED_EVIDENCE"


class RecordingAssociationPublicationResult(StrictModel):
    """The immutable report and whether its exact bytes were already present."""

    status: RecordingAssociationPublicationStatus
    report: RecordingAssociationReport | None
    replayed: bool

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status is RecordingAssociationPublicationStatus.NO_ACCEPTED_EVIDENCE:
            if self.report is not None or self.replayed:
                raise ValueError("no-accepted-evidence result cannot carry a report or replay")
            return self
        if self.report is None:
            raise ValueError("published or replayed result requires a report")
        expected = (
            RecordingAssociationPublicationStatus.REPLAYED
            if self.replayed
            else RecordingAssociationPublicationStatus.PUBLISHED
        )
        if self.status is not expected:
            raise ValueError("publication status must match replayed")
        return self


class RecordingAssociationReportStore:
    """Exact-canonical, content-addressed storage for derived reports.

    Reports are intentionally stored independently of primary recording
    completion files.  A caller may invoke this store from an asynchronous
    worker after completion has become durable; failure or delay here cannot
    alter that completed run.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).absolute()
        _ensure_regular_directory(self._root, "recording association report store root")
        _ensure_regular_directory(self._reports_directory, "recording association report directory")

    @property
    def root(self) -> Path:
        """Return the store root without resolving any potentially hostile link."""

        return self._root

    @property
    def _reports_directory(self) -> Path:
        return self._root / "reports"

    def report_path(self, semantic_sha256: Sha256Digest | str) -> Path:
        """Return the deterministic path for one report semantic digest."""

        digest = _validate_digest(semantic_sha256)
        return self._reports_directory / f"{digest}.json"

    def put_or_get(
        self,
        report: RecordingAssociationReport,
    ) -> tuple[RecordingAssociationReport, bool]:
        """Publish a report once, or replay only byte-identical canonical JSON."""

        checked = _require_report(report)
        path = self.report_path(checked.semantic_sha256)
        expected = canonical_json_bytes(checked)
        stored, replayed = self._publish_or_read(path, expected)
        if (
            stored != checked
            or stored.semantic_sha256 != checked.semantic_sha256
            or stored.logical_key != checked.logical_key
        ):
            raise RecordingAssociationReportConflict(
                "report path contains different immutable canonical bytes"
            )
        return stored, replayed

    def get(self, semantic_sha256: Sha256Digest | str) -> RecordingAssociationReport | None:
        """Load the exact report identified by its semantic content digest."""

        digest = _validate_digest(semantic_sha256)
        path = self.report_path(digest)
        if not path.exists() and not path.is_symlink():
            return None
        report = self._load(path)
        if report.semantic_sha256 != digest:
            raise RecordingAssociationReportStorageError(
                "report path does not match its semantic content digest"
            )
        return report

    def _publish_or_read(
        self,
        path: Path,
        expected: bytes,
    ) -> tuple[RecordingAssociationReport, bool]:
        _ensure_regular_directory(path.parent, "recording association report directory")
        if path.exists() or path.is_symlink():
            actual = self._read_exact(path)
            if actual != expected:
                raise RecordingAssociationReportConflict(
                    "existing report contains different immutable canonical bytes"
                )
            return _parse_exact_report(actual), True

        descriptor, temporary = make_temp_file(
            path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                actual = self._read_exact(path)
                if actual != expected:
                    raise RecordingAssociationReportConflict(
                        "concurrent report contains different immutable canonical bytes"
                    ) from None
                return _parse_exact_report(actual), True
            return _parse_exact_report(expected), False
        finally:
            temporary.unlink(missing_ok=True)

    def _load(self, path: Path) -> RecordingAssociationReport:
        return _parse_exact_report(self._read_exact(path))

    @staticmethod
    def _read_exact(path: Path) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise RecordingAssociationReportStorageError("report is not a regular file")
        try:
            return path.read_bytes()
        except OSError as error:
            raise RecordingAssociationReportStorageError(f"cannot read report: {error}") from error


class CanonicalRecordingAssociationBridge:
    """Derive and publish a report without coupling it to primary completion.

    The engine consumes explicit accepted evidence and an already-completed
    recording binding.  This bridge does not enqueue, amend, or wait on any
    recording completion operation; applications may call it only from their
    independent derived-report worker.
    """

    def __init__(
        self,
        engine: RecordingAssociationEngine,
        store: RecordingAssociationReportStore,
    ) -> None:
        if not isinstance(engine, RecordingAssociationEngine):
            raise TypeError("engine must be a RecordingAssociationEngine")
        if not isinstance(store, RecordingAssociationReportStore):
            raise TypeError("store must be a RecordingAssociationReportStore")
        self._engine = engine
        self._store = store

    def derive_and_publish(
        self,
        *,
        recording: CompletedRecordingAssociationBinding,
        inputs: Sequence[RecordingAssociationInput],
        bridge_evidence: Sequence[RecordingAssociationBridgeEvidence] = (),
    ) -> RecordingAssociationPublicationResult:
        """Build one detached report and atomically publish its exact bytes."""

        if not isinstance(recording, CompletedRecordingAssociationBinding):
            raise TypeError("recording must be a CompletedRecordingAssociationBinding")
        frozen_inputs = tuple(inputs)
        frozen_bridges = tuple(bridge_evidence)
        if not frozen_inputs:
            if frozen_bridges:
                raise ValueError("bridge_evidence cannot exist without accepted source inputs")
            return RecordingAssociationPublicationResult(
                status=RecordingAssociationPublicationStatus.NO_ACCEPTED_EVIDENCE,
                report=None,
                replayed=False,
            )
        report = self._engine.derive(
            recording=recording,
            inputs=frozen_inputs,
            bridge_evidence=frozen_bridges,
        )
        return self.publish(report)

    def publish(
        self,
        report: RecordingAssociationReport,
    ) -> RecordingAssociationPublicationResult:
        """Publish a precomputed report, retaining its event-neutral outcome verbatim."""

        stored, replayed = self._store.put_or_get(report)
        return RecordingAssociationPublicationResult(
            status=(
                RecordingAssociationPublicationStatus.REPLAYED
                if replayed
                else RecordingAssociationPublicationStatus.PUBLISHED
            ),
            report=stored,
            replayed=replayed,
        )


def _ensure_regular_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RecordingAssociationReportStorageError(f"cannot create {label}: {error}") from error
    if path.is_symlink() or not path.is_dir():
        raise RecordingAssociationReportStorageError(f"{label} must be a regular directory")


def _validate_digest(value: Sha256Digest | str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("semantic_sha256 must be a lowercase SHA-256 digest")
    return value


def _parse_exact_report(raw: bytes) -> RecordingAssociationReport:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RecordingAssociationReportStorageError(f"invalid report JSON: {error}") from error
    if not isinstance(document, dict):
        raise RecordingAssociationReportStorageError("report root must be an object")
    if canonical_json_bytes(document) != raw:
        raise RecordingAssociationReportStorageError("report bytes are not exact canonical JSON")
    try:
        report = RecordingAssociationReport.model_validate_json(raw)
    except ValueError as error:
        raise RecordingAssociationReportStorageError(f"invalid report: {error}") from error
    try:
        verified = verify_recording_association_report(report)
    except ValueError as error:
        raise RecordingAssociationReportStorageError(
            f"report does not reproduce from accepted evidence: {error}"
        ) from error
    if canonical_json_bytes(verified) != raw:
        raise RecordingAssociationReportStorageError("report bytes are inconsistent with its model")
    return verified


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _require_report(value: object) -> RecordingAssociationReport:
    if not isinstance(value, RecordingAssociationReport):
        raise TypeError("report must be a RecordingAssociationReport")
    try:
        # ``model_copy(update=...)`` bypasses Pydantic validation.  Rehydrating
        # the exact JSON form prevents such an in-memory mutation from gaining
        # a durable report path.
        return verify_recording_association_report(
            RecordingAssociationReport.model_validate_json(canonical_json_bytes(value))
        )
    except ValueError as error:
        raise ValueError(f"invalid report: {error}") from error


__all__ = [
    "CanonicalRecordingAssociationBridge",
    "RecordingAssociationPublicationResult",
    "RecordingAssociationPublicationStatus",
    "RecordingAssociationReportConflict",
    "RecordingAssociationReportError",
    "RecordingAssociationReportStorageError",
    "RecordingAssociationReportStore",
]
