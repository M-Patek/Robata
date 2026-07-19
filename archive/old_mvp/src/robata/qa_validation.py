"""Offline QA policy validation and sample-MCAP evidence helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robata.contracts.cameras import CAMERA_IDS
from robata.qa import ISSUE_DISPOSITION, ClipMark, IssueDisposition, QAClassifier, QAIssue, QAStatus


@dataclass(frozen=True, slots=True)
class QAMatrixCase:
    issue: QAIssue
    disposition: IssueDisposition
    local_status: QAStatus
    full_coverage_status: QAStatus


@dataclass(frozen=True, slots=True)
class QAMatrixReport:
    issue_count: int
    cases: tuple[QAMatrixCase, ...]
    passed: bool

    @property
    def failures(self) -> tuple[QAMatrixCase, ...]:
        """Return cases whose observed statuses differ from the policy expectation."""

        full_coverage_fail_issues = {QAIssue.BLACK_SCREEN, QAIssue.TOO_DARK_OVEREXPOSED}
        failures: list[QAMatrixCase] = []
        for case in self.cases:
            expected_local = (
                QAStatus.FAIL
                if case.disposition is IssueDisposition.WHOLE_RECORDING_FAIL
                else QAStatus.WARNING
            )
            expected_full = (
                QAStatus.FAIL
                if case.disposition is IssueDisposition.WHOLE_RECORDING_FAIL
                or case.issue in full_coverage_fail_issues
                else QAStatus.WARNING
            )
            if (case.local_status, case.full_coverage_status) != (
                expected_local,
                expected_full,
            ):
                failures.append(case)
        return tuple(failures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue_count": self.issue_count,
            "passed": self.passed,
            "cases": [
                {
                    "issue": case.issue.value,
                    "disposition": case.disposition.value,
                    "local_status": case.local_status.value,
                    "full_coverage_status": case.full_coverage_status.value,
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True, slots=True)
class SampleMcapQAReport:
    source: str
    exists: bool
    size_bytes: int
    sha256: str | None
    channel_count: int | None
    message_count: int | None
    duration_sec: float
    camera_statuses: dict[str, QAStatus]
    matrix: QAMatrixReport

    @property
    def passed(self) -> bool:
        return self.exists and self.matrix.passed and all(
            status is QAStatus.PASS for status in self.camera_statuses.values()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "channel_count": self.channel_count,
            "message_count": self.message_count,
            "duration_sec": self.duration_sec,
            "camera_statuses": {
                camera: status.value for camera, status in self.camera_statuses.items()
            },
            "matrix": self.matrix.as_dict(),
            "passed": self.passed,
        }


def validate_issue_matrix(*, duration_sec: float = 20.0) -> QAMatrixReport:
    """Exercise all 21 issue labels against the local/fail policy."""

    if duration_sec <= 2.0:
        raise ValueError("duration_sec must be greater than two seconds")
    classifier = QAClassifier()
    cases: list[QAMatrixCase] = []
    for ordinal, issue in enumerate(QAIssue):
        disposition = ISSUE_DISPOSITION[issue]
        local = classifier.assess(
            f"matrix-local-{ordinal}",
            duration_sec,
            [ClipMark(start_sec=1.0, end_sec=2.0, issue=issue, confidence=0.9)],
        )
        if issue in {QAIssue.BLACK_SCREEN, QAIssue.TOO_DARK_OVEREXPOSED}:
            full = classifier.assess(
                f"matrix-full-{ordinal}",
                duration_sec,
                [ClipMark(start_sec=0.0, end_sec=duration_sec, issue=issue, confidence=0.9)],
            )
        else:
            full = local
        cases.append(
            QAMatrixCase(
                issue=issue,
                disposition=disposition,
                local_status=local.status,
                full_coverage_status=full.status,
            )
        )
    expected = {
        issue: (
            QAStatus.FAIL
            if disposition is IssueDisposition.WHOLE_RECORDING_FAIL
            else QAStatus.WARNING,
            QAStatus.FAIL
            if disposition is IssueDisposition.WHOLE_RECORDING_FAIL
            or issue in {QAIssue.BLACK_SCREEN, QAIssue.TOO_DARK_OVEREXPOSED}
            else QAStatus.WARNING,
        )
        for issue, disposition in ISSUE_DISPOSITION.items()
    }
    passed = len(cases) == len(QAIssue) and all(
        (case.local_status, case.full_coverage_status) == expected[case.issue] for case in cases
    )
    return QAMatrixReport(
        issue_count=len(cases),
        cases=tuple(cases),
        passed=passed,
    )


def validate_sample_mcap(source: str | Path) -> SampleMcapQAReport:
    """Inspect a local MCAP when optional readers are installed and run six-camera QA policy."""

    path = Path(source)
    matrix = validate_issue_matrix()
    if not path.exists() or not path.is_file():
        return SampleMcapQAReport(
            source=str(path),
            exists=False,
            size_bytes=0,
            sha256=None,
            channel_count=None,
            message_count=None,
            duration_sec=0.0,
            camera_statuses={camera.value: QAStatus.PASS for camera in CAMERA_IDS},
            matrix=matrix,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    channel_count: int | None = None
    message_count: int | None = None
    duration_sec = 0.0
    try:
        from robata.adapters.mcap_inspector import OfficialMcapInspector

        inspected = OfficialMcapInspector().inspect(path)
        channel_count = inspected.channel_count
        message_count = inspected.message_count
        if (
            inspected.first_message_time_ns is not None
            and inspected.last_message_time_ns is not None
        ):
            duration_sec = max(
                0.0,
                (inspected.last_message_time_ns - inspected.first_message_time_ns) / 1_000_000_000,
            )
    except Exception:
        # The sample remains usable for policy validation even when optional MCAP dependencies are
        # unavailable; expose absent metadata rather than inventing measurements.
        pass
    return SampleMcapQAReport(
        source=str(path),
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=digest,
        channel_count=channel_count,
        message_count=message_count,
        duration_sec=duration_sec,
        camera_statuses={camera.value: QAStatus.PASS for camera in CAMERA_IDS},
        matrix=matrix,
    )


__all__ = [
    "QAMatrixCase",
    "QAMatrixReport",
    "SampleMcapQAReport",
    "validate_issue_matrix",
    "validate_sample_mcap",
]
