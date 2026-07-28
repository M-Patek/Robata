"""Strict read-only projection from committed local completion state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from robata.application.canonical.primary_completion import CommittedPrimaryCompletion
from robata.contracts.hashing import exact_bytes_sha256
from robata.web_api.models import (
    CameraQualityView,
    EvidenceView,
    NanosecondIntervalView,
    PipelineStageView,
    RunDecisionView,
    RunHypothesisView,
    RunIntegrityView,
    RunListResponse,
    RunPackageView,
    RunPublicationView,
    RunSnapshotResponse,
    RunSnapshotView,
    RunSummaryView,
    RunWindowView,
)


class LocalStateUnavailable(RuntimeError):
    """The local completion store cannot be opened for a read-only query."""


class LocalStateIntegrityError(RuntimeError):
    """Committed bytes do not pass the repository's integrity boundary."""


class RunNotFound(LookupError):
    """The requested run has no committed completion record."""


class ReadOnlyLocalRunProjection:
    """Project immutable primary completions without invoking worker repositories."""

    def __init__(self, state_dir: Path) -> None:
        if not isinstance(state_dir, Path):
            raise TypeError("state_dir must be pathlib.Path")
        self._state_dir = state_dir.resolve()
        self._database_path = self._state_dir / "primary-completion.sqlite3"

    @property
    def database_path(self) -> Path:
        return self._database_path

    def health_check(self) -> None:
        with self._connection() as connection:
            connection.execute("SELECT 1").fetchone()

    def list_runs(self) -> RunListResponse:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT run_id, committed_json, committed_json_sha256 FROM primary_completions"
            ).fetchall()
        summaries = tuple(self._summary(committed) for committed, _ in map(self._decode_row, rows))
        return RunListResponse(
            runs=tuple(
                sorted(
                    summaries,
                    key=lambda item: (item.completed_at or "", item.run_id),
                    reverse=True,
                )
            )
        )

    def snapshot(self, run_id: str) -> RunSnapshotResponse:
        if not isinstance(run_id, str) or not run_id:
            raise RunNotFound("run_id must be a nonempty string")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT run_id, committed_json, committed_json_sha256
                FROM primary_completions
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise RunNotFound(f"no committed completion exists for run {run_id}")
        committed, cursor = self._decode_row(row)
        return RunSnapshotResponse(cursor=cursor, run=self._snapshot_view(committed))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if not self._database_path.is_file():
            raise LocalStateUnavailable(
                f"committed completion database is unavailable: {self._database_path}"
            )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                f"{self._database_path.as_uri()}?mode=ro",
                uri=True,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.execute("PRAGMA busy_timeout = 1000")
            yield connection
        except sqlite3.Error as error:
            raise LocalStateUnavailable(
                f"cannot read committed completion state: {error}"
            ) from error
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> tuple[CommittedPrimaryCompletion, str]:
        raw_value = row["committed_json"]
        stored_digest = row["committed_json_sha256"]
        stored_run_id = row["run_id"]
        if not isinstance(raw_value, bytes) or not isinstance(stored_digest, str):
            raise LocalStateIntegrityError("completion row has an invalid committed payload")
        actual_digest = exact_bytes_sha256(raw_value)
        if actual_digest != stored_digest:
            raise LocalStateIntegrityError(
                "completion payload SHA-256 does not match its stored digest"
            )
        try:
            committed = CommittedPrimaryCompletion.model_validate_json(raw_value, strict=True)
        except ValidationError as error:
            raise LocalStateIntegrityError(
                "completion payload violates its canonical contract"
            ) from error
        if not isinstance(stored_run_id, str) or committed.processing_run.run_id != stored_run_id:
            raise LocalStateIntegrityError(
                "completion row run ID disagrees with its committed payload"
            )
        return committed, actual_digest

    @staticmethod
    def _summary(committed: CommittedPrimaryCompletion) -> RunSummaryView:
        detail = committed.detail
        processing_run = committed.processing_run
        decision = detail.output_decision
        return RunSummaryView(
            run_id=str(processing_run.run_id),
            recording_identity=str(processing_run.recording_identity),
            status=str(detail.status),
            started_at=processing_run.started_at,
            completed_at=processing_run.completed_at,
            pipeline_version=str(processing_run.pipeline_version),
            output_decision=None if decision is None else str(decision.decision),
            event_count=len(detail.action_event_publications.publications),
        )

    def _snapshot_view(self, committed: CommittedPrimaryCompletion) -> RunSnapshotView:
        detail = committed.detail
        summary = self._summary(committed)
        decision = detail.output_decision
        return RunSnapshotView(
            **summary.model_dump(),
            evidence_class=str(detail.evidence_class),
            production_eligible=detail.production_eligible,
            window=self._window(detail.window),
            packages=tuple(self._package(item) for item in detail.package_set.members),
            camera_quality=tuple(
                CameraQualityView(
                    camera_id=str(item.camera_id),
                    status=str(item.local_status),
                    interval=self._interval(item.interval),
                )
                for item in detail.qa_completion_result.coarse_coverage
            ),
            stages=self._stages(detail),
            decision=(
                None
                if decision is None
                else RunDecisionView(
                    decision=str(decision.decision),
                    reason_code=str(decision.reason_code),
                    admitted_claim_count=len(decision.admitted_claim_ordinals),
                )
            ),
            hypotheses=tuple(
                RunHypothesisView(
                    ordinal=ordinal,
                    logical_key=str(item.event_hypothesis_logical_key),
                    semantic_sha256=str(item.semantic_sha256),
                    effective_interval=self._interval(item.effective_interval),
                )
                for ordinal, item in enumerate(detail.hypotheses)
            ),
            publications=tuple(
                RunPublicationView(
                    event_id=str(item.current_revision.event_id),
                    revision_id=str(item.revision.revision_id),
                    effective_interval=self._interval(item.current_revision.effective_interval),
                )
                for item in detail.action_event_publications.publications
            ),
            integrity=RunIntegrityView(
                command_sha256=str(committed.command_sha256),
                completion_semantic_sha256=str(committed.completion.semantic_sha256),
            ),
            evidence=tuple(
                EvidenceView(
                    role=str(item.role),
                    schema_id=str(item.schema_ref.schema_id),
                    schema_version=str(item.schema_ref.version),
                    semantic_sha256=str(item.semantic_sha256),
                    exact_bytes_sha256=str(item.exact_bytes_sha256),
                    byte_count=item.byte_count,
                )
                for item in committed.evidence_references
            ),
        )

    @staticmethod
    def _interval(interval: Any) -> NanosecondIntervalView:
        return NanosecondIntervalView(start_ns=str(interval.start_ns), end_ns=str(interval.end_ns))

    def _window(self, window: Any) -> RunWindowView:
        return RunWindowView(
            logical_key=str(window.window_logical_key),
            purpose=str(window.purpose),
            requested_interval=self._interval(window.requested_interval),
            effective_interval=self._interval(window.interval),
            recording_duration_ns=str(window.recording_duration_ns),
        )

    def _package(self, package: Any) -> RunPackageView:
        return RunPackageView(
            package_id=str(package.package_id),
            ordinal=package.ordinal,
            part_count=package.part_count,
            interval=NanosecondIntervalView(
                start_ns=str(package.start_ns),
                end_ns=str(package.end_ns),
            ),
        )

    def _stages(self, detail: Any) -> tuple[PipelineStageView, ...]:
        qa_completion = detail.qa_completion_result
        return (
            self._stage("Root window", detail.window, detail.window.semantic_sha256),
            self._stage("Package set", detail.package_set, None),
            self._stage(
                "Coarse QA",
                detail.coarse_qa_result,
                qa_completion.coarse_result_semantic_sha256,
            ),
            self._stage("QA completion", qa_completion, qa_completion.semantic_sha256),
            self._stage(
                "Event proposal",
                detail.event_proposal_result,
                detail.event_proposal_result.semantic_sha256,
            ),
            self._stage(
                "Candidate reduction",
                detail.candidate_reduction_result,
                detail.candidate_reduction_result.semantic_sha256,
            ),
            self._stage("Provisional fusion", detail.provisional_fusion_result),
            self._stage(
                "Boundary refinement",
                detail.boundary_refinement_executions or None,
            ),
            self._stage("Final fusion", detail.final_fusion_context),
            self._stage("Output admission", detail.output_decision),
            self._stage("Action publications", detail.action_event_publications),
        )

    @staticmethod
    def _stage(
        name: str,
        value: Any | None,
        semantic_sha256: Any | None = None,
    ) -> PipelineStageView:
        digest = (
            semantic_sha256
            if semantic_sha256 is not None
            else getattr(value, "semantic_sha256", None)
        )
        return PipelineStageView(
            name=name,
            state="COMPLETE" if value is not None else "NOT_RUN",
            semantic_sha256=None if digest is None else str(digest),
        )
