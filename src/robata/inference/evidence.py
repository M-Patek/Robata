"""Persistence port for restartable post-selection inference evidence."""

from __future__ import annotations

from threading import RLock
from typing import Protocol, runtime_checkable

from robata.inference.enrichment import (
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    SelectedAttemptOutput,
)


class InferenceEvidenceStoreError(RuntimeError):
    """Persisted post-selection evidence is inconsistent."""


@runtime_checkable
class InferenceEvidenceStore(Protocol):
    """Append-only evidence required to resume after a selected attempt."""

    def append_parsed_claim(
        self, artifact: ParsedProviderClaimArtifact
    ) -> ParsedProviderClaimArtifact:
        """Append or return the identical parsed provider claim."""
        ...

    def get_parsed_claim(self, artifact_id: str) -> ParsedProviderClaimArtifact | None:
        """Return a parsed provider claim by artifact identity."""
        ...

    def append_selected_output(self, output: SelectedAttemptOutput) -> SelectedAttemptOutput:
        """Append or return the identical selected-attempt output."""
        ...

    def get_selected_output(self, selection_id: str) -> SelectedAttemptOutput | None:
        """Return the output bound to one selection identity."""
        ...

    def append_enriched_output(
        self, output: OrchestratorEnrichedOutput
    ) -> OrchestratorEnrichedOutput:
        """Append or return the identical enriched output."""
        ...

    def get_enriched_output(self, artifact_id: str) -> OrchestratorEnrichedOutput | None:
        """Return an enriched output by artifact identity."""
        ...


class InMemoryInferenceEvidenceStore:
    """Thread-safe append-only reference store for local execution."""

    def __init__(self) -> None:
        self._parsed: dict[str, ParsedProviderClaimArtifact] = {}
        self._selected: dict[str, SelectedAttemptOutput] = {}
        self._enriched: dict[str, OrchestratorEnrichedOutput] = {}
        self._lock = RLock()

    def append_parsed_claim(
        self, artifact: ParsedProviderClaimArtifact
    ) -> ParsedProviderClaimArtifact:
        if not isinstance(artifact, ParsedProviderClaimArtifact):
            raise TypeError("artifact must be a ParsedProviderClaimArtifact")
        with self._lock:
            existing = self._parsed.get(artifact.artifact_id)
            if existing is not None and existing != artifact:
                raise InferenceEvidenceStoreError(
                    "parsed provider claim identity has conflicting content"
                )
            self._parsed[artifact.artifact_id] = artifact
            return existing or artifact

    def get_parsed_claim(self, artifact_id: str) -> ParsedProviderClaimArtifact | None:
        with self._lock:
            return self._parsed.get(artifact_id)

    def append_selected_output(self, output: SelectedAttemptOutput) -> SelectedAttemptOutput:
        if not isinstance(output, SelectedAttemptOutput):
            raise TypeError("output must be a SelectedAttemptOutput")
        with self._lock:
            existing = self._selected.get(output.selection_id)
            if existing is not None and existing != output:
                raise InferenceEvidenceStoreError(
                    "selected attempt output identity has conflicting content"
                )
            self._selected[output.selection_id] = output
            return existing or output

    def get_selected_output(self, selection_id: str) -> SelectedAttemptOutput | None:
        with self._lock:
            return self._selected.get(selection_id)

    def append_enriched_output(
        self, output: OrchestratorEnrichedOutput
    ) -> OrchestratorEnrichedOutput:
        if not isinstance(output, OrchestratorEnrichedOutput):
            raise TypeError("output must be an OrchestratorEnrichedOutput")
        with self._lock:
            existing = self._enriched.get(output.artifact_id)
            if existing is not None and existing != output:
                raise InferenceEvidenceStoreError(
                    "enriched output identity has conflicting content"
                )
            self._enriched[output.artifact_id] = output
            return existing or output

    def get_enriched_output(self, artifact_id: str) -> OrchestratorEnrichedOutput | None:
        with self._lock:
            return self._enriched.get(artifact_id)


__all__ = [
    "InMemoryInferenceEvidenceStore",
    "InferenceEvidenceStore",
    "InferenceEvidenceStoreError",
]
