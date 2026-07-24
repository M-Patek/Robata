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

    def append_accepted_lineage(
        self,
        parsed_claim: ParsedProviderClaimArtifact,
        selected_output: SelectedAttemptOutput,
        enriched_output: OrchestratorEnrichedOutput,
    ) -> tuple[
        ParsedProviderClaimArtifact,
        SelectedAttemptOutput,
        OrchestratorEnrichedOutput,
    ]:
        """Atomically append or replay one complete accepted-call lineage."""
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

    def append_accepted_lineage(
        self,
        parsed_claim: ParsedProviderClaimArtifact,
        selected_output: SelectedAttemptOutput,
        enriched_output: OrchestratorEnrichedOutput,
    ) -> tuple[
        ParsedProviderClaimArtifact,
        SelectedAttemptOutput,
        OrchestratorEnrichedOutput,
    ]:
        if not isinstance(parsed_claim, ParsedProviderClaimArtifact):
            raise TypeError("parsed_claim must be a ParsedProviderClaimArtifact")
        if not isinstance(selected_output, SelectedAttemptOutput):
            raise TypeError("selected_output must be a SelectedAttemptOutput")
        if not isinstance(enriched_output, OrchestratorEnrichedOutput):
            raise TypeError("enriched_output must be an OrchestratorEnrichedOutput")
        if (
            selected_output.parsed_claim_artifact_id != parsed_claim.artifact_id
            or selected_output.raw_response_artifact_id != parsed_claim.raw_response.artifact_id
            or selected_output.inference_id != parsed_claim.raw_response.inference_id
            or enriched_output.selected_attempt != selected_output
            or enriched_output.provider_claim_schema != parsed_claim.provider_claim_schema
        ):
            raise InferenceEvidenceStoreError("accepted-call evidence lineage is inconsistent")

        with self._lock:
            parsed_identity = (
                parsed_claim.raw_response.artifact_id,
                parsed_claim.provider_claim_schema.sha256,
                parsed_claim.parser_version,
            )
            parsed_candidates = (
                self._parsed.get(parsed_claim.artifact_id),
                next(
                    (
                        item
                        for item in self._parsed.values()
                        if (
                            item.raw_response.artifact_id,
                            item.provider_claim_schema.sha256,
                            item.parser_version,
                        )
                        == parsed_identity
                    ),
                    None,
                ),
            )
            selected_candidates = (
                self._selected.get(selected_output.selection_id),
                next(
                    (
                        item
                        for item in self._selected.values()
                        if item.output_sha256 == selected_output.output_sha256
                    ),
                    None,
                ),
            )
            enriched_candidates = (
                self._enriched.get(enriched_output.artifact_id),
                next(
                    (
                        item
                        for item in self._enriched.values()
                        if item.enrichment_logical_key == enriched_output.enrichment_logical_key
                    ),
                    None,
                ),
                next(
                    (
                        item
                        for item in self._enriched.values()
                        if item.semantic_sha256 == enriched_output.semantic_sha256
                    ),
                    None,
                ),
            )
            for candidates, candidate, conflict in (
                (
                    parsed_candidates,
                    parsed_claim,
                    "parsed provider claim identity has conflicting content",
                ),
                (
                    selected_candidates,
                    selected_output,
                    "selected attempt output identity has conflicting content",
                ),
                (
                    enriched_candidates,
                    enriched_output,
                    "enriched output identity has conflicting content",
                ),
            ):
                if any(item is not None and item != candidate for item in candidates):
                    raise InferenceEvidenceStoreError(conflict)

            stored_parsed = next(
                (item for item in parsed_candidates if item is not None), parsed_claim
            )
            stored_selected = next(
                (item for item in selected_candidates if item is not None), selected_output
            )
            stored_enriched = next(
                (item for item in enriched_candidates if item is not None), enriched_output
            )
            self._parsed[parsed_claim.artifact_id] = stored_parsed
            self._selected[selected_output.selection_id] = stored_selected
            self._enriched[enriched_output.artifact_id] = stored_enriched
            return stored_parsed, stored_selected, stored_enriched


__all__ = [
    "InMemoryInferenceEvidenceStore",
    "InferenceEvidenceStore",
    "InferenceEvidenceStoreError",
]
