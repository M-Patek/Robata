"""MCAP ingestion service."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from robata.contracts.mcap import (
    MCAPManifest,
    MCAPRecording,
    MCAPRecordingStatus,
    TerminalErrorCode,
)


class MCAPIngestionService:
    """Orchestrate MCAP ingestion through the DISCOVERED..READY/INVALID state machine."""

    def __init__(self) -> None:
        self._state: MCAPRecordingStatus | None = None

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------
    def transition_to(self, new_state: MCAPRecordingStatus) -> None:
        """Advance the ingestion state machine to *new_state*."""
        self._state = new_state

    def get_state(self) -> MCAPRecordingStatus | None:
        """Return the current ingestion state, or None if not yet started."""
        return self._state

    # ------------------------------------------------------------------
    # Core ingestion steps
    # ------------------------------------------------------------------
    def discover(self, source_uri: str, observed_size_bytes: int) -> MCAPRecording:
        """Create a DISCOVERED recording from a source notification."""
        raise NotImplementedError("discover() requires external storage and ID generation.")

    def hash_source(self, recording: MCAPRecording) -> MCAPRecording:
        """Transition DISCOVERED -> HASHING and compute content SHA-256."""
        raise NotImplementedError("hash_source() requires artifact streaming and digest.")

    def inspect(self, recording: MCAPRecording) -> MCAPRecording:
        """Transition HASHING -> INSPECTING and scan MCAP channels."""
        raise NotImplementedError("inspect() requires an MCAPInspector adapter.")

    def validate_mcap(self, recording: MCAPRecording) -> MCAPRecording:
        """Transition INSPECTING -> VALIDATING -> READY or INVALID.

        Runs structural, camera-count, mapping, codec, and timestamp validation.
        """
        raise NotImplementedError("validate_mcap() requires an MCAPValidator and mapping policy.")

    def publish_ready_manifest(self, recording: MCAPRecording) -> MCAPManifest:
        """Publish the immutable MCAPManifest for a READY recording."""
        raise NotImplementedError(
            "publish_ready_manifest() requires storage, registry, and manifest artifact writes."
        )

    # ------------------------------------------------------------------
    # Terminal / quarantine helpers
    # ------------------------------------------------------------------
    def quarantine(self, recording: MCAPRecording, error_code: TerminalErrorCode) -> MCAPRecording:
        """Mark a recording as INVALID with a terminal error code."""
        raise NotImplementedError("quarantine() requires storage persistence layer.")

    def retry_or_fail(self, recording: MCAPRecording) -> MCAPRecording:
        """Handle RETRY_WAIT / FAILED transitions for infrastructure errors."""
        raise NotImplementedError("retry_or_fail() requires retry-budget and queue infrastructure.")


__all__ = [
    "MCAPIngestionService",
]
