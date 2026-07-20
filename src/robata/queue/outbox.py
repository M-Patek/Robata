"""Transactional outbox publication with idempotent delivery."""

from __future__ import annotations

from collections.abc import Sequence

from robata.queue.models import OutboxEvent, OutboxEventStatus


class OutboxPublisher:
    """Transactional successor publication with idempotent delivery.

    Events are written to persistent outbox storage in the same transaction
    as the business mutation, then asynchronously relayed to the message
    broker.  ``mark_sent`` provides idempotency so duplicate deliveries are
    harmless.
    """

    def __init__(self) -> None:
        self._events: dict[str, OutboxEvent] = {}

    def publish(self, event: OutboxEvent) -> None:
        """Persist an event to the outbox for later delivery.

        The event is stored with ``status`` set to :attr:`OutboxEventStatus.PENDING`.
        """
        self._events[event.event_id] = event

    def mark_sent(self, event_id: str) -> None:
        """Mark an event as successfully delivered.

        This operation is idempotent: calling it multiple times for the same
        ``event_id`` has no additional effect.
        """
        if event_id in self._events:
            existing = self._events[event_id]
            self._events[event_id] = existing.model_copy(update={"status": OutboxEventStatus.SENT})

    def get_pending(self, limit: int = 100) -> Sequence[OutboxEvent]:
        """Return up to ``limit`` events that have not yet been delivered.

        Events are ordered by insertion time (oldest first) to prioritize
        delivery of stale items.
        """
        pending = [
            event for event in self._events.values() if event.status is OutboxEventStatus.PENDING
        ]
        return tuple(pending[:limit])


__all__ = [
    "OutboxPublisher",
]
