"""Distributed queue and concurrency skeleton for the Robata pipeline.

This package provides the core abstractions for work-item lifecycle management,
stage dispatching, barrier coordination, backpressure control, and transactional
outbox publication.  Production adapters (e.g. Redis) live in sibling modules;
the contracts here are intentionally infrastructure-agnostic.
"""

from robata.queue.backpressure import (
    AdmissionDecision,
    BackpressureConfig,
    BackpressureController,
    QueueMetrics,
    SheddingAction,
)
from robata.queue.barrier import (
    AggregateStatus,
    Barrier,
    BarrierCoordinator,
    BarrierMember,
    BarrierState,
    BarrierStorage,
    InMemoryBarrierStorage,
    ReductionPolicy,
)
from robata.queue.dispatcher import (
    CapacityReservation,
    DispatcherConfig,
    DispatchResult,
    StageDispatcher,
)
from robata.queue.models import (
    DependencyCriticality,
    OutboxEvent,
    OutboxEventStatus,
    WorkBarrier,
    WorkBarrierMember,
    WorkDependency,
    WorkItem,
    WorkItemSubjectType,
)
from robata.queue.outbox import OutboxPublisher
from robata.queue.redis_adapter import RedisTaskQueue
from robata.queue.stage import Stage, StageStatus

__all__ = [
    # stage
    "Stage",
    "StageStatus",
    # models
    "WorkItem",
    "WorkItemSubjectType",
    "WorkDependency",
    "DependencyCriticality",
    "WorkBarrier",
    "WorkBarrierMember",
    "OutboxEvent",
    "OutboxEventStatus",
    # dispatcher
    "DispatcherConfig",
    "DispatchResult",
    "CapacityReservation",
    "StageDispatcher",
    # barrier
    "Barrier",
    "BarrierState",
    "BarrierStorage",
    "BarrierCoordinator",
    "BarrierMember",
    "AggregateStatus",
    "ReductionPolicy",
    "InMemoryBarrierStorage",
    # backpressure
    "BackpressureConfig",
    "QueueMetrics",
    "AdmissionDecision",
    "SheddingAction",
    "BackpressureController",
    # outbox
    "OutboxPublisher",
    # redis adapter
    "RedisTaskQueue",
]
