"""Leaf modules for the canonical offline application flow."""

__all__ = [
    "CanonicalLocalFixtureJob",
    "CanonicalLocalProviderQueue",
    "CanonicalLocalProviderQueueSnapshot",
    "CanonicalLocalRecordingService",
    "CanonicalLocalRecordingServiceSnapshot",
    "run_local_canonical_fixtures",
]


def __getattr__(name: str) -> object:
    """Load parallel service symbols without importing optional media adapters."""

    if name not in __all__:
        raise AttributeError(name)
    from robata.application.canonical.parallel_service import (
        CanonicalLocalFixtureJob,
        CanonicalLocalProviderQueue,
        CanonicalLocalProviderQueueSnapshot,
        CanonicalLocalRecordingService,
        CanonicalLocalRecordingServiceSnapshot,
        run_local_canonical_fixtures,
    )

    return {
        "CanonicalLocalFixtureJob": CanonicalLocalFixtureJob,
        "CanonicalLocalProviderQueue": CanonicalLocalProviderQueue,
        "CanonicalLocalProviderQueueSnapshot": CanonicalLocalProviderQueueSnapshot,
        "CanonicalLocalRecordingService": CanonicalLocalRecordingService,
        "CanonicalLocalRecordingServiceSnapshot": CanonicalLocalRecordingServiceSnapshot,
        "run_local_canonical_fixtures": run_local_canonical_fixtures,
    }[name]
