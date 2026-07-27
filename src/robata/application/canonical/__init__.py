"""Leaf modules for the canonical offline application flow."""

__all__ = [
    "CANONICAL_EVENT_INDEX_PROJECTION_VERSION",
    "CameraCalibrationProfile",
    "CanonicalLocalFixtureJob",
    "CanonicalLocalProviderQueue",
    "CanonicalLocalProviderQueueSnapshot",
    "CanonicalLocalRecordingService",
    "CanonicalLocalRecordingServiceSnapshot",
    "CanonicalRecoveryEvidenceClass",
    "CanonicalRecoveryQualificationEvidence",
    "CanonicalRecoveryReceiptEvidence",
    "CanonicalRecoveryScenario",
    "DerivedGeometryArtifact",
    "DerivedGeometryFrame",
    "FramePreprocessPolicy",
    "GeometryInterpolation",
    "GeometryMapCache",
    "GeometryProcessingError",
    "GeometryView",
    "build_canonical_recovery_qualification_evidence",
    "canonical_event_index_batch_projection",
    "canonical_event_index_projection",
    "canonical_event_index_projection_batch",
    "canonical_event_index_projection_values",
    "canonical_event_index_revision_projection",
    "canonical_event_index_row_projection",
    "canonical_terminal_event_index_projection",
    "materialize_geometry_view",
    "run_local_canonical_fixtures",
]


def __getattr__(name: str) -> object:
    """Load parallel service symbols without importing optional media adapters."""

    if name not in __all__:
        raise AttributeError(name)
    if name in {
        "CanonicalRecoveryEvidenceClass",
        "CanonicalRecoveryQualificationEvidence",
        "CanonicalRecoveryReceiptEvidence",
        "CanonicalRecoveryScenario",
        "build_canonical_recovery_qualification_evidence",
    }:
        from robata.application.canonical.qualification_evidence import (
            CanonicalRecoveryEvidenceClass,
            CanonicalRecoveryQualificationEvidence,
            CanonicalRecoveryReceiptEvidence,
            CanonicalRecoveryScenario,
            build_canonical_recovery_qualification_evidence,
        )

        return {
            "CanonicalRecoveryEvidenceClass": CanonicalRecoveryEvidenceClass,
            "CanonicalRecoveryQualificationEvidence": CanonicalRecoveryQualificationEvidence,
            "CanonicalRecoveryReceiptEvidence": CanonicalRecoveryReceiptEvidence,
            "CanonicalRecoveryScenario": CanonicalRecoveryScenario,
            "build_canonical_recovery_qualification_evidence": (
                build_canonical_recovery_qualification_evidence
            ),
        }[name]
    if name in {
        "CANONICAL_EVENT_INDEX_PROJECTION_VERSION",
        "canonical_event_index_batch_projection",
        "canonical_event_index_projection",
        "canonical_event_index_projection_batch",
        "canonical_event_index_projection_values",
        "canonical_event_index_revision_projection",
        "canonical_event_index_row_projection",
        "canonical_terminal_event_index_projection",
    }:
        from robata.application.canonical.projections import (
            CANONICAL_EVENT_INDEX_PROJECTION_VERSION,
            canonical_event_index_batch_projection,
            canonical_event_index_projection,
            canonical_event_index_projection_batch,
            canonical_event_index_projection_values,
            canonical_event_index_revision_projection,
            canonical_event_index_row_projection,
            canonical_terminal_event_index_projection,
        )

        return {
            "CANONICAL_EVENT_INDEX_PROJECTION_VERSION": CANONICAL_EVENT_INDEX_PROJECTION_VERSION,
            "canonical_event_index_batch_projection": canonical_event_index_batch_projection,
            "canonical_event_index_projection": canonical_event_index_projection,
            "canonical_event_index_projection_batch": canonical_event_index_projection_batch,
            "canonical_event_index_projection_values": canonical_event_index_projection_values,
            "canonical_event_index_revision_projection": canonical_event_index_revision_projection,
            "canonical_event_index_row_projection": canonical_event_index_row_projection,
            "canonical_terminal_event_index_projection": canonical_terminal_event_index_projection,
        }[name]
    if name in {
        "CameraCalibrationProfile",
        "DerivedGeometryArtifact",
        "DerivedGeometryFrame",
        "FramePreprocessPolicy",
        "GeometryInterpolation",
        "GeometryMapCache",
        "GeometryProcessingError",
        "GeometryView",
        "materialize_geometry_view",
    }:
        from robata.application.canonical.media_geometry import (
            CameraCalibrationProfile,
            DerivedGeometryArtifact,
            DerivedGeometryFrame,
            FramePreprocessPolicy,
            GeometryInterpolation,
            GeometryMapCache,
            GeometryProcessingError,
            GeometryView,
            materialize_geometry_view,
        )

        return {
            "CameraCalibrationProfile": CameraCalibrationProfile,
            "DerivedGeometryArtifact": DerivedGeometryArtifact,
            "DerivedGeometryFrame": DerivedGeometryFrame,
            "FramePreprocessPolicy": FramePreprocessPolicy,
            "GeometryInterpolation": GeometryInterpolation,
            "GeometryMapCache": GeometryMapCache,
            "GeometryProcessingError": GeometryProcessingError,
            "GeometryView": GeometryView,
            "materialize_geometry_view": materialize_geometry_view,
        }[name]
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
