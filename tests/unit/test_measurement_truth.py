from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import semantic_sha256
from robata.contracts.measurement_truth import (
    EvidenceClass,
    MeasurementAxes,
    MeasurementEnvironment,
    MeasurementExecutionMode,
    MeasurementStatus,
    MeasurementWorkload,
    ScopeDigestInputs,
    ScopeEvidenceRegister,
    ScopeFingerprint,
    scope_evidence_register_projection,
)
from robata.runtime.canonical_profile import canonical_profile_workload_fingerprint
from robata.runtime.measurement_truth import (
    build_profile_evidence_register,
    load_profile_evidence_register,
    repository_code_digest,
)
from tests.unit.test_canonical_profile import _v3_report


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _scope() -> ScopeFingerprint:
    return ScopeFingerprint.create(
        inputs=ScopeDigestInputs(
            code_revision="a" * 40,
            code_digest=_digest("code"),
            schema_catalog_digest=_digest("catalog"),
            workload_digest=_digest("workload"),
            policy_digest=_digest("policy"),
            identity_formula_version="recording-identity-v1",
            identity_projection_digest=_digest("identity"),
        )
    )


def _workload() -> MeasurementWorkload:
    return MeasurementWorkload(
        workload_fingerprint=_digest("workload"),
        recording_count=1,
        camera_count=6,
        recording_duration_ns=100,
        frame_count=12,
        source_bytes=10,
    )


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        provider="fixture",
        provider_mode="LOCAL_OFFLINE_FIXTURE",
        hardware="test-platform/test-machine",
    )


def _register(
    *,
    evidence_class: EvidenceClass = EvidenceClass.LOCAL_BENCHMARK,
    execution_mode: MeasurementExecutionMode = MeasurementExecutionMode.FRESH,
    observed_at: str = "2026-01-01T00:00:00Z",
    measurement_status: MeasurementStatus = MeasurementStatus.MEASURED,
) -> ScopeEvidenceRegister:
    return ScopeEvidenceRegister.create(
        scope=_scope(),
        evidence_class=evidence_class,
        execution_mode=execution_mode,
        workload=_workload(),
        environment=_environment(),
        axes=MeasurementAxes(decoded_frames=12, process_cpu_ns=10),
        observed_at=observed_at,
        measurement_status=measurement_status,
    )


def test_repository_code_digest_changes_when_code_bytes_change(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_bytes(b"version = 1\n")

    first = repository_code_digest(tmp_path)
    source.write_bytes(b"version = 2\n")

    assert repository_code_digest(tmp_path) != first


def test_default_scope_fingerprint_hashes_the_source_checkout() -> None:
    report = _v3_report(replayed=False)
    scope = build_profile_evidence_register(report).scope
    assert scope.inputs.code_digest == repository_code_digest(Path(__file__).parents[2])
    assert repository_code_digest(Path(__file__).parents[2] / 'src') == scope.inputs.code_digest


def test_scope_and_register_digests_reject_tampering() -> None:
    scope = _scope()
    with pytest.raises(ValidationError, match="scope_digest"):
        ScopeFingerprint.model_validate(
            {**scope.model_dump(mode="python"), "scope_digest": _digest("tampered")},
            strict=True,
        )

    register = _register()
    tampered_model = register.model_copy(
        update={"axes": register.axes.model_copy(update={"decoded_frames": 99})}
    )
    tampered = tampered_model.model_dump(mode="python")
    with pytest.raises(ValidationError, match="register_digest"):
        ScopeEvidenceRegister.model_validate(tampered, strict=True)


def test_fresh_and_replay_registers_keep_execution_mode_and_workload_scope() -> None:
    fresh = build_profile_evidence_register(
        _v3_report(replayed=False),
        evidence_class=EvidenceClass.LOCAL_BENCHMARK,
        observed_at="2026-01-01T00:00:00Z",
    )
    replay = build_profile_evidence_register(
        _v3_report(replayed=True),
        evidence_class=EvidenceClass.LOCAL_BENCHMARK,
        observed_at="2026-01-01T00:00:00Z",
    )

    assert fresh.execution_mode is MeasurementExecutionMode.FRESH
    assert replay.execution_mode is MeasurementExecutionMode.REPLAY
    assert fresh.scope.scope_digest == replay.scope.scope_digest
    assert fresh.workload.workload_fingerprint == replay.workload.workload_fingerprint
    assert fresh.register_digest != replay.register_digest


@pytest.mark.parametrize("evidence_class", tuple(EvidenceClass))
def test_all_evidence_classes_are_expressible_without_production_grant(
    evidence_class: EvidenceClass,
) -> None:
    status = (
        MeasurementStatus.NOT_MEASURED
        if evidence_class is EvidenceClass.LOCAL_CONFORMANCE
        else MeasurementStatus.MEASURED
    )
    register = _register(evidence_class=evidence_class, measurement_status=status)

    assert register.evidence_class is evidence_class
    assert register.measurement_status is status
    assert register.production_eligible is False
    assert register.historical_snapshot is True


def test_workload_fingerprint_is_the_scope_workload_digest() -> None:
    report = _v3_report(replayed=False)
    register = build_profile_evidence_register(
        report,
        evidence_class=EvidenceClass.LOCAL_BENCHMARK,
        observed_at="2026-01-01T00:00:00Z",
    )
    expected = canonical_profile_workload_fingerprint(report.manifest)

    assert register.workload.workload_fingerprint == expected
    assert register.scope.inputs.workload_digest == expected
    assert register.workload.camera_count == report.manifest.camera_count
    assert register.workload.source_bytes == report.manifest.source.byte_count
    assert report.capacity is not None
    assert register.workload.recording_duration_ns == report.capacity.recording_duration_ns


def test_local_conformance_is_not_measured_and_cannot_be_promoted() -> None:
    report = _v3_report(replayed=False)
    register = build_profile_evidence_register(
        report,
        evidence_class=EvidenceClass.LOCAL_CONFORMANCE,
        observed_at="2026-01-01T00:00:00Z",
    )

    assert register.measurement_status is MeasurementStatus.NOT_MEASURED
    assert register.production_eligible is False

    measured_model = register.model_copy(
        update={"measurement_status": MeasurementStatus.MEASURED}
    )
    measured = measured_model.model_dump(mode="python")
    measured["register_digest"] = semantic_sha256(
        scope_evidence_register_projection(measured_model)
    )
    with pytest.raises(ValidationError, match="local conformance evidence"):
        ScopeEvidenceRegister.model_validate(measured, strict=True)

    promoted_model = register.model_copy(update={"production_eligible": True})
    promoted = promoted_model.model_dump(mode="python")
    promoted["register_digest"] = semantic_sha256(
        scope_evidence_register_projection(promoted_model)
    )
    with pytest.raises(ValidationError, match="production_eligible"):
        ScopeEvidenceRegister.model_validate(promoted, strict=True)


def test_profile_builder_cannot_self_label_production_qualification() -> None:
    with pytest.raises(ValueError, match='external qualification gate'):
        build_profile_evidence_register(
            _v3_report(replayed=False),
            evidence_class=EvidenceClass.PRODUCTION_QUALIFIED,
            observed_at='2026-01-01T00:00:00Z',
        )


def test_explicit_empty_observation_time_is_rejected() -> None:
    with pytest.raises(ValidationError, match='observed_at'):
        ScopeEvidenceRegister.create(
            scope=_scope(),
            evidence_class=EvidenceClass.LOCAL_BENCHMARK,
            execution_mode=MeasurementExecutionMode.FRESH,
            workload=_workload(),
            environment=_environment(),
            axes=MeasurementAxes(decoded_frames=12),
            observed_at='',
            measurement_status=MeasurementStatus.MEASURED,
        )


def test_explicit_empty_environment_labels_are_rejected() -> None:
    report = _v3_report(replayed=False)
    with pytest.raises(ValidationError, match='provider'):
        build_profile_evidence_register(
            report,
            evidence_class=EvidenceClass.LOCAL_BENCHMARK,
            provider='',
            observed_at='2026-01-01T00:00:00Z',
        )
    with pytest.raises(ValidationError, match='hardware'):
        build_profile_evidence_register(
            report,
            evidence_class=EvidenceClass.LOCAL_BENCHMARK,
            hardware='',
            observed_at='2026-01-01T00:00:00Z',
        )


def test_profile_json_loader_preserves_fresh_replay_and_scope(tmp_path: Path) -> None:
    report = _v3_report(replayed=False)
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(report.model_dump_json().encode("utf-8"))

    register = load_profile_evidence_register(
        profile_path,
        repository_root=tmp_path,
        evidence_class=EvidenceClass.LOCAL_BENCHMARK,
        observed_at="2026-01-01T00:00:00Z",
    )

    assert register.execution_mode is MeasurementExecutionMode.FRESH
    assert register.measurement_status is MeasurementStatus.MEASURED
    assert register.scope.inputs.workload_digest == register.workload.workload_fingerprint
