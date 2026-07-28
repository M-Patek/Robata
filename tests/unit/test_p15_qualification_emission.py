from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robata.benchmark.p15_emission import (
    assemble_local_p15_qualification_package,
    verify_local_p15_qualification_package_artifacts,
)
from robata.benchmark.p15_qualification import (
    P15ExternalGateId,
    P15ExternalGateStatus,
    P15QualificationPackage,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
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
)
from robata.contracts.phase_contract_decisions import OptimizationPhase

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _digest(value: int) -> str:
    return f"{value:064x}"


def _scope_evidence() -> ScopeEvidenceRegister:
    workload_digest = _digest(4)
    scope = ScopeFingerprint.create(
        inputs=ScopeDigestInputs(
            code_revision="p15-local-emission-test",
            code_digest=_digest(1),
            schema_catalog_digest=_digest(2),
            workload_digest=workload_digest,
            policy_digest=_digest(3),
            identity_formula_version="1.0",
            identity_projection_digest=_digest(5),
            seam_versions=("p15:local-emission-v1",),
        )
    )
    return ScopeEvidenceRegister.create(
        scope=scope,
        evidence_class=EvidenceClass.LOCAL_CONFORMANCE,
        execution_mode=MeasurementExecutionMode.FRESH,
        workload=MeasurementWorkload(
            workload_fingerprint=workload_digest,
            recording_count=2,
            camera_count=6,
            recording_duration_ns=1_000,
        ),
        environment=MeasurementEnvironment(
            provider="local-fixture",
            provider_mode="LOCAL",
            hardware="local-cpu",
        ),
        axes=MeasurementAxes(recording_hours=1.0, camera_hours=6.0),
        observed_at="2026-01-01T00:00:00Z",
        measurement_status=MeasurementStatus.NOT_MEASURED,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )


def _write_inputs(
    tmp_path: Path,
) -> tuple[ScopeEvidenceRegister, Path, Path, dict[OptimizationPhase, Path], dict[str, object]]:
    scope = _scope_evidence()
    scope_path = tmp_path / "scope.json"
    scope_path.write_bytes(canonical_json_bytes(scope.model_dump(mode="json")) + b"\n")

    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    artifact_paths: dict[OptimizationPhase, Path] = {}
    entries: list[dict[str, str]] = []
    for phase in (phase for phase in OptimizationPhase if phase is not OptimizationPhase.P15):
        artifact_path = artifact_directory / f"{phase.value.lower()}.json"
        artifact_path.write_bytes(
            canonical_json_bytes({"phase": phase.value, "local_fixture": True}) + b"\n"
        )
        artifact_paths[phase] = artifact_path
        entries.append(
            {
                "phase": phase.value,
                "artifact_id": f"{phase.value.lower()}-local-artifact",
                "artifact_path": artifact_path.relative_to(tmp_path).as_posix(),
                "scope_digest": scope.scope.scope_digest,
                "evidence_class": EvidenceClass.LOCAL_CONFORMANCE.value,
                "measurement_status": MeasurementStatus.NOT_MEASURED.value,
                "summary": f"{phase.value} local conformance artifact",
            }
        )
    manifest_payload: dict[str, object] = {
        "manifest_version": "p15-local-qualification-manifest-v1",
        "scope_digest": scope.scope.scope_digest,
        "phase_artifacts": entries,
        "pareto_selection": {
            "source_artifact_id": "p8-local-artifact",
            "candidate_operating_point_ids": ["balanced", "high-recall"],
            "selected_operating_point_id": "balanced",
            "preference_rationale": "Retains the local Pareto frontier without a scalar score.",
        },
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest_payload)
    return scope, scope_path, manifest_path, artifact_paths, manifest_payload


def test_local_emission_assembles_real_files_and_unresolved_gates(tmp_path: Path) -> None:
    scope, scope_path, manifest_path, artifact_paths, _ = _write_inputs(tmp_path)

    package = assemble_local_p15_qualification_package(
        scope_path=scope_path,
        manifest_path=manifest_path,
    )

    assert package.scope_evidence_register == scope
    assert package.production_eligible is False
    assert package.technical_requirements_satisfied is False
    assert tuple(gate.gate_id for gate in package.external_gates) == tuple(P15ExternalGateId)
    assert (
        tuple(gate.status for gate in package.external_gates[:-1])
        == (P15ExternalGateStatus.NOT_MEASURED,) * 6
    )
    assert package.external_gates[-1].status is P15ExternalGateStatus.PENDING_INDEPENDENT_REVIEW
    assert {risk.gate_id for risk in package.unresolved_risks} == set(P15ExternalGateId)
    for artifact in package.phase_artifacts:
        assert artifact.artifact_uri.startswith("file:")
        assert artifact.artifact_sha256 == exact_bytes_sha256(
            artifact_paths[artifact.phase].read_bytes()
        )
    verify_local_p15_qualification_package_artifacts(package)


def test_local_emission_detects_tampered_referenced_artifact(tmp_path: Path) -> None:
    _, scope_path, manifest_path, artifact_paths, _ = _write_inputs(tmp_path)
    package = assemble_local_p15_qualification_package(
        scope_path=scope_path,
        manifest_path=manifest_path,
    )

    artifact_paths[OptimizationPhase.P3].write_bytes(b"tampered artifact\n")

    with pytest.raises(ValueError, match="SHA-256 changed"):
        verify_local_p15_qualification_package_artifacts(package)


def test_local_emission_rejects_missing_phase_and_scope_drift(tmp_path: Path) -> None:
    _, scope_path, manifest_path, _, manifest_payload = _write_inputs(tmp_path)

    missing_phase = copy.deepcopy(manifest_payload)
    missing_entries = missing_phase["phase_artifacts"]
    assert isinstance(missing_entries, list)
    missing_phase["phase_artifacts"] = [
        entry
        for entry in missing_entries
        if isinstance(entry, dict) and entry["phase"] != OptimizationPhase.P14.value
    ]
    _write_json(manifest_path, missing_phase)
    with pytest.raises(ValueError, match=r"P1 through P14|at least 14"):
        assemble_local_p15_qualification_package(
            scope_path=scope_path,
            manifest_path=manifest_path,
        )

    scope_drift = copy.deepcopy(manifest_payload)
    drift_entries = scope_drift["phase_artifacts"]
    assert isinstance(drift_entries, list)
    assert isinstance(drift_entries[0], dict)
    drift_entries[0]["scope_digest"] = _digest(999)
    _write_json(manifest_path, scope_drift)
    with pytest.raises(ValueError, match=r"manifest scope digest|scope digest"):
        assemble_local_p15_qualification_package(
            scope_path=scope_path,
            manifest_path=manifest_path,
        )


def test_local_emission_rejects_nonlocal_artifact_evidence(tmp_path: Path) -> None:
    _, scope_path, manifest_path, _, manifest_payload = _write_inputs(tmp_path)

    nonlocal_evidence = copy.deepcopy(manifest_payload)
    entries = nonlocal_evidence["phase_artifacts"]
    assert isinstance(entries, list)
    assert isinstance(entries[0], dict)
    entries[0]["evidence_class"] = EvidenceClass.LOCAL_BENCHMARK.value
    _write_json(manifest_path, nonlocal_evidence)

    with pytest.raises(ValueError, match="LOCAL_CONFORMANCE"):
        assemble_local_p15_qualification_package(
            scope_path=scope_path,
            manifest_path=manifest_path,
        )


def test_cli_atomically_emits_and_round_trips_local_package(tmp_path: Path) -> None:
    _, scope_path, manifest_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "output" / "p15-package.json"
    output_path.parent.mkdir()
    output_path.write_text("stale output", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "emit_p15_qualification.py"),
            str(scope_path),
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    reloaded = P15QualificationPackage.model_validate_json(output_path.read_bytes(), strict=True)
    assert json.loads(result.stdout) == reloaded.as_dict()
    assert output_path.read_bytes().endswith(b"\n")
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
    verify_local_p15_qualification_package_artifacts(reloaded)
