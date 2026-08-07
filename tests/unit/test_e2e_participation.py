from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from robata.runtime.e2e_participation import (
    E2EParticipationBoundary,
    E2EParticipationCoverage,
    E2EParticipationDeclaration,
    E2EParticipationIssueCode,
    E2EParticipationManifest,
    E2EParticipationState,
    build_e2e_participation_manifest,
    e2e_participation_manifest_digest,
    runtime_fragment_digest,
    serialize_e2e_participation_manifest,
    validate_e2e_participation_manifest_against_fragment,
    write_e2e_participation_manifest,
)
from robata.runtime.e2e_trace import (
    E2ETraceFragmentRole,
    build_e2e_trace_runtime_fragment,
)
from robata.runtime.observability import (
    ProcessResourceSample,
    RuntimeProfileRecorder,
    RuntimeResourceMeasurement,
    RuntimeResourceStatus,
    runtime_span,
)


def _available(value: int) -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(status=RuntimeResourceStatus.AVAILABLE, value=value)


def _fragment(*span_names: str):
    def resource_sample() -> ProcessResourceSample:
        return ProcessResourceSample(
            rss_bytes=_available(1024),
            read_bytes=_available(0),
            write_bytes=_available(0),
        )

    recorder = RuntimeProfileRecorder(resource_sampler=resource_sample)
    for span_name in span_names:
        with runtime_span(recorder, span_name):
            pass
    return build_e2e_trace_runtime_fragment(
        role=E2ETraceFragmentRole.LAUNCHER,
        runtime_profile=recorder.snapshot(),
    )


def _declarations_for(fragment, *, source_state=E2EParticipationState.PARTICIPATING):
    declarations = []
    measured_stages = {stage.stage for stage in fragment.stages if stage.observed_span_count}
    for boundary in E2EParticipationBoundary:
        if boundary.to_trace_stage() in measured_stages:
            state = (
                source_state
                if boundary is E2EParticipationBoundary.SOURCE
                else E2EParticipationState.PARTICIPATING
            )
            declarations.append(
                E2EParticipationDeclaration(
                    boundary=boundary,
                    state=state,
                    required=True,
                    reason=(
                        None if state is E2EParticipationState.PARTICIPATING else "source omitted"
                    ),
                )
            )
        else:
            declarations.append(
                E2EParticipationDeclaration(
                    boundary=boundary,
                    state=E2EParticipationState.BYPASSED,
                    required=False,
                    reason="boundary is not part of this bounded run",
                )
            )
    return tuple(declarations)


def test_complete_coverage_excludes_declared_optional_boundaries() -> None:
    fragment = _fragment("source.decode", "inference.request")
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=_declarations_for(fragment),
        trace_id="11111111-1111-4111-8111-111111111111",
        observed_at="2026-08-05T00:00:00Z",
    )

    assert manifest.coverage is E2EParticipationCoverage.COMPLETE
    assert manifest.issues == ()
    assert manifest.measured_boundaries == (
        E2EParticipationBoundary.SOURCE,
        E2EParticipationBoundary.INFERENCE,
    )
    assert E2EParticipationBoundary.SCHEDULING in manifest.excluded_boundaries


def test_participating_without_measurement_is_partial_not_zero() -> None:
    fragment = _fragment("source.decode")
    declarations = list(_declarations_for(fragment))
    declarations[3] = E2EParticipationDeclaration(
        boundary=E2EParticipationBoundary.INFERENCE,
        state=E2EParticipationState.PARTICIPATING,
        required=True,
    )
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=declarations,
    )

    assert manifest.coverage is E2EParticipationCoverage.PARTIAL
    assert any(
        issue.code is E2EParticipationIssueCode.PARTICIPATING_NOT_MEASURED
        and issue.boundary is E2EParticipationBoundary.INFERENCE
        for issue in manifest.issues
    )
    inference = manifest.boundaries[3]
    assert inference.observed_span_count == 0


def test_unclassified_span_is_visible_and_prevents_complete_coverage() -> None:
    fragment = _fragment("source.decode", "unknown.instrumentation")
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=_declarations_for(fragment),
    )

    assert manifest.coverage is E2EParticipationCoverage.PARTIAL
    assert manifest.unclassified_span_count == 1
    assert any(
        issue.code is E2EParticipationIssueCode.UNCLASSIFIED_SPANS for issue in manifest.issues
    )


def test_measured_bypassed_boundary_is_failed_not_silently_excluded() -> None:
    fragment = _fragment("source.decode")
    declarations = list(_declarations_for(fragment))
    declarations[1] = E2EParticipationDeclaration(
        boundary=E2EParticipationBoundary.SOURCE,
        state=E2EParticipationState.BYPASSED,
        required=False,
        reason="operator expected source to be bypassed",
    )
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=declarations,
    )

    assert manifest.coverage is E2EParticipationCoverage.FAILED
    assert any(
        issue.code is E2EParticipationIssueCode.MEASURED_WHILE_EXCLUDED for issue in manifest.issues
    )


def test_declared_failed_boundary_is_failed_even_when_unmeasured() -> None:
    fragment = _fragment("source.decode")
    declarations = list(_declarations_for(fragment))
    declarations[3] = E2EParticipationDeclaration(
        boundary=E2EParticipationBoundary.INFERENCE,
        state=E2EParticipationState.FAILED,
        required=True,
        reason="endpoint returned a terminal failure",
    )
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=declarations,
    )

    assert manifest.coverage is E2EParticipationCoverage.FAILED
    assert any(
        issue.code is E2EParticipationIssueCode.DECLARED_FAILURE for issue in manifest.issues
    )


def test_required_bypassed_boundary_is_partial_but_optional_bypass_is_complete() -> None:
    fragment = _fragment("source.decode")
    declarations = list(_declarations_for(fragment))
    declarations[3] = E2EParticipationDeclaration(
        boundary=E2EParticipationBoundary.INFERENCE,
        state=E2EParticipationState.BYPASSED,
        required=True,
        reason="this is a source-only qualification",
    )
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=declarations,
    )
    assert manifest.coverage is E2EParticipationCoverage.PARTIAL
    assert any(
        issue.code is E2EParticipationIssueCode.REQUIRED_BOUNDARY_UNAVAILABLE
        for issue in manifest.issues
    )


def test_declarations_require_fixed_order_and_reasons() -> None:
    fragment = _fragment("source.decode")
    declarations = list(_declarations_for(fragment))
    declarations[0], declarations[1] = declarations[1], declarations[0]
    with pytest.raises(ValueError, match="stable stage order"):
        build_e2e_participation_manifest(runtime_fragment=fragment, declarations=declarations)
    with pytest.raises(ValueError, match="require a reason"):
        E2EParticipationDeclaration(
            boundary=E2EParticipationBoundary.SOURCE,
            state=E2EParticipationState.BYPASSED,
        )


def test_mapping_shorthand_can_declare_optional_states() -> None:
    fragment = _fragment("source.decode")
    declarations = {
        boundary: (
            E2EParticipationState.PARTICIPATING
            if boundary is E2EParticipationBoundary.SOURCE
            else E2EParticipationState.NOT_CONFIGURED
        )
        for boundary in E2EParticipationBoundary
    }
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=declarations,
        required_boundaries=(E2EParticipationBoundary.SOURCE,),
    )

    assert manifest.coverage is E2EParticipationCoverage.COMPLETE
    assert manifest.excluded_boundaries == tuple(
        boundary
        for boundary in E2EParticipationBoundary
        if boundary is not E2EParticipationBoundary.SOURCE
    )


def test_sidecar_digest_is_stable_and_binds_fragment() -> None:
    fragment = _fragment("source.decode", "inference.request")
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=_declarations_for(fragment),
    )
    output_path = Path(".local") / f"robata-participation-{os.getpid()}.json"
    output_path.unlink(missing_ok=True)
    try:
        digest = write_e2e_participation_manifest(manifest, output_path)

        assert digest == e2e_participation_manifest_digest(manifest)
        assert output_path.read_bytes() == serialize_e2e_participation_manifest(manifest)
        assert digest == hashlib.sha256(output_path.read_bytes()).hexdigest()
        assert manifest.runtime_fragment_sha256 == runtime_fragment_digest(fragment)
        assert (
            manifest.trace_fragment_sha256
            == manifest.trace_digest
            == manifest.runtime_fragment_sha256
        )
        validate_e2e_participation_manifest_against_fragment(manifest, fragment)

        changed_fragment = _fragment("source.decode", "inference.request", "publication.seal")
        with pytest.raises(ValueError, match="digest does not match"):
            validate_e2e_participation_manifest_against_fragment(manifest, changed_fragment)
    finally:
        output_path.unlink(missing_ok=True)


def test_manifest_rejects_tampered_coverage_or_issues() -> None:
    fragment = _fragment("source.decode")
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=_declarations_for(fragment),
    )
    payload = manifest.model_dump()
    payload["coverage"] = E2EParticipationCoverage.PARTIAL
    with pytest.raises(ValueError, match="coverage does not match"):
        E2EParticipationManifest(**payload)


def test_sidecar_serializes_and_round_trips_derived_boundary_lists() -> None:
    fragment = _fragment("source.decode", "inference.request")
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=_declarations_for(fragment),
    )

    payload = json.loads(serialize_e2e_participation_manifest(manifest))

    assert payload["excluded_boundaries"] == [
        boundary.value for boundary in manifest.excluded_boundaries
    ]
    assert payload["measured_boundaries"] == [
        boundary.value for boundary in manifest.measured_boundaries
    ]
    # Python-value round-trip uses ``model_validate``; wire-value round-trip
    # uses ``model_validate_json`` because StrictModel intentionally rejects
    # implicit JSON coercion when given a Python mapping.
    assert E2EParticipationManifest.model_validate(manifest.model_dump()) == manifest
    restored = E2EParticipationManifest.model_validate_json(
        serialize_e2e_participation_manifest(manifest)
    )
    assert restored == manifest


@pytest.mark.parametrize("field", ("excluded_boundaries", "measured_boundaries"))
def test_sidecar_rejects_tampered_derived_boundary_lists(field: str) -> None:
    fragment = _fragment("source.decode", "inference.request")
    manifest = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=_declarations_for(fragment),
    )
    payload = json.loads(serialize_e2e_participation_manifest(manifest))
    original = payload[field]
    payload[field] = list(reversed(original))
    if payload[field] == original:
        payload[field] = []

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        E2EParticipationManifest.model_validate_json(json.dumps(payload))
