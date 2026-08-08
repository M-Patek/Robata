from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    CanonicalLocalRunReceipt,
)
from robata.application.canonical.local_outbox_delivery import (
    LocalOutboxDeliveryOutcome,
    LocalOutboxDeliverySummary,
)
from robata.application.canonical.local_review_routing import LocalReviewRoutingSummary
from robata.review.routing import ReviewRoutingDisposition
from robata.runtime.canonical_profile import (
    CanonicalProfileManifest,
    CanonicalProfilePolicyFacts,
    ProfileFileFact,
    ProfileGitFacts,
    ProfileRuntimeFacts,
)
from robata.runtime.observability import RuntimeProfileRecorder, runtime_span
from scripts import profile_canonical_mcap as cli


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _manifest() -> CanonicalProfileManifest:
    return CanonicalProfileManifest.create(
        source=ProfileFileFact(sha256=_digest("source"), byte_count=10),
        mapping_config=ProfileFileFact(sha256=_digest("mapping"), byte_count=11),
        uv_lock=ProfileFileFact(sha256=_digest("lock"), byte_count=12),
        schema_catalog=ProfileFileFact(sha256=_digest("catalog"), byte_count=13),
        git=ProfileGitFacts(head_commit="a" * 40, dirty=False),
        runtime=ProfileRuntimeFacts(
            python_version="3.13.5",
            python_implementation="CPython",
            platform="test-platform",
            machine="test-machine",
            logical_cpu_count=8,
        ),
        policies=CanonicalProfilePolicyFacts(
            composition_version="composition-v1",
            pipeline_version="pipeline-v1",
            execution_policy_semantic_sha256=_digest("execution"),
            runtime_policy_semantic_sha256=_digest("runtime"),
            input_planner_version="planner-v1",
            parser_version="parser-v1",
            inference_policy_versions=("coarse-v1", "dense-v1"),
        ),
        run_key="profile-test",
        max_duration_ns=180_000_000_000,
        allow_unapproved_profile=True,
    )


def _receipt(*, replayed: bool) -> CanonicalLocalRunReceipt:
    return CanonicalLocalRunReceipt(
        schema_version="1.0",
        model_version="canonical-local-run-receipt-v4",
        ok=True,
        run_id="run-1",
        recording_identity="recording-1",
        status="NO_EVENTS",
        command_sha256=_digest("command"),
        completion_semantic_sha256=_digest("completion"),
        event_ids=(),
        revision_ids=(),
        outbox_ids=(),
        outbox_count=0,
        outbox_delivery=LocalOutboxDeliverySummary(
            model_version="canonical-local-outbox-delivery-v1",
            outcome=LocalOutboxDeliveryOutcome.NOT_APPLICABLE,
            outbox_ids=(),
            relay_attempt_count=0,
            pending_count=0,
            leased_count=0,
            retry_wait_count=0,
            delivered_count=0,
            dead_letter_count=0,
            unknown_count=0,
            budget_exhausted=False,
            last_error=None,
        ),
        media_quality_binding=None,
        supplemental_qa_evidence=None,
        review_routing=LocalReviewRoutingSummary(
            disposition=ReviewRoutingDisposition.NOT_ROUTED,
        ),
        replayed=replayed,
        fixture_inference_calls=0,
        network_call_count=0,
        evidence_class="LOCAL_CONFORMANCE",
        production_eligible=False,
    )


def _arguments(tmp_path: Path, output: Path) -> list[str]:
    return [
        str(tmp_path / "source.mcap"),
        "--profile",
        "legacy_window_v1",
        "--mapping-config",
        str(tmp_path / "mapping.json"),
        "--allow-unapproved-profile",
        "--state-dir",
        str(tmp_path / "state"),
        "--run-key",
        "profile-test",
        "--output",
        str(output),
    ]


def test_parser_matches_canonical_defaults_and_requires_output() -> None:
    args = cli._parser().parse_args(
        [
            "source.mcap",
            "--profile",
            "legacy_window_v1",
            "--mapping-config",
            "mapping.json",
            "--state-dir",
            "state",
            "--output",
            "profile.json",
        ]
    )

    assert args.max_duration_seconds == cli.DEFAULT_MAX_DURATION_SECONDS == 180
    assert args.require_clean is False
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "source.mcap",
                "--profile",
                "legacy_window_v1",
                "--mapping-config",
                "mapping.json",
                "--state-dir",
                "state",
            ]
        )


def test_parser_exposes_clean_worktree_gate() -> None:
    args = cli._parser().parse_args(
        [
            "source.mcap",
            "--profile",
            "legacy_window_v1",
            "--mapping-config",
            "mapping.json",
            "--state-dir",
            "state",
            "--output",
            "profile.json",
            "--require-clean",
        ]
    )

    assert args.require_clean is True


@pytest.mark.parametrize("value", ("0", "-1", "1.5", "invalid"))
def test_parser_rejects_invalid_duration(value: str) -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "source.mcap",
                "--profile",
                "legacy_window_v1",
                "--mapping-config",
                "mapping.json",
                "--state-dir",
                "state",
                "--output",
                "profile.json",
                "--max-duration-seconds",
                value,
            ]
        )


def test_success_writes_atomic_fresh_report_from_fake_canonical_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest()
    (tmp_path / "source.mcap").write_bytes(b"fixture source")
    output = tmp_path / "reports" / "profile.json"
    output.parent.mkdir()
    output.write_bytes(b"old report")
    calls: list[str] = []

    def build_manifest(**_kwargs: object) -> CanonicalProfileManifest:
        calls.append("manifest")
        return manifest

    def run_fake(**kwargs: object) -> CanonicalLocalRunReceipt:
        calls.append("run")
        observer = kwargs["runtime_observer"]
        assert isinstance(observer, RuntimeProfileRecorder)
        with runtime_span(observer, "fake.canonical"):
            pass
        observer.increment_counter("source.span_duration_ns", 80)
        observer.increment_counter("source.recording_duration_ns", 100)
        observer.increment_counter("source.requested_duration_ns", 90)
        return _receipt(replayed=False)

    monkeypatch.setattr(cli, "build_canonical_profile_manifest", build_manifest)
    monkeypatch.setattr(cli, "run_local_canonical_mcap", run_fake)
    monkeypatch.setattr(cli, "discover_canonical_profile_durations", lambda *_a, **_k: (100, 90))

    exit_code = cli.main(_arguments(tmp_path, output))

    assert exit_code == 0
    assert calls == ["manifest", "run"]
    report = json.loads(output.read_bytes())
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == report
    assert report["manifest_sha256"] == manifest.manifest_sha256
    assert report["execution_mode"] == "FRESH"
    assert report["receipt"]["replayed"] is False
    assert report["error"] is None
    assert report["recording_duration_ns"] == "100"
    assert report["requested_duration_ns"] == "90"
    assert report["source_span_duration_ns"] == "80"
    assert report["measurement_status"] == "NOT_MEASURED"
    assert report["work_queue_after"]["status"] == "ABSENT"
    assert [span["name"] for span in report["observer"]["spans"]] == ["fake.canonical"]
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_canonical_error_still_publishes_structured_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest()
    (tmp_path / "source.mcap").write_bytes(b"fixture source")
    output = tmp_path / "profile.json"

    def run_failed(**kwargs: object) -> CanonicalLocalRunReceipt:
        observer = kwargs["runtime_observer"]
        assert isinstance(observer, RuntimeProfileRecorder)
        with (
            pytest.raises(CanonicalLocalCompositionError),
            runtime_span(observer, "fake.failed"),
        ):
            raise CanonicalLocalCompositionError(
                CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
                "invalid local source",
            )
        raise CanonicalLocalCompositionError(
            CanonicalLocalCompositionErrorCode.SOURCE_INVALID,
            "invalid local source",
        )

    monkeypatch.setattr(cli, "build_canonical_profile_manifest", lambda **_kwargs: manifest)
    monkeypatch.setattr(cli, "run_local_canonical_mcap", run_failed)
    monkeypatch.setattr(
        cli,
        "discover_canonical_profile_durations",
        lambda *_args, **_kwargs: (None, None),
    )

    exit_code = cli.main(_arguments(tmp_path, output))

    assert exit_code == 2
    report = json.loads(output.read_bytes())
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == report
    assert report["receipt"] is None
    assert report["execution_mode"] == "UNKNOWN"
    assert report["error"] == {
        "code": "SOURCE_INVALID",
        "detail": "invalid local source",
        "error_type": "CanonicalLocalCompositionError",
    }
    assert report["observer"]["spans"][0]["status"] == "ERROR"
    assert report["qualification_status"] == "NOT_PRODUCTION_QUALIFIED"


def test_comparison_output_compares_a_prior_v3_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest()
    (tmp_path / "source.mcap").write_bytes(b"fixture source")

    def run_fake(**kwargs: object) -> CanonicalLocalRunReceipt:
        observer = kwargs["runtime_observer"]
        assert isinstance(observer, RuntimeProfileRecorder)
        observer.increment_counter("source.recording_duration_ns", 100)
        observer.increment_counter("source.requested_duration_ns", 90)
        return _receipt(replayed=False)

    monkeypatch.setattr(cli, "build_canonical_profile_manifest", lambda **_kwargs: manifest)
    monkeypatch.setattr(cli, "run_local_canonical_mcap", run_fake)
    monkeypatch.setattr(cli, "discover_canonical_profile_durations", lambda *_a, **_k: (100, 90))
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    comparison = tmp_path / "comparison.json"

    assert cli.main(_arguments(tmp_path, baseline)) == 0
    capsys.readouterr()
    exit_code = cli.main(
        [
            *_arguments(tmp_path, candidate),
            "--compare-with",
            str(baseline),
            "--comparison-output",
            str(comparison),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == json.loads(candidate.read_bytes())
    comparison_document = json.loads(comparison.read_bytes())
    assert comparison_document["model_version"] == "canonical-profile-comparison-v1"
    assert comparison_document["capacity"]["comparable"] is True
    assert comparison_document["capacity"]["comparison_kind"] == "LIKE_FOR_LIKE"


def test_comparison_arguments_must_be_supplied_as_a_pair(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            *_arguments(tmp_path, tmp_path / "profile.json"),
            "--compare-with",
            str(tmp_path / "baseline.json"),
        ]
    )

    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "PROFILE_PRECONDITION_FAILED"
    assert "must be supplied together" in error["detail"]


def test_profiler_requires_an_explicit_legacy_profile() -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "source.mcap",
                "--mapping-config",
                "mapping.json",
                "--state-dir",
                "state",
                "--output",
                "profile.json",
            ]
        )
