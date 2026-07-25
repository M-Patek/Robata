from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    run_local_canonical_mcap,
)
from scripts.run_canonical_mcap import DEFAULT_MAX_DURATION_SECONDS, _parser


def test_mcap_cli_defaults_to_first_three_minutes() -> None:
    args = _parser().parse_args(
        [
            "recording.mcap",
            "--mapping-config",
            "mapping.json",
            "--state-dir",
            "state",
        ]
    )

    assert args.max_duration_seconds == DEFAULT_MAX_DURATION_SECONDS == 180


@pytest.mark.parametrize("value", ("0", "-1", "1.5", "invalid"))
def test_mcap_cli_rejects_invalid_duration_cap(value: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "recording.mcap",
                "--mapping-config",
                "mapping.json",
                "--state-dir",
                "state",
                "--max-duration-seconds",
                value,
            ]
        )


@pytest.mark.parametrize("value", (True, 0, -1, 1.5))
def test_mcap_api_rejects_invalid_duration_cap(value: object) -> None:
    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_mcap(
            Path("recording.mcap"),
            Path("mapping.json"),
            Path("state"),
            allow_unapproved_profile=True,
            max_duration_ns=value,  # type: ignore[arg-type]
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.INVALID_REQUEST
    assert "max_duration_ns must be a positive integer or None" in str(caught.value)


@pytest.mark.parametrize("value", (True, "invalid", object()))
def test_mcap_api_rejects_invalid_media_processing_policy(value: object) -> None:
    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_mcap(
            Path("recording.mcap"),
            Path("mapping.json"),
            Path("state"),
            allow_unapproved_profile=True,
            media_processing_policy=value,  # type: ignore[arg-type]
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.INVALID_REQUEST
    assert "media_processing_policy must be McapMediaProcessingPolicy or None" in str(
        caught.value
    )


class _AuthorizedMapping:
    semantic_sha256 = "test-mapping-semantic-sha256"


def test_mcap_media_policy_changes_binding_and_reaches_source_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robata.application.canonical import local_composition as local_composition_module
    from robata.application.canonical import mcap_source as mcap_source_module
    from robata.application.canonical.mcap_source import (
        DEFAULT_MCAP_MEDIA_PROCESSING_POLICY,
        McapMediaProcessingPolicy,
    )

    captured_runs: list[dict[str, object]] = []
    source_loader_calls: list[dict[str, object]] = []

    def capture_run(**kwargs: object) -> object:
        captured_runs.append(kwargs)
        return object()

    def capture_source_loader(*args: object, **kwargs: object) -> object:
        source_loader_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        mcap_source_module,
        "authorize_mcap_mapping",
        lambda *args, **kwargs: _AuthorizedMapping(),
    )
    monkeypatch.setattr(
        mcap_source_module,
        "load_canonical_mcap_source",
        capture_source_loader,
    )
    monkeypatch.setattr(
        local_composition_module,
        "_hash_source_file",
        lambda *args, **kwargs: "test-source-sha256",
    )
    monkeypatch.setattr(local_composition_module, "_run_local_canonical", capture_run)

    default_result = run_local_canonical_mcap(
        Path("recording.mcap"),
        Path("mapping.json"),
        Path("state"),
        allow_unapproved_profile=True,
    )
    custom_policy = McapMediaProcessingPolicy(sentinel_rate_numerator=1)
    custom_result = run_local_canonical_mcap(
        Path("recording.mcap"),
        Path("mapping.json"),
        Path("state"),
        allow_unapproved_profile=True,
        media_processing_policy=custom_policy,
    )

    assert default_result is not custom_result
    assert len(captured_runs) == 2
    assert captured_runs[0]["source_binding_sha256"] != captured_runs[1]["source_binding_sha256"]
    for run in captured_runs:
        source_loader = run["source_loader"]
        assert callable(source_loader)
        source_loader(object(), object(), "stream-run")
    assert [call["media_processing_policy"] for call in source_loader_calls] == [
        DEFAULT_MCAP_MEDIA_PROCESSING_POLICY,
        custom_policy,
    ]
