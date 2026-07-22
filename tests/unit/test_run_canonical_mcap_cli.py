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
