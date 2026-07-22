from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical import local_composition

SOURCE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical" / "source-recording.json"


def test_local_fixture_mock_covers_every_coordinate_in_each_call_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = local_composition._capabilities

    def multipart_capabilities(observed_at: str):  # type: ignore[no-untyped-def]
        return original(observed_at).model_copy(update={"max_images_per_request": 3})

    monkeypatch.setattr(local_composition, "_capabilities", multipart_capabilities)

    receipt = local_composition.run_local_canonical_fixture(
        source_path=SOURCE_FIXTURE,
        state_dir=tmp_path / "state",
        run_key="multipart-fixture",
    )

    assert receipt.status == "SUCCEEDED"
    assert receipt.fixture_inference_calls > 1
