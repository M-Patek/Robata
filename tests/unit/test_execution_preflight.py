from __future__ import annotations

from pathlib import Path

from robata.runtime.preflight import run_preflight

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "config" / "genrobot-observed-v0.json"


def test_preflight_is_offline_and_reports_unapproved_mapping(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"not decoded by preflight")
    result = run_preflight(
        source,
        tmp_path / "output",
        mapping_config=MAPPING,
        allow_unapproved=False,
    )
    assert result["ok"] is False
    assert result["provider_requests"] == 0
    assert any(
        check["name"] == "mapping_authorization" and check["ok"] is False
        for check in result["checks"]
    )


def test_preflight_accepts_explicit_local_override_and_rejects_nested_registry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"not decoded by preflight")
    output = tmp_path / "output"
    result = run_preflight(
        source,
        output,
        mapping_config=MAPPING,
        registry_root=output / "registry",
        allow_unapproved=True,
    )
    assert result["ok"] is False
    assert any(
        check["name"] == "mapping_authorization" and check["ok"] is True
        for check in result["checks"]
    )
    assert any(
        check["name"] == "registry_root" and check["ok"] is False for check in result["checks"]
    )
