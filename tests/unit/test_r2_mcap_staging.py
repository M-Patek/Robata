"""Focused proofs for pinned R2 MCAP source staging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.adapters.fake_object_store import FakeObjectStore
from robata.application.canonical.r2_mcap_staging import (
    R2McapSourceManifest,
    R2McapSourceStagingError,
    parse_r2_mcap_source_manifest_bytes,
    stage_r2_mcap_source,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.object_storage import ObjectPutRequest
from scripts import stage_r2_mcap_source as cli


def _manifest_for(store: FakeObjectStore, payload: bytes) -> R2McapSourceManifest:
    receipt = store.put(
        ObjectPutRequest(
            key="recordings/real-sample.mcap",
            payload=payload,
            sha256=exact_bytes_sha256(payload),
            byte_count=len(payload),
            media_type="application/x-mcap",
            object_version="v1",
        )
    )
    return R2McapSourceManifest(
        locator=receipt.locator,
        expected_sha256=receipt.sha256,
        expected_byte_count=receipt.byte_count,
        expected_media_type="application/x-mcap",
    )


def _environment() -> dict[str, str]:
    return {
        "R2_ENDPOINT_URL": "https://account-id.r2.cloudflarestorage.com",
        "R2_BUCKET": "robata-production",
        "R2_ACCESS_KEY_ID": "access-secret",
        "R2_SECRET_ACCESS_KEY": "r2-super-secret",
    }


def test_stage_fetches_and_integrity_verifies_pinned_r2_bytes(tmp_path: Path) -> None:
    store = FakeObjectStore()
    payload = b"real mcap bytes"
    manifest = _manifest_for(store, payload)
    destination = tmp_path / "inputs" / "real-sample.mcap"

    receipt = stage_r2_mcap_source(
        manifest=manifest,
        object_store=store,
        destination=destination,
    )

    assert destination.read_bytes() == payload
    assert receipt.destination == destination
    assert receipt.content_sha256 == exact_bytes_sha256(payload)
    assert receipt.byte_count == len(payload)
    assert not receipt.reused_existing_file

    replay = stage_r2_mcap_source(
        manifest=manifest,
        object_store=store,
        destination=destination,
    )

    assert replay.reused_existing_file
    assert destination.read_bytes() == payload


def test_stage_rejects_source_metadata_mismatch_before_get(tmp_path: Path) -> None:
    store = FakeObjectStore()
    manifest = _manifest_for(store, b"real mcap bytes")
    mismatched = manifest.model_copy(update={"expected_media_type": "application/octet-stream"})

    with pytest.raises(R2McapSourceStagingError, match="metadata differs"):
        stage_r2_mcap_source(
            manifest=mismatched,
            object_store=store,
            destination=tmp_path / "real-sample.mcap",
        )

    assert store.operation_counts().get("get", 0) == 0


def test_stage_never_overwrites_different_existing_destination(tmp_path: Path) -> None:
    store = FakeObjectStore()
    manifest = _manifest_for(store, b"real mcap bytes")
    destination = tmp_path / "real-sample.mcap"
    destination.write_bytes(b"unrelated local data")

    with pytest.raises(R2McapSourceStagingError, match="will not be overwritten"):
        stage_r2_mcap_source(
            manifest=manifest,
            object_store=store,
            destination=destination,
        )

    assert destination.read_bytes() == b"unrelated local data"


def test_manifest_requires_exact_canonical_json_and_rejects_duplicate_keys() -> None:
    store = FakeObjectStore()
    manifest = _manifest_for(store, b"real mcap bytes")
    raw = canonical_json_bytes(manifest)

    assert parse_r2_mcap_source_manifest_bytes(raw) == manifest

    pretty = json.dumps(manifest.model_dump(mode="json"), indent=2).encode("utf-8")
    with pytest.raises(R2McapSourceStagingError, match="exact canonical"):
        parse_r2_mcap_source_manifest_bytes(pretty)

    duplicate = (
        b'{"expected_sha256":"'
        + manifest.expected_sha256.encode("ascii")
        + b'","expected_sha256":"'
        + manifest.expected_sha256.encode("ascii")
        + b'"}'
    )
    with pytest.raises(R2McapSourceStagingError, match="duplicate JSON object key"):
        parse_r2_mcap_source_manifest_bytes(duplicate)


def test_cli_config_only_performs_no_r2_factory_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeObjectStore()
    manifest = _manifest_for(store, b"real mcap bytes")
    manifest_path = tmp_path / "source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    def unexpected_factory(_: object) -> object:
        raise AssertionError("CONFIG_ONLY must not construct an R2 client")

    monkeypatch.setattr(cli, "create_r2_object_store_from_environment", unexpected_factory)

    status = cli.main(
        [
            "--source-manifest",
            str(manifest_path),
            "--destination",
            str(tmp_path / "input.mcap"),
        ],
        environment=_environment(),
    )

    captured = capsys.readouterr()
    assert status == 0
    assert json.loads(captured.out) == {
        "destination": str(tmp_path / "input.mcap"),
        "external_calls": False,
        "mode": "CONFIG_ONLY",
        "ok": True,
        "production_eligible": False,
        "r2": {
            "bucket": "robata-production",
            "endpoint_url": "https://account-id.r2.cloudflarestorage.com",
            "prefix": "",
        },
        "source": manifest.model_dump(mode="json"),
    }
    assert captured.err == ""


def test_cli_stage_is_explicit_and_uses_injected_object_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeObjectStore()
    payload = b"real mcap bytes"
    manifest = _manifest_for(store, payload)
    manifest_path = tmp_path / "source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    monkeypatch.setattr(cli, "create_r2_object_store_from_environment", lambda _: store)

    destination = tmp_path / "input.mcap"
    status = cli.main(
        [
            "--source-manifest",
            str(manifest_path),
            "--destination",
            str(destination),
            "--stage",
        ],
        environment=_environment(),
    )

    captured = capsys.readouterr()
    payload_json = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert destination.read_bytes() == payload
    assert payload_json["mode"] == "STAGED"
    assert payload_json["external_calls"] is True
    assert payload_json["receipt"] == {
        "byte_count": len(payload),
        "content_sha256": exact_bytes_sha256(payload),
        "reused_existing_file": False,
    }
