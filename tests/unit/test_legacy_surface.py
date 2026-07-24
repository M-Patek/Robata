from __future__ import annotations

import importlib.util
from pathlib import Path

import robata.adapters as adapters
import robata.application as application
import robata.contracts as contracts
import robata.ports as ports
import robata.runtime as runtime
import robata.runtime.local_streaming_benchmark as virtual_streaming_benchmark


def test_removed_legacy_analysis_modules_are_not_importable() -> None:
    removed_modules = (
        "robata.adapters.fake_vision_model",
        "robata.application.mainline",
        "robata.contracts.mainline",
        "robata.ports.mainline",
        "robata.runtime.execution",
        "robata.runtime.preflight",
        "robata.runtime.verification",
    )

    assert all(importlib.util.find_spec(module_name) is None for module_name in removed_modules)


def test_removed_legacy_analysis_scripts_are_absent() -> None:
    repository_root = Path(__file__).parents[2]
    removed_scripts = (
        "preflight_local_mainline.py",
        "run_local_mainline.py",
        "verify_local_mainline.py",
    )

    assert all(not (repository_root / "scripts" / name).exists() for name in removed_scripts)


def test_public_packages_do_not_export_legacy_analysis_api() -> None:
    removed_exports = {
        "DeterministicFakeVisionModelAdapter",
        "FakeVisionModelAdapter",
        "LocalMainlineConfig",
        "LocalMainlinePipeline",
        "MainlineRunError",
        "PublishedMainlineRun",
        "verify_local_mainline_output",
    }

    for package in (adapters, application, runtime):
        assert removed_exports.isdisjoint(package.__all__)


def test_public_contract_and_port_packages_do_not_export_legacy_api() -> None:
    removed_contract_exports = {
        "MainlineBundle",
        "MainlineRunReport",
        "MainlineStage",
        "RunStatus",
        "StageReport",
        "StageStatus",
    }

    assert removed_contract_exports.isdisjoint(contracts.__all__)
    assert "VisionModelAdapter" not in ports.__all__


def test_live_sources_do_not_reference_removed_legacy_namespaces() -> None:
    repository_root = Path(__file__).parents[2]
    live_roots = (repository_root / "src" / "robata", repository_root / "scripts")
    forbidden = (
        "robata.adapters.fake_vision_model",
        "robata.application.mainline",
        "robata.contracts.mainline",
        "robata.ports.mainline",
        "robata.runtime.execution",
        "robata.runtime.preflight",
        "robata.runtime.verification",
    )

    for live_root in live_roots:
        for source_path in live_root.rglob("*.py"):
            source = source_path.read_text(encoding="utf-8")
            assert all(namespace not in source for namespace in forbidden), source_path


def test_virtual_streaming_estimator_does_not_export_wp6_authority_names() -> None:
    removed_authority_names = {
        "LOCAL_STREAMING_QUALIFICATION_REPORT_VERSION",
        "LocalStreamingQualificationReport",
        "evaluate_local_streaming_qualification",
        "local_streaming_qualification_report_projection",
    }

    assert removed_authority_names.isdisjoint(virtual_streaming_benchmark.__all__)
    assert all(not hasattr(virtual_streaming_benchmark, name) for name in removed_authority_names)
