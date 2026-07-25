"""Complete local product-QA projection for the adaptive quality cascade.

The provider-facing QA contracts deliberately carry only coarse camera
observations.  This module is the local reduction boundary that makes the
product requirement explicit: every eligible recording has one deterministic
state for every :class:`~robata.contracts.qa.ProductQAIssue`, even where the
available evidence says ``NO_ISSUE``, ``ABSTAINED``, or ``INCOMPLETE_INPUT``.

It is intentionally a local, non-published result.  It does not alter a
registered wire shape, a logical key, or the 21-class product vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import StrictModel
from robata.contracts.qa import (
    LocalQARecordingResult,
    ProductQAIssue,
    ProductQAIssueEvidence,
    QAClassifier,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]

LOCAL_PRODUCT_QA_CASCADE_POLICY_VERSION = "local-product-qa-cascade-v1"


class ProductQAClassState(StrEnum):
    """Explicit disposition for one required product QA class."""

    NO_ISSUE = "NO_ISSUE"
    OBSERVED = "OBSERVED"
    ABSTAINED = "ABSTAINED"
    INCOMPLETE_INPUT = "INCOMPLETE_INPUT"


class ProductQACascadeStatus(StrEnum):
    """Recording-level status of the complete 21-class projection."""

    COMPLETE = "COMPLETE"
    ABSTAINED = "ABSTAINED"
    INCOMPLETE_INPUT = "INCOMPLETE_INPUT"


class ProductQAClassCoverage(StrictModel):
    """All retained evidence and one deterministic state for a product class."""

    issue: ProductQAIssue
    state: ProductQAClassState
    evidence: tuple[ProductQAIssueEvidence, ...] = ()
    reason_codes: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if any(item.issue is not self.issue for item in self.evidence):
            raise ValueError("class coverage evidence must match its product issue")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("class coverage reason codes must be unique and sorted")
        if self.state is ProductQAClassState.OBSERVED:
            if not self.evidence or self.reason_codes:
                raise ValueError("OBSERVED coverage requires evidence and no unresolved reason")
        elif self.state is ProductQAClassState.NO_ISSUE:
            if self.evidence or self.reason_codes:
                raise ValueError("NO_ISSUE coverage cannot carry evidence or unresolved reason")
        elif self.evidence:
            raise ValueError("unresolved product coverage cannot claim issue evidence")
        if self.state in {
            ProductQAClassState.ABSTAINED,
            ProductQAClassState.INCOMPLETE_INPUT,
        } and not self.reason_codes:
            raise ValueError("unresolved product coverage requires a reason code")
        return self


class ProductQACascadeResult(StrictModel):
    """One complete local product result reduced from retained evidence."""

    recording_id: NonEmptyString
    recording_duration_ns: PositiveInt
    policy_version: NonEmptyString
    status: ProductQACascadeStatus
    class_coverage: tuple[ProductQAClassCoverage, ...]
    product_result: LocalQARecordingResult
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        expected_issues = tuple(ProductQAIssue)
        if tuple(item.issue for item in self.class_coverage) != expected_issues:
            raise ValueError("product QA cascade must cover all 21 classes in vocabulary order")

        observed = tuple(
            evidence
            for coverage in self.class_coverage
            for evidence in coverage.evidence
        )
        if (
            self.product_result.assessment.recording_id != self.recording_id
            or self.product_result.assessment.duration_sec
            != self.recording_duration_ns / 1_000_000_000
            or self.product_result.issue_evidence != observed
        ):
            raise ValueError("product result does not match complete cascade evidence")

        states = {item.state for item in self.class_coverage}
        expected_status = (
            ProductQACascadeStatus.INCOMPLETE_INPUT
            if ProductQAClassState.INCOMPLETE_INPUT in states
            else (
                ProductQACascadeStatus.ABSTAINED
                if ProductQAClassState.ABSTAINED in states
                else ProductQACascadeStatus.COMPLETE
            )
        )
        if self.status is not expected_status:
            raise ValueError("cascade status does not match class coverage states")
        return self


class ProductQACascadeProjector:
    """Reduce retained evidence without inventing unobserved product issues."""

    def __init__(
        self,
        policy_version: str = LOCAL_PRODUCT_QA_CASCADE_POLICY_VERSION,
    ) -> None:
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("policy_version must be a nonempty string")
        self._policy_version = policy_version
        self._classifier = QAClassifier()

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def project(
        self,
        *,
        recording_id: str,
        recording_duration_ns: int,
        observed_evidence: Iterable[ProductQAIssueEvidence] = (),
        incomplete_issues: Iterable[ProductQAIssue] = (),
        abstained_issues: Iterable[ProductQAIssue] = (),
        incomplete_reason_codes: Iterable[str] = (),
        abstained_reason_codes: Iterable[str] = (),
    ) -> ProductQACascadeResult:
        """Project each product class with a fail-closed unresolved fallback.

        A caller may mark only selected classes incomplete or abstained.  When
        no class is explicitly selected but a corresponding reason is supplied,
        that reason applies to every class without already-observed evidence.
        This is useful for a source-wide timestamp/decode gap while preserving
        a separately observed black/blur/freeze fact.
        """

        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError("recording_id must be a nonempty string")
        if isinstance(recording_duration_ns, bool) or recording_duration_ns <= 0:
            raise ValueError("recording_duration_ns must be a positive integer")

        evidence = _canonical_evidence(observed_evidence)
        incomplete = _canonical_issues(incomplete_issues, "incomplete_issues")
        abstained = _canonical_issues(abstained_issues, "abstained_issues")
        if incomplete & abstained:
            raise ValueError("a product class cannot be both incomplete and abstained")
        incomplete_reasons = _canonical_reason_codes(
            incomplete_reason_codes,
            "incomplete_reason_codes",
        )
        abstained_reasons = _canonical_reason_codes(
            abstained_reason_codes,
            "abstained_reason_codes",
        )

        observed_by_issue: dict[ProductQAIssue, tuple[ProductQAIssueEvidence, ...]] = {
            issue: tuple(item for item in evidence if item.issue is issue)
            for issue in ProductQAIssue
        }
        unresolved_all_incomplete = not incomplete and bool(incomplete_reasons)
        unresolved_all_abstained = not abstained and bool(abstained_reasons)

        coverage: list[ProductQAClassCoverage] = []
        for issue in ProductQAIssue:
            issue_evidence = observed_by_issue[issue]
            if issue_evidence:
                coverage.append(
                    ProductQAClassCoverage(
                        issue=issue,
                        state=ProductQAClassState.OBSERVED,
                        evidence=issue_evidence,
                    )
                )
            elif issue in incomplete or unresolved_all_incomplete:
                coverage.append(
                    ProductQAClassCoverage(
                        issue=issue,
                        state=ProductQAClassState.INCOMPLETE_INPUT,
                        reason_codes=incomplete_reasons or ("INCOMPLETE_INPUT",),
                    )
                )
            elif issue in abstained or unresolved_all_abstained:
                coverage.append(
                    ProductQAClassCoverage(
                        issue=issue,
                        state=ProductQAClassState.ABSTAINED,
                        reason_codes=abstained_reasons or ("SEMANTIC_ABSTAINED",),
                    )
                )
            else:
                coverage.append(
                    ProductQAClassCoverage(
                        issue=issue,
                        state=ProductQAClassState.NO_ISSUE,
                    )
                )

        class_coverage = tuple(coverage)
        product_result = self._classifier.assess_evidence(
            recording_id=recording_id,
            duration_ns=recording_duration_ns,
            issues=tuple(item for entry in class_coverage for item in entry.evidence),
        )
        states = {item.state for item in class_coverage}
        status = (
            ProductQACascadeStatus.INCOMPLETE_INPUT
            if ProductQAClassState.INCOMPLETE_INPUT in states
            else (
                ProductQACascadeStatus.ABSTAINED
                if ProductQAClassState.ABSTAINED in states
                else ProductQACascadeStatus.COMPLETE
            )
        )
        return ProductQACascadeResult(
            recording_id=recording_id,
            recording_duration_ns=recording_duration_ns,
            policy_version=self._policy_version,
            status=status,
            class_coverage=class_coverage,
            product_result=product_result,
        )


def _canonical_evidence(
    values: Iterable[ProductQAIssueEvidence],
) -> tuple[ProductQAIssueEvidence, ...]:
    checked: list[ProductQAIssueEvidence] = []
    for value in values:
        if not isinstance(value, ProductQAIssueEvidence):
            raise TypeError("observed_evidence must contain ProductQAIssueEvidence")
        checked.append(
            ProductQAIssueEvidence.model_validate(value.model_dump(mode="python"), strict=True)
        )
    unique = {
        item.model_dump_json(): item
        for item in checked
    }
    return tuple(sorted(unique.values(), key=_evidence_sort_key))


def _canonical_issues(
    values: Iterable[ProductQAIssue],
    label: str,
) -> set[ProductQAIssue]:
    result: set[ProductQAIssue] = set()
    for value in values:
        if not isinstance(value, ProductQAIssue):
            raise TypeError(f"{label} must contain ProductQAIssue values")
        result.add(value)
    return result


def _canonical_reason_codes(values: Iterable[str], label: str) -> tuple[str, ...]:
    checked: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain nonempty strings")
        checked.append(value)
    return tuple(sorted(set(checked)))


def _evidence_sort_key(
    item: ProductQAIssueEvidence,
) -> tuple[object, ...]:
    interval = item.interval
    return (
        item.issue.value,
        item.scope.kind.value,
        "" if item.scope.camera_id is None else item.scope.camera_id.value,
        -1 if interval is None else interval.start_ns,
        -1 if interval is None else interval.end_ns,
        item.confidence,
        item.confidence_kind.value,
        item.scope.subject_refs,
        item.evidence_refs,
        "" if item.note is None else item.note,
    )


__all__ = [
    "LOCAL_PRODUCT_QA_CASCADE_POLICY_VERSION",
    "ProductQACascadeProjector",
    "ProductQACascadeResult",
    "ProductQACascadeStatus",
    "ProductQAClassCoverage",
    "ProductQAClassState",
]
