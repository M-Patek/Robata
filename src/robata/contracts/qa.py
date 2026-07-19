"""Requirements QA wire contracts.

Kept under ``contracts`` as a compatibility import path; implementation lives in ``robata.qa``
so the policy can also be used by application code without coupling to the larger mainline model.
"""

from robata.qa import (
    ISSUE_DISPOSITION,
    QA_ISSUE_GROUPS,
    ClipMark,
    ClipQAResult,
    ClipQAStatus,
    IssueDisposition,
    QAAssessment,
    QAClassifier,
    QAIssue,
    QAIssueType,
    QAResult,
    QAStatus,
    QualityAssessment,
    VideoQAStatus,
)

__all__ = [
    "ISSUE_DISPOSITION",
    "QA_ISSUE_GROUPS",
    "ClipMark",
    "ClipQAResult",
    "ClipQAStatus",
    "IssueDisposition",
    "QAAssessment",
    "QAClassifier",
    "QAIssue",
    "QAIssueType",
    "QAResult",
    "QAStatus",
    "QualityAssessment",
    "VideoQAStatus",
]
