from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    case_id: str
    split: str = "test"
    stratum: str = "unspecified"
    image_paths: list[str] = Field(default_factory=list)
    reference_findings: str = ""
    reference_impression: str = ""
    indication: str = ""
    comparison: str = ""
    problems: str = ""
    mesh: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def reference_report(self) -> str:
        return join_report(self.reference_findings, self.reference_impression)


class EvaluationRecord(BaseModel):
    case_id: str
    method_id: str
    split: str = "test"
    stratum: str = "unspecified"
    image_paths: list[str] = Field(default_factory=list)
    reference_findings: str = ""
    reference_impression: str = ""
    candidate_findings: str = ""
    candidate_impression: str = ""
    state: dict[str, Any] = Field(default_factory=dict)
    latency_seconds: float | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def reference_report(self) -> str:
        return join_report(self.reference_findings, self.reference_impression)

    @property
    def candidate_report(self) -> str:
        return join_report(self.candidate_findings, self.candidate_impression)


def join_report(findings: str, impression: str) -> str:
    parts: list[str] = []
    if findings.strip():
        parts.append(f"FINDINGS: {findings.strip()}")
    if impression.strip():
        parts.append(f"IMPRESSION: {impression.strip()}")
    return "\n".join(parts)


def statements_to_text(items: list[dict[str, Any]] | None) -> str:
    return " ".join(
        str(item.get("text", "")).strip()
        for item in (items or [])
        if str(item.get("text", "")).strip()
    )


def record_from_state(
    case: EvaluationCase,
    method_id: str,
    state: dict[str, Any],
    latency_seconds: float,
) -> EvaluationRecord:
    draft = state.get("report_draft") or {}
    return EvaluationRecord(
        case_id=case.case_id,
        method_id=method_id,
        split=case.split,
        stratum=case.stratum,
        image_paths=case.image_paths,
        reference_findings=case.reference_findings,
        reference_impression=case.reference_impression,
        candidate_findings=statements_to_text(draft.get("findings")),
        candidate_impression=statements_to_text(draft.get("impression")),
        state=state,
        latency_seconds=latency_seconds,
        success=True,
    )

