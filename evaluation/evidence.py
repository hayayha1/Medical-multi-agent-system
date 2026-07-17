from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from evaluation.models import EvaluationRecord


SECTIONS = ("findings", "impression", "recommendations")


def structural_evidence_metrics(record: EvaluationRecord) -> dict[str, float | int | str]:
    state = record.state
    draft = state.get("report_draft") or {}
    findings = state.get("image_findings") or []
    evidence = state.get("retrieved_evidence") or []
    valid_image_ids = {str(item.get("finding_id")) for item in findings if item.get("finding_id")}
    valid_kb_ids = {str(item.get("evidence_id")) for item in evidence if item.get("evidence_id")}
    valid_ids = valid_image_ids | valid_kb_ids
    statements = [item for section in SECTIONS for item in (draft.get(section) or [])]
    citations = [str(value) for item in statements for value in (item.get("evidence_ids") or [])]
    valid_citations = [value for value in citations if value in valid_ids]
    kb_citations = [value for value in citations if value in valid_kb_ids]
    cited_kb = set(kb_citations)
    scores = [
        float(item["score"])
        for item in evidence
        if item.get("score") is not None and str(item.get("evidence_id")) in cited_kb
    ]
    return {
        "case_id": record.case_id,
        "method_id": record.method_id,
        "statement_count": len(statements),
        "citation_count": len(citations),
        "statement_citation_coverage": (
            sum(bool(item.get("evidence_ids")) for item in statements) / len(statements)
            if statements else 1.0
        ),
        "citation_existence_precision": len(valid_citations) / len(citations) if citations else 0.0,
        "knowledge_citation_fraction": len(kb_citations) / len(citations) if citations else 0.0,
        "retrieved_evidence_utilization": len(cited_kb) / len(valid_kb_ids) if valid_kb_ids else 0.0,
        "mean_cited_retrieval_score": mean(scores) if scores else 0.0,
        "unsupported_citation_count": len(citations) - len(valid_citations),
    }


def evidence_annotation_rows(record: EvaluationRecord, blind_report_id: str) -> list[dict[str, Any]]:
    state = record.state
    draft = state.get("report_draft") or {}
    evidence_by_id = {
        str(item.get("finding_id")): {**item, "source_type": "image_finding"}
        for item in (state.get("image_findings") or [])
        if item.get("finding_id")
    }
    evidence_by_id.update({
        str(item.get("evidence_id")): {**item, "source_type": "knowledge_document"}
        for item in (state.get("retrieved_evidence") or [])
        if item.get("evidence_id")
    })
    rows: list[dict[str, Any]] = []
    for section in SECTIONS:
        for statement_index, statement in enumerate(draft.get(section) or []):
            statement_id = f"{section}-{statement_index + 1}"
            ids = statement.get("evidence_ids") or [""]
            for evidence_id in ids:
                item = evidence_by_id.get(str(evidence_id), {})
                rows.append({
                    "blind_report_id": blind_report_id,
                    "case_id": record.case_id,
                    "statement_id": statement_id,
                    "statement_text": statement.get("text", ""),
                    "evidence_id": evidence_id,
                    "evidence_type": item.get("source_type", "missing"),
                    "evidence_title": item.get("title", item.get("finding_type", "")),
                    "evidence_content": item.get("summary", item.get("location", "")),
                    "evidence_source": item.get("source", ""),
                    "support": "",
                    "source_appropriate": "",
                    "clinically_requires_evidence": "",
                    "notes": "",
                })
    return rows


def aggregate_evidence_annotations(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method_id") or row.get("blind_report_id"))].append(row)
    output: dict[str, dict[str, float | int]] = {}
    for method, items in grouped.items():
        support_values = []
        full_support = 0
        appropriate = []
        required = 0
        unsupported_required = 0
        for item in items:
            support = str(item.get("support", "")).strip().lower()
            if support in {"yes", "y", "1", "true", "supported"}:
                support_values.append(1.0)
                full_support += 1
            elif support in {"partial", "partly", "0.5"}:
                support_values.append(0.5)
            elif support in {"no", "n", "0", "false", "unsupported"}:
                support_values.append(0.0)
            source = str(item.get("source_appropriate", "")).strip().lower()
            if source:
                appropriate.append(float(source in {"yes", "y", "1", "true"}))
            needs = str(item.get("clinically_requires_evidence", "")).strip().lower()
            if needs in {"yes", "y", "1", "true"}:
                required += 1
                if support in {"no", "n", "0", "false", "unsupported", ""}:
                    unsupported_required += 1
        output[method] = {
            "n_citations": len(support_values),
            "citation_support_precision": mean(support_values) if support_values else 0.0,
            "full_support_rate": full_support / len(support_values) if support_values else 0.0,
            "source_appropriateness_rate": mean(appropriate) if appropriate else 0.0,
            "unsupported_required_claim_rate": unsupported_required / required if required else 0.0,
        }
    return output

