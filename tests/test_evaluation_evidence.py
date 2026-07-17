from evaluation.evidence import structural_evidence_metrics
from evaluation.models import EvaluationRecord


def test_structural_evidence_metrics_detect_invalid_citation():
    record = EvaluationRecord(
        case_id="CXR1",
        method_id="full",
        candidate_findings="Finding",
        state={
            "image_findings": [{"finding_id": "F-1"}],
            "retrieved_evidence": [{"evidence_id": "KB-1", "score": 0.8}],
            "report_draft": {
                "findings": [{"text": "Finding", "evidence_ids": ["F-1", "MISSING"]}],
                "impression": [{"text": "Impression", "evidence_ids": ["KB-1"]}],
                "recommendations": [],
            },
        },
    )
    metrics = structural_evidence_metrics(record)
    assert metrics["citation_count"] == 3
    assert metrics["unsupported_citation_count"] == 1
    assert metrics["citation_existence_precision"] == 2 / 3

