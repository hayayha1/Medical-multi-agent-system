import pytest

from app.agents.auditor import AuditorAgent


@pytest.mark.asyncio
async def test_auditor_rejects_missing_evidence():
    state = {
        "image_findings": [],
        "retrieved_evidence": [],
        "report_draft": {
            "clinical_history": "无",
            "technique": "CT",
            "findings": [{"text": "右肺结节", "evidence_ids": ["missing"]}],
            "impression": [{"text": "右肺结节", "evidence_ids": ["missing"]}],
        },
    }
    result = await AuditorAgent().run(state)
    assert result["audit_result"]["approved"] is False
    assert result["audit_result"]["risk_level"] == "high"


@pytest.mark.asyncio
async def test_auditor_accepts_supported_report():
    state = {
        "image_findings": [{"finding_id": "F1"}],
        "retrieved_evidence": [],
        "report_draft": {
            "clinical_history": "无",
            "technique": "CT",
            "findings": [{"text": "右肺结节", "evidence_ids": ["F1"]}],
            "impression": [{"text": "右肺结节", "evidence_ids": ["F1"]}],
        },
    }
    result = await AuditorAgent().run(state)
    assert result["audit_result"]["approved"] is True

