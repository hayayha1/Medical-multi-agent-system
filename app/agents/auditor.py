import json

from app.config import Settings
from app.integrations.ollama_client import OllamaClient
from app.schemas import AuditIssue, AuditResult, ReportDraft
from app.state import MedicalReportState


class AuditorAgent:
    """Deterministic safety rules; an LLM may add review comments but cannot bypass these rules."""

    def __init__(self, settings: Settings | None = None, ollama: OllamaClient | None = None):
        self.settings = settings
        self.ollama = ollama

    async def run(self, state: MedicalReportState) -> dict:
        draft = ReportDraft.model_validate(state["report_draft"])
        valid_evidence = {
            item["finding_id"] for item in state.get("image_findings", [])
        } | {
            item["evidence_id"] for item in state.get("retrieved_evidence", [])
        }
        issues: list[AuditIssue] = []

        for section_name in ("findings", "impression", "recommendations"):
            for index, statement in enumerate(getattr(draft, section_name)):
                missing = set(statement.evidence_ids) - valid_evidence
                if missing:
                    issues.append(AuditIssue(
                        code="UNSUPPORTED_EVIDENCE",
                        severity="high",
                        message=f"引用了不存在的证据：{sorted(missing)}",
                        field=f"{section_name}[{index}]",
                        suggestion="删除陈述或绑定有效影像/指南证据。",
                    ))

        finding_text = " ".join(x.text for x in draft.findings)
        impression_text = " ".join(x.text for x in draft.impression)
        if ("左" in finding_text and "右" in impression_text) or (
            "右" in finding_text and "左" in impression_text
        ):
            issues.append(AuditIssue(
                code="LATERALITY_CONFLICT",
                severity="high",
                message="影像所见与诊断意见的左右侧可能不一致。",
                suggestion="根据原始影像及证据重新确认左右侧。",
            ))

        if not state.get("demo_mode") and self.ollama and self.settings:
            semantic = await self.ollama.chat_json(
                model=self.settings.auditor_model,
                system_prompt=(
                    "你是医疗报告质量审计员。检查所见、诊断意见和建议之间的语义矛盾、"
                    "过度诊断、遗漏不确定性和无证据陈述。不得修改报告，只输出审计问题。"
                    "approved仅表示语义审计是否通过；任何高风险问题必须approved=false。"
                ),
                user_prompt=json.dumps({
                    "影像发现": state.get("image_findings", []),
                    "检索证据": state.get("retrieved_evidence", []),
                    "报告草稿": draft.model_dump(),
                }, ensure_ascii=False),
                response_model=AuditResult,
            )
            issues.extend(semantic.issues)

        high_risk = any(issue.severity == "high" for issue in issues)
        result = AuditResult(
            approved=not high_risk,
            risk_level="high" if high_risk else ("medium" if issues else "low"),
            issues=issues,
        )
        return {
            "audit_result": result.model_dump(),
            "workflow_status": "awaiting_approval" if result.approved else "analyzing",
        }
