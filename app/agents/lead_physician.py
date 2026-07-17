import json

from app.config import Settings
from app.integrations.ollama_client import OllamaClient
from app.schemas import ReportDraft, ReportStatement
from app.state import MedicalReportState


class LeadPhysicianAgent:
    def __init__(self, settings: Settings, ollama: OllamaClient):
        self.settings = settings
        self.ollama = ollama

    async def run(self, state: MedicalReportState) -> dict:
        findings = state.get("image_findings", [])
        evidence = state.get("retrieved_evidence", [])
        if state.get("demo_mode"):
            if not findings:
                raise ValueError("No image findings available")
            finding = findings[0]
            draft = ReportDraft(
                clinical_history="演示模式",
                technique="胸部X线检查。",
                findings=[ReportStatement(
                    text="演示性影像发现。", evidence_ids=[finding["finding_id"]]
                )],
                impression=[ReportStatement(
                    text="需要影像科医生确认。", evidence_ids=[finding["finding_id"]]
                )],
                uncertainties=["演示模式未执行真实医学影像推理"],
            )
            return {"report_draft": draft.model_dump()}

        payload = {
            "检查": {
                "study_uid": state["study_uid"],
                "modality": state.get("modality"),
                "body_part": state.get("body_part"),
            },
            "临床信息": state.get("clinical_context", {}),
            "影像发现": findings,
            "检索证据": evidence,
        }
        draft = await self.ollama.chat_json(
            model=self.settings.lead_physician_model,
            system_prompt=(
                "你是主治放射科医生助手。只能使用输入中的影像发现、临床信息和检索证据生成"
                "报告草稿。每条findings、impression和recommendations必须填写真实存在的"
                "evidence_ids。没有异常时应明确写正常表现；证据不足时写入uncertainties，"
                "不得自行确诊或虚构指南。"
            ),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            response_model=ReportDraft,
        )
        return {"report_draft": draft.model_dump()}
