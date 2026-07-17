from app.config import Settings
from app.integrations.ollama_client import OllamaClient
from app.schemas import ImageAnalysisResult, ImageFinding
from app.state import MedicalReportState


class ImageAnalystAgent:
    """Calls validated pixel-level models. Demo mode returns an explicit synthetic finding."""

    def __init__(self, settings: Settings, ollama: OllamaClient):
        self.settings = settings
        self.ollama = ollama

    async def run(self, state: MedicalReportState) -> dict:
        if self.settings.app_mode == "demo":
            finding = ImageFinding(
                finding_type="demo_pulmonary_nodule",
                location="右肺上叶后段（演示数据，非真实分析）",
                size_mm=[8.2, 6.7],
                density="部分实性",
                margin="分叶",
                confidence=0.91,
                series_uid=f"DEMO-{state['study_uid']}",
                instance_uid="DEMO_ONLY",
            )
            return {"image_findings": [finding.model_dump()], "workflow_status": "analyzing"}

        paths = state.get("image_paths", [])
        if not paths:
            raise ValueError("No IU X-Ray images were resolved for this study")
        result = await self.ollama.chat_json(
            model=self.settings.image_analyst_model,
            system_prompt=(
                "你是放射科影像分析助手。仅描述图像中可见的胸部X线征象，不得编造病史，"
                "不得把不确定发现写成确诊。必须输出病灶位置、形态、尺寸（只有能够可靠测量时）"
                "和置信度。正常影像也必须返回一条finding_type=no_acute_abnormality的发现。"
            ),
            user_prompt=(
                f"检查标识：{state['study_uid']}；模态：{state.get('modality', 'DX')}；"
                "请分别查看提供的正位/侧位图像并给出结构化影像发现。"
            ),
            response_model=ImageAnalysisResult,
            image_paths=paths,
        )
        return {
            "image_findings": [item.model_dump() for item in result.findings],
            "workflow_status": "analyzing",
        }
