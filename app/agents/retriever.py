import json

from app.config import Settings
from app.integrations.ollama_client import OllamaClient
from app.knowledge.store import PgVectorKnowledgeStore
from app.schemas import Evidence
from app.state import MedicalReportState


class RetrieverAgent:
    def __init__(
        self,
        settings: Settings,
        ollama: OllamaClient,
        knowledge_store: PgVectorKnowledgeStore,
    ):
        self.settings = settings
        self.ollama = ollama
        self.knowledge_store = knowledge_store

    async def run(self, state: MedicalReportState) -> dict:
        if self.settings.app_mode == "demo":
            evidence = Evidence(
                title="演示知识条目",
                summary="该内容仅用于验证工作流；生产环境必须导入经医院审核的指南。",
                source="DEMO_ONLY",
                version="0.0",
                score=1.0,
            )
            return {"retrieved_evidence": [evidence.model_dump()]}
        findings = state.get("image_findings", [])
        context = state.get("clinical_context", {})
        query = json.dumps({
            "影像发现": findings,
            "主诉": context.get("chief_complaint"),
            "病史": context.get("history", []),
        }, ensure_ascii=False)
        vector = (await self.ollama.embed(
            self.settings.retriever_embedding_model, [query]
        ))[0]
        hits = await self.knowledge_store.search(vector, self.settings.knowledge_top_k)
        return {
            "retrieved_evidence": [
                Evidence(
                    evidence_id=f"KB-{hit.document_id}",
                    title=hit.title,
                    summary=hit.content[:1500],
                    source=hit.source,
                    version=hit.version,
                    score=hit.score,
                ).model_dump()
                for hit in hits
            ]
        }
