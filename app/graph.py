from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.auditor import AuditorAgent
from app.agents.image_analyst import ImageAnalystAgent
from app.agents.lead_physician import LeadPhysicianAgent
from app.agents.retriever import RetrieverAgent
from app.config import get_settings
from app.integrations.ollama_client import OllamaClient
from app.knowledge.store import PgVectorKnowledgeStore
from app.state import MedicalReportState


def route_after_audit(state: MedicalReportState) -> str:
    audit = state.get("audit_result", {})
    retry_count = state.get("retry_count", 0)
    if audit.get("approved") or retry_count >= 2:
        return "finish"
    return "rewrite"


async def increment_retry(state: MedicalReportState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


@lru_cache
def get_ollama_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        verify_ssl=settings.ollama_verify_ssl,
        ca_cert=settings.ollama_ca_cert,
        keep_alive=settings.ollama_keep_alive,
    )


@lru_cache
def get_knowledge_store() -> PgVectorKnowledgeStore:
    return PgVectorKnowledgeStore(get_settings().database_url)


@lru_cache
def build_graph():
    settings = get_settings()
    ollama = get_ollama_client()
    knowledge_store = get_knowledge_store()
    image_agent = ImageAnalystAgent(settings, ollama)
    retriever = RetrieverAgent(settings, ollama, knowledge_store)
    physician = LeadPhysicianAgent(settings, ollama)
    auditor = AuditorAgent(settings, ollama)

    builder = StateGraph(MedicalReportState)
    builder.add_node("image_analyst", image_agent.run)
    builder.add_node("retriever", retriever.run)
    builder.add_node("lead_physician", physician.run)
    builder.add_node("auditor", auditor.run)
    builder.add_node("increment_retry", increment_retry)

    builder.add_edge(START, "image_analyst")
    builder.add_edge("image_analyst", "retriever")
    builder.add_edge("retriever", "lead_physician")
    builder.add_edge("lead_physician", "auditor")
    builder.add_conditional_edges(
        "auditor", route_after_audit,
        {"finish": END, "rewrite": "increment_retry"},
    )
    builder.add_edge("increment_retry", "lead_physician")
    return builder.compile()
