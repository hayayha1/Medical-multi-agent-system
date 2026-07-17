from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.config import get_settings
from app.graph import get_knowledge_store, get_ollama_client
from app.repository import repository


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.assert_production_ready()
    if settings.app_mode == "production":
        await get_ollama_client().version()
    try:
        yield
    finally:
        await get_ollama_client().close()
        await get_knowledge_store().close()
        await repository.close()


settings = get_settings()
app = FastAPI(
    title="医疗影像多智能体报告系统",
    version="0.1.0",
    description="仅生成医生待审核报告草稿，不替代医生诊断。",
    lifespan=lifespan,
)
app.include_router(router, prefix=settings.api_prefix)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": settings.app_mode}
