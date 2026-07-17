from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_mode: Literal["demo", "production"] = "demo"
    app_name: str = "medical-multi-agent"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    app_secret: str = "CHANGE_ME_TO_A_LONG_RANDOM_SECRET"

    database_url: str = "postgresql+asyncpg://medical:CHANGE_ME@localhost:5432/medical_agents"
    redis_url: str = "redis://localhost:6379/0"

    ollama_base_url: str = "http://172.16.98.104:11434"
    ollama_verify_ssl: bool = True
    ollama_ca_cert: str | None = None
    ollama_timeout_seconds: float = 600
    ollama_keep_alive: str = "0"
    image_analyst_model: str = "medgemma1.5:4b"
    retriever_embedding_model: str = "bge-m3:latest"
    retriever_reranker_model: str | None = None
    lead_physician_model: str = "medgemma:27b"
    auditor_model: str = "qwen3.6:27b-bf16"

    iu_xray_dataset_path: str = "/home/ubuntu/hdd/mwz"
    knowledge_document_path: str = "/home/ubuntu/hdd/mwz/knowledge_base/approved_documents"
    knowledge_top_k: int = 5
    triton_url: str | None = None
    xray_model_name: str | None = None
    chest_ct_model_name: str | None = None

    dicomweb_base_url: str = "https://YOUR_PACS/dicom-web"
    dicomweb_username: str = "CHANGE_ME"
    dicomweb_password: str = "CHANGE_ME"
    fhir_base_url: str = "https://YOUR_FHIR_SERVER/fhir"
    fhir_token: str = "CHANGE_ME"

    hospital_name: str = "YOUR_HOSPITAL_NAME"
    department_name: str = "YOUR_DEPARTMENT_NAME"
    knowledge_approver: str = "YOUR_MEDICAL_KNOWLEDGE_APPROVER"

    def assert_production_ready(self) -> None:
        if self.app_mode != "production":
            return
        values = self.model_dump()
        placeholders = [
            key for key, value in values.items()
            if isinstance(value, str) and ("CHANGE_ME" in value or "YOUR_" in value)
        ]
        optional_prefixes = ("object_storage_", "dicomweb_", "fhir_", "langfuse_")
        placeholders = [key for key in placeholders if not key.startswith(optional_prefixes)]
        if placeholders:
            raise RuntimeError(f"Production placeholders not configured: {', '.join(placeholders)}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
