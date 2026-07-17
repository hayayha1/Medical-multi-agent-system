"""Ingest medically approved documents into PostgreSQL + pgvector."""

import asyncio
import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from docx import Document
from pypdf import PdfReader

from app.config import get_settings
from app.graph import get_knowledge_store, get_ollama_client


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    return ""


def chunks(text: str, size: int = 1800, overlap: int = 200):
    normalized = " ".join(text.split())
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        yield normalized[start:end]
        if end == len(normalized):
            break
        start = end - overlap


async def main() -> None:
    settings = get_settings()
    root = Path(settings.knowledge_document_path)
    if not root.is_dir():
        raise SystemExit(f"Knowledge directory does not exist: {root}")
    if "YOUR_" in settings.knowledge_approver:
        raise SystemExit("Set KNOWLEDGE_APPROVER before ingestion")

    ollama = get_ollama_client()
    store = get_knowledge_store()
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}
    ]
    count = 0
    try:
        for path in files:
            content = read_document(path)
            file_version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            for index, chunk in enumerate(chunks(content)):
                if len(chunk) < 80:
                    continue
                identifier = str(uuid5(NAMESPACE_URL, f"{path.resolve()}#{index}"))
                embedding = (await ollama.embed(
                    settings.retriever_embedding_model, [chunk]
                ))[0]
                await store.upsert(
                    document_id=identifier,
                    title=f"{path.stem} #{index + 1}",
                    source=str(path.relative_to(root)),
                    version=file_version,
                    approved_by=settings.knowledge_approver,
                    content=chunk,
                    embedding=embedding,
                )
                count += 1
                print(f"indexed {count}: {path.name} #{index + 1}")
    finally:
        await ollama.close()
        await store.close()
    print(f"done: {count} knowledge chunks")


if __name__ == "__main__":
    asyncio.run(main())

