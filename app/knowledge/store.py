import json
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


@dataclass
class KnowledgeHit:
    document_id: str
    title: str
    source: str
    version: str
    content: str
    score: float


class PgVectorKnowledgeStore:
    def __init__(self, database_url: str):
        self.engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)

    async def close(self) -> None:
        await self.engine.dispose()

    async def search(self, embedding: list[float], limit: int = 5) -> list[KnowledgeHit]:
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        statement = text("""
            SELECT id::text, title, source, version, content,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM knowledge_documents
            WHERE embedding IS NOT NULL
              AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)
        async with self.engine.connect() as connection:
            rows = (await connection.execute(
                statement, {"embedding": vector_literal, "limit": limit}
            )).mappings().all()
        return [
            KnowledgeHit(
                document_id=row["id"],
                title=row["title"],
                source=row["source"],
                version=row["version"],
                content=row["content"],
                score=max(0.0, min(1.0, float(row["score"]))),
            )
            for row in rows
        ]

    async def upsert(
        self,
        document_id: str,
        title: str,
        source: str,
        version: str,
        approved_by: str,
        content: str,
        embedding: list[float],
    ) -> None:
        vector_literal = json.dumps(embedding, separators=(",", ":"))
        statement = text("""
            INSERT INTO knowledge_documents
                (id, title, source, version, approved_by, content, embedding)
            VALUES
                (CAST(:id AS uuid), :title, :source, :version, :approved_by,
                 :content, CAST(:embedding AS vector))
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                source = EXCLUDED.source,
                version = EXCLUDED.version,
                approved_by = EXCLUDED.approved_by,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding
        """)
        async with self.engine.begin() as connection:
            await connection.execute(statement, {
                "id": document_id,
                "title": title,
                "source": source,
                "version": version,
                "approved_by": approved_by,
                "content": content,
                "embedding": vector_literal,
            })

