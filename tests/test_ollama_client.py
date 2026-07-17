import json

import httpx
import pytest

from app.integrations.ollama_client import OllamaClient
from app.schemas import AuditResult


@pytest.mark.asyncio
async def test_chat_json_and_embed():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            assert body["stream"] is False
            assert isinstance(body["format"], dict)
            return httpx.Response(200, json={
                "message": {
                    "role": "assistant",
                    "content": '{"approved":true,"risk_level":"low","issues":[]}',
                }
            })
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        return httpx.Response(404)

    client = OllamaClient("http://ollama.test")
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    try:
        result = await client.chat_json(
            model="auditor",
            system_prompt="audit",
            user_prompt="report",
            response_model=AuditResult,
        )
        embeddings = await client.embed("embedding", ["query"])
    finally:
        await client.close()

    assert result.approved is True
    assert embeddings == [[0.1, 0.2]]

