import base64
import json
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 600,
        verify_ssl: bool = True,
        ca_cert: str | None = None,
        keep_alive: str = "0",
    ):
        verify: bool | str = ca_cert or verify_ssl
        self.keep_alive = keep_alive
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            verify=verify,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def version(self) -> str:
        response = await self.client.get("/api/version")
        response.raise_for_status()
        return response.json()["version"]

    @staticmethod
    def encode_images(paths: list[str]) -> list[str]:
        encoded: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                raise OllamaError(f"Image does not exist: {path}")
            encoded.append(base64.b64encode(path.read_bytes()).decode("ascii"))
        return encoded

    async def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        image_paths: list[str] | None = None,
        max_attempts: int = 2,
    ) -> ModelT:
        schema = response_model.model_json_schema()
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt + "\n严格按照以下JSON Schema输出：\n" + json.dumps(
                    schema, ensure_ascii=False
                ),
            },
        ]
        if image_paths:
            messages[1]["images"] = self.encode_images(image_paths)

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "format": schema,
                    "think": False,
                    "keep_alive": self.keep_alive,
                    "options": {"temperature": 0},
                },
            )
            if response.is_error:
                raise OllamaError(
                    f"Ollama chat failed ({response.status_code}): {response.text[:500]}"
                )
            content = response.json().get("message", {}).get("content", "")
            try:
                return response_model.model_validate_json(content)
            except Exception as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "上一个输出未通过JSON校验，请只输出符合Schema的JSON。",
                })
        raise OllamaError(f"Model returned invalid structured output: {last_error}")

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        response = await self.client.post(
            "/api/embed",
            json={"model": model, "input": texts, "truncate": True},
        )
        if response.is_error:
            raise OllamaError(
                f"Ollama embedding failed ({response.status_code}): {response.text[:500]}"
            )
        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise OllamaError("Ollama returned an unexpected embedding count")
        return embeddings

