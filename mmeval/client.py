from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Generation:
    content: str
    reasoning: str
    raw: dict[str, Any]
    latency_seconds: float
    usage: dict[str, Any]
    finish_reason: str | None


def split_reasoning(message: dict[str, Any]) -> tuple[str, str]:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    blocks = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        embedded = "\n\n".join(block.strip() for block in blocks if block.strip())
        reasoning = "\n\n".join(value for value in (reasoning.strip(), embedded) if value)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    return content, reasoning


class OpenAICompatibleClient:
    def __init__(self, api_base: str, api_key: str, timeout: float = 900.0, retries: int = 5):
        self.api_base = api_base.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), headers=self.headers)
        self.retries = retries

    async def close(self) -> None:
        await self.client.aclose()

    async def model_ids(self) -> list[str]:
        response = await self.client.get(f"{self.api_base}/models")
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", [])]

    async def generate(self, *, model: str, content: list[dict[str, Any]], max_tokens: int,
                       temperature: float, seed: int, extra_body: dict[str, Any] | None = None) -> Generation:
        return await self.generate_messages(
            model=model, messages=[{"role": "user", "content": content}], max_tokens=max_tokens,
            temperature=temperature, seed=seed, extra_body=extra_body,
        )

    async def generate_messages(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int,
                                temperature: float, seed: int,
                                extra_body: dict[str, Any] | None = None) -> Generation:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        if extra_body:
            payload.update(extra_body)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            started = time.perf_counter()
            try:
                response = await self.client.post(f"{self.api_base}/chat/completions", json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
                raw = response.json()
                choice = raw["choices"][0]
                content_text, reasoning = split_reasoning(choice["message"])
                return Generation(
                    content=content_text,
                    reasoning=reasoning,
                    raw=raw,
                    latency_seconds=time.perf_counter() - started,
                    usage=raw.get("usage") or {},
                    finish_reason=choice.get("finish_reason"),
                )
            except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    await asyncio.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"Generation failed after {self.retries} attempts: {last_error}")


def parse_extra_body(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body-json must be a JSON object")
    return parsed
