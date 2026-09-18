"""Search, page-reading, and sandboxed Python tools for agentic evaluation.

The search interface and XML-style tool protocol are adapted from OpenSearch-VL
at commit c5c02a49780e26ae9cb6f1fb56731d1e594d59f0, especially
``opensearch_infer/search.py`` and the RL search/visit/Python tool modules.
The implementation here adds deterministic disk caching, structured provenance,
strict output limits, URL validation, and a network-disabled Docker executor.

Copyright 2026 OpenSearch-VL Authors. Licensed under Apache-2.0. See
``licenses/OpenSearch-VL-Apache-2.0.txt`` and ``THIRD_PARTY_NOTICES.md``.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


AGENTIC_TOOL_MODE = "agentic-open-book-v1"
OFFLINE_PYTHON_TOOL_MODE = "offline-python-v1"
LOCAL_VISION_PYTHON_TOOL_MODE = "local-vision-python-v1"
MAX_SEARCH_RESULTS = 10
MAX_TEXT_CHARS = 30_000
MAX_STDIO_CHARS = 12_000

RESEARCH_TOOL_DEFINITIONS = [
    {"name": "web_search", "description": "Search the public web for current or factual information.",
     "arguments": {"query": "search terms (q is also accepted)", "num_results": 5}},
    {"name": "text_search", "description": "Alias of web_search for textual retrieval.",
     "arguments": {"query": "search terms", "num_results": 5}},
    {"name": "image_search", "description": "Search the public web for images by a text query.",
     "arguments": {"query": "visual subject", "num_results": 5}},
    {"name": "visit", "description": "Read the text of a public http(s) page returned by search.",
     "arguments": {"url": "https://example.org/page", "goal": "information to find (optional)"}},
]

PYTHON_TOOL_DEFINITION = {"name": "python_interpreter", "description":
     "Run Python for calculations, data analysis, or plotting in a network-disabled sandbox. Print useful results and save a plot as a PNG when visual output is needed.",
     "arguments": {"code": "print(2 + 2)"}}

# Kept as a public aggregate for callers that want the complete agentic tool set.
SEARCH_TOOL_DEFINITIONS = [*RESEARCH_TOOL_DEFINITIONS, PYTHON_TOOL_DEFINITION]


def _trim(value: str, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit] + f"\n...[truncated after {limit} characters]", True


def _cache_key(name: str, arguments: dict[str, Any]) -> str:
    encoded = json.dumps({"name": name, "arguments": arguments}, sort_keys=True,
                         ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_public_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("visit accepts only absolute http(s) URLs")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are blocked")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("local and private hosts are blocked")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local
                    or address.is_reserved or address.is_multicast or address.is_unspecified):
        raise ValueError("local and private addresses are blocked")
    return value.strip()


class ResearchToolSession:
    """Asynchronous Serper/Jina tools with an auditable on-disk response cache."""

    def __init__(self, cache_dir: Path, *, timeout: float = 30.0):
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.serper_key = os.environ.get("SERPER_API_KEY", "")
        self.jina_key = os.environ.get("JINA_API_KEY", "")
        self.jina_base = os.environ.get("JINA_READER_URL", "https://r.jina.ai").rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True)

    async def close(self) -> None:
        await self.client.aclose()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        path = self.cache_dir / f"{key}.json"
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)

    async def _serper(self, kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.serper_key:
            raise RuntimeError("SERPER_API_KEY is not set")
        query = str(arguments.get("query") or arguments.get("q") or "").strip()
        if not query or len(query) > 1000:
            raise ValueError("query must contain 1 to 1000 characters")
        number = max(1, min(MAX_SEARCH_RESULTS, int(
            arguments.get("num_results", arguments.get("top_k", 5)))))
        endpoint = "images" if kind == "image_search" else "search"
        response = await self.client.post(
            f"https://google.serper.dev/{endpoint}",
            headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
            json={"q": query, "num": number, "hl": str(arguments.get("hl") or arguments.get("lang") or "en")},
        )
        response.raise_for_status()
        raw = response.json()
        source = raw.get("images", []) if kind == "image_search" else raw.get("organic", [])
        fields = ("title", "source", "link", "imageUrl", "thumbnailUrl") if kind == "image_search" else (
            "title", "link", "snippet", "date", "position")
        results = [{field: item.get(field) for field in fields if item.get(field) is not None}
                   for item in source[:number]]
        return {"query": query, "provider": "serper", "result_count": len(results), "results": results}

    async def _visit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = _validate_public_url(str(arguments.get("url", "")))
        headers = {"Accept": "text/plain"}
        if self.jina_key:
            headers["Authorization"] = f"Bearer {self.jina_key}"
        reader_url = f"{self.jina_base}/{url}"
        response = await self.client.get(reader_url, headers=headers)
        response.raise_for_status()
        content, truncated = _trim(response.text)
        return {"url": url, "goal": str(arguments.get("goal", "")),
                "reader": self.jina_base, "content": content,
                "content_chars": len(response.text), "truncated": truncated}

    async def execute(self, call: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
        started = time.perf_counter()
        name = str(call.get("name", ""))
        arguments = call.get("arguments", {})
        record: dict[str, Any] = {"name": name, "arguments": arguments, "ok": False}
        try:
            if name not in {"web_search", "text_search", "image_search", "visit"}:
                raise ValueError(f"unknown research tool {name!r}")
            normalized_name = "web_search" if name == "text_search" else name
            backend = self.jina_base if normalized_name == "visit" else "serper"
            key = _cache_key(normalized_name, {**arguments, "_backend": backend})
            payload = self._read_cache(key)
            cache_hit = payload is not None
            if payload is None:
                payload = (await self._visit(arguments) if normalized_name == "visit"
                           else await self._serper(normalized_name, arguments))
                self._write_cache(key, payload)
            observation, truncated = _trim(json.dumps(payload, ensure_ascii=False, indent=2))
            record.update({"ok": True, "cache_hit": cache_hit, "cache_key": key,
                           "observation": observation, "observation_truncated": truncated})
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["observation"] = "Tool execution error: " + record["error"]
        record["elapsed_seconds"] = time.perf_counter() - started
        return record, None


class PythonToolExecutor:
    """Execute model-authored Python without network or access to the evaluation suite."""

    def __init__(self, mode: str, image: str, timeout: float, work_root: Path):
        if mode not in {"docker", "podman", "local"}:
            raise ValueError("tool executor must be docker, podman, or local")
        self.mode, self.image, self.timeout = mode, image, timeout
        self.work_root = work_root.resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        if mode in {"docker", "podman"}:
            probe = subprocess.run([mode, "image", "inspect", image], capture_output=True, text=True)
            if probe.returncode != 0:
                raise RuntimeError(
                    f"Container image {image!r} is unavailable. Run: "
                    f"{mode} build -t {image} -f docker/Dockerfile .")

    def execute(self, call: dict[str, Any], artifact_dir: Path) -> tuple[dict[str, Any], Path | None]:
        started = time.perf_counter()
        code = str(call.get("arguments", {}).get("code", ""))
        record: dict[str, Any] = {"name": "python_interpreter", "arguments": {"code": code}, "ok": False}
        directory: Path | None = None
        container_name: str | None = None
        try:
            if not code.strip():
                raise ValueError("code must not be empty")
            if len(code) > 100_000:
                raise ValueError("code exceeds the 100000 character limit")
            directory = Path(tempfile.mkdtemp(prefix="python_tool_", dir=self.work_root))
            shutil.copyfile(Path(__file__).with_name("python_tool_worker.py"), directory / "worker.py")
            (directory / "job.json").write_text(
                json.dumps({"code": code, "timeout": self.timeout}, ensure_ascii=False), encoding="utf-8")
            if self.mode in {"docker", "podman"}:
                container_name = "mmeval-tool-" + uuid.uuid4().hex[:16]
                command = [
                    self.mode, "run", "--rm", "--name", container_name,
                    "--network", "none", "--read-only", "--memory", "768m", "--cpus", "1",
                    "--pids-limit", "64", "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
                    "--ulimit", "fsize=20971520:20971520", "--ulimit", "nofile=128:128",
                    "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
                ]
                if hasattr(os, "getuid"):
                    command += ["--user", f"{os.getuid()}:{os.getgid()}"]
                command += ["-v", f"{directory}:/work:rw", "-w", "/work", self.image,
                            "python", "worker.py", "job.json"]
            else:
                command = [sys.executable, str(directory / "worker.py"), str(directory / "job.json")]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout + 15)
            except subprocess.TimeoutExpired:
                if container_name:
                    subprocess.run([self.mode, "rm", "-f", container_name], capture_output=True, timeout=10)
                raise TimeoutError(f"Python tool exceeded {self.timeout} seconds")
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            result = json.loads(lines[-1]) if lines else {
                "ok": False, "error": "Python worker returned no structured output"}
            result["worker_returncode"] = completed.returncode
            if completed.stderr:
                result["worker_stderr"] = completed.stderr[-3000:]
            artifact_dir.mkdir(parents=True, exist_ok=True)
            output_path = None
            copied = []
            for index, filename in enumerate(result.pop("generated_images", []), 1):
                source = directory / filename
                if source.is_file() and source.stat().st_size <= 20_000_000:
                    target = artifact_dir / f"python_output_{int(time.time() * 1000)}_{index}.png"
                    shutil.copyfile(source, target)
                    copied.append(str(target))
                    output_path = output_path or target
            stdout, stdout_cut = _trim(str(result.get("stdout", "")), MAX_STDIO_CHARS)
            stderr, stderr_cut = _trim(str(result.get("stderr", "")), MAX_STDIO_CHARS)
            result.update({"stdout": stdout, "stderr": stderr,
                           "stdout_truncated": stdout_cut, "stderr_truncated": stderr_cut})
            result["generated_images"] = copied
            observation, observation_cut = _trim(json.dumps(result, ensure_ascii=False, indent=2))
            record.update(result)
            record.update({"observation": observation, "observation_truncated": observation_cut})
            return record, output_path
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["observation"] = "Tool execution error: " + record["error"]
            return record, None
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            if directory:
                shutil.rmtree(directory, ignore_errors=True)

    async def execute_async(self, call: dict[str, Any], artifact_dir: Path) -> tuple[dict[str, Any], Path | None]:
        return await asyncio.to_thread(self.execute, call, artifact_dir)
