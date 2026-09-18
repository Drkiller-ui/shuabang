from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

from .agent_tools import PythonToolExecutor, ResearchToolSession
from .client import OpenAICompatibleClient, parse_extra_body
from .common import BENCHMARKS, append_jsonl, latest_records, load_benchmark, utc_now, write_json
from .diagnostics import DIAGNOSTICS_VERSION, analyze_thinking
from .prompts import build_user_content, prompt_text
from .tool_agent import run_tool_agent, safe_artifact_name


HARD_MAX_TOKENS = 32_768
DEFAULT_MAX_TOKENS = {
    benchmark: HARD_MAX_TOKENS for benchmark in BENCHMARKS
}

AGENTIC_BENCHMARK_TOOL_POLICY = {
    "mathvision": "local-vision-python",
    "mmmu": "local-vision",
    "mmlu_pro": "none",
    "livecodebench": "python",
    "multimodalqa": "agentic",
    "gpqa": "agentic",
}
NETWORK_SEARCH_BENCHMARKS = frozenset({"multimodalqa", "gpqa"})
PYTHON_TOOL_PROFILES = frozenset({"python", "local-vision-python", "agentic"})
TOOL_POLICY_VERSION = "benchmark-routed-v1"


def effective_tool_profile(benchmark: str, requested_mode: str) -> str:
    """Return the registered tool set for a benchmark under the requested run mode."""
    if requested_mode == "none":
        return "none"
    if requested_mode == "local-vision":
        return "local-vision" if benchmark in {"mathvision", "mmmu", "multimodalqa"} else "none"
    if requested_mode == "agentic":
        return AGENTIC_BENCHMARK_TOOL_POLICY[benchmark]
    raise ValueError(f"unsupported tools mode {requested_mode!r}")


def profile_track(profile: str) -> str:
    if profile == "agentic":
        return "open_book_agentic"
    if profile == "none":
        return "closed_book"
    return "closed_book_tool_augmented"


def parse_benchmarks(value: str) -> list[str]:
    values = list(BENCHMARKS) if value == "all" else [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(values) - set(BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
    return values


def request_hash(row: dict[str, Any], args: argparse.Namespace, extra_body: dict[str, Any]) -> str:
    requested_tools = getattr(args, "tools", "none")
    profile = effective_tool_profile(row["benchmark"], requested_tools)
    value = {
        "model": args.model,
        "benchmark": row["benchmark"],
        "prompt_text": prompt_text(row),
        "images": row["images"],
        "max_tokens": args.max_tokens or DEFAULT_MAX_TOKENS[row["benchmark"]],
        "temperature": args.temperature,
        "seed": args.seed,
        "extra_body": extra_body,
        "thinking_diagnostics": DIAGNOSTICS_VERSION,
        "requested_tools": requested_tools,
        "tool_profile": profile,
        "tool_policy": TOOL_POLICY_VERSION,
        "max_tool_turns": getattr(args, "max_tool_turns", 6),
        "tool_executor": getattr(args, "tool_executor", "docker"),
        "tool_container_image": getattr(args, "tool_container_image", "qwen35-eval-sandbox:latest"),
        "tool_python_timeout": getattr(args, "tool_python_timeout", 10.0),
        "tool_search_provider": "serper",
        "tool_reader_url": os.environ.get("JINA_READER_URL", "https://r.jina.ai"),
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


async def run_inference(args: argparse.Namespace) -> None:
    suite = Path(args.suite).resolve()
    run = Path(args.run).resolve()
    benchmarks = parse_benchmarks(args.benchmarks)
    api_key = os.environ.get(args.api_key_env, "")
    extra_body = parse_extra_body(args.extra_body_json)
    tools_mode = getattr(args, "tools", "none")
    max_tool_turns = getattr(args, "max_tool_turns", 6)
    tool_executor_mode = getattr(args, "tool_executor", "docker")
    tool_container_image = getattr(args, "tool_container_image", "qwen35-eval-sandbox:latest")
    tool_python_timeout = getattr(args, "tool_python_timeout", 10.0)
    benchmark_profiles = {benchmark: effective_tool_profile(benchmark, tools_mode)
                          for benchmark in benchmarks}
    search_enabled = any(profile == "agentic" for profile in benchmark_profiles.values())
    python_enabled = any(profile in PYTHON_TOOL_PROFILES for profile in benchmark_profiles.values())
    if args.max_tokens is not None and not 1 <= args.max_tokens <= HARD_MAX_TOKENS:
        raise ValueError(f"--max-tokens must be between 1 and {HARD_MAX_TOKENS}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if max_tool_turns < 0:
        raise ValueError("--max-tool-turns cannot be negative")
    if not 0.1 <= tool_python_timeout <= 60:
        raise ValueError("--tool-python-timeout must be between 0.1 and 60 seconds")
    if search_enabled and not os.environ.get("SERPER_API_KEY"):
        raise RuntimeError(
            "MultiModalQA/GPQA agentic search requires SERPER_API_KEY for web_search and image_search")
    client = OpenAICompatibleClient(args.api_base, api_key, args.timeout, args.retries)
    research_session: ResearchToolSession | None = None
    python_executor: PythonToolExecutor | None = None
    try:
        models = await client.model_ids()
        if args.model not in models:
            print(f"Warning: requested model {args.model!r}; endpoint advertises {models}")
        tracks = {profile_track(profile) for profile in benchmark_profiles.values()}
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "suite": str(suite),
            "model": args.model,
            "api_base": args.api_base,
            "benchmarks": benchmarks,
            "generation": {"temperature": args.temperature, "seed": args.seed,
                           "extra_body": extra_body, "max_tokens": args.max_tokens},
            "reasoning_capture": ["reasoning_content", "reasoning", "<think>...</think>"],
            "hard_max_tokens": HARD_MAX_TOKENS,
            "thinking_diagnostics": DIAGNOSTICS_VERSION,
            "evaluation_track": next(iter(tracks)) if len(tracks) == 1 else "mixed_benchmark_policy",
            "benchmark_tracks": {name: profile_track(profile)
                                 for name, profile in benchmark_profiles.items()},
            "tools": {
                "requested_mode": tools_mode, "policy_version": TOOL_POLICY_VERSION,
                "benchmark_profiles": benchmark_profiles, "max_turns": max_tool_turns,
                "available_on": [name for name, profile in benchmark_profiles.items() if profile != "none"],
                "network_search_available_on": [name for name, profile in benchmark_profiles.items()
                                                if profile == "agentic"],
                "search_provider": "serper" if search_enabled else None,
                "page_reader": os.environ.get("JINA_READER_URL", "https://r.jina.ai")
                               if search_enabled else None,
                "serper_configured": bool(os.environ.get("SERPER_API_KEY")),
                "python_executor": tool_executor_mode if python_enabled else None,
                "python_container_image": tool_container_image if python_enabled else None,
                "python_timeout_seconds": tool_python_timeout if python_enabled else None,
            },
        }
        write_json(run / "run_manifest.json", manifest)
        if search_enabled:
            research_session = ResearchToolSession(run / "tool_cache")
        if python_enabled:
            python_executor = PythonToolExecutor(
                tool_executor_mode, tool_container_image, tool_python_timeout, run / "work" / "python_tools")
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(args.concurrency)

        async def one(row: dict[str, Any], output: Path, fingerprint: str) -> None:
            async with semaphore:
                started = utc_now()
                prompt_log: dict[str, Any] = {}
                try:
                    content, prompt_log = build_user_content(row, suite)
                    effective_max = args.max_tokens or DEFAULT_MAX_TOKENS[row["benchmark"]]
                    tool_profile = effective_tool_profile(row["benchmark"], tools_mode)
                    use_tools = tool_profile != "none"
                    if use_tools:
                        generation = await run_tool_agent(
                            client=client, model=args.model, initial_content=content,
                            image_paths=[suite / relative for relative in row["images"]],
                            artifact_dir=run / "artifacts/tools" / safe_artifact_name(row["id"]),
                            max_tokens=effective_max, max_turns=max_tool_turns,
                            temperature=args.temperature, seed=args.seed, extra_body=extra_body,
                            tool_mode=tool_profile, research_session=research_session,
                            python_executor=python_executor,
                        )
                    else:
                        generation = await client.generate(
                            model=args.model, content=content, max_tokens=effective_max,
                            temperature=args.temperature, seed=args.seed, extra_body=extra_body,
                        )
                    raw_message = generation.raw.get("choices", [{}])[0].get("message", {})
                    if use_tools:
                        raw_message = {"content": "\n".join(
                            turn.get("assistant", "") for turn in generation.trajectory)}
                    diagnostics = analyze_thinking(
                        reasoning=generation.reasoning, answer=generation.content,
                        raw_content=raw_message.get("content") or "",
                        finish_reason=generation.finish_reason, usage=generation.usage,
                        max_tokens=effective_max,
                    )
                    record = {
                        "schema_version": 1, "id": row["id"], "benchmark": row["benchmark"],
                        "request_sha256": fingerprint,
                        "status": "ok", "started_at": started, "completed_at": utc_now(),
                        "prompt": prompt_log, "reasoning": generation.reasoning,
                        "answer": generation.content, "finish_reason": generation.finish_reason,
                        "usage": generation.usage, "latency_seconds": generation.latency_seconds,
                        "thinking_diagnostics": diagnostics,
                        "tool_profile": tool_profile,
                        "evaluation_track": profile_track(tool_profile),
                        "raw_response": generation.raw,
                    }
                    if use_tools:
                        record["tool_mode"] = generation.diagnostics["tool_mode"]
                        record["tool_trajectory"] = generation.trajectory
                        record["tool_diagnostics"] = generation.diagnostics
                except Exception as exc:
                    record = {
                        "schema_version": 1, "id": row["id"], "benchmark": row["benchmark"],
                        "request_sha256": fingerprint,
                        "status": "error", "started_at": started, "completed_at": utc_now(),
                        "prompt": prompt_log, "reasoning": "", "answer": "",
                        "tool_profile": effective_tool_profile(row["benchmark"], tools_mode),
                        "evaluation_track": profile_track(
                            effective_tool_profile(row["benchmark"], tools_mode)),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                    }
                async with lock:
                    append_jsonl(output, record)
                print(f"[{row['benchmark']}] {row['id']}: {record['status']}", flush=True)

        for benchmark in benchmarks:
            questions, _ = load_benchmark(suite, benchmark)
            if args.limit:
                questions = questions[:args.limit]
            output = run / "responses" / f"{benchmark}.jsonl"
            fingerprints = {row["id"]: request_hash(row, args, extra_body) for row in questions}
            prior = latest_records(output)
            completed = {
                key for key, value in prior.items()
                if value.get("status") == "ok" and value.get("request_sha256") == fingerprints.get(key)
                and not args.force
            }
            pending = [row for row in questions if row["id"] not in completed]
            print(f"{benchmark}: {len(completed)} completed, {len(pending)} pending", flush=True)
            await asyncio.gather(*(one(row, output, fingerprints[row["id"]]) for row in pending))
    finally:
        if research_session is not None:
            await research_session.close()
        await client.close()
