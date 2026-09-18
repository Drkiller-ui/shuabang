from __future__ import annotations

import json
import re
import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_tools import (AGENTIC_TOOL_MODE, LOCAL_VISION_PYTHON_TOOL_MODE,
                          OFFLINE_PYTHON_TOOL_MODE, PYTHON_TOOL_DEFINITION,
                          RESEARCH_TOOL_DEFINITIONS, PythonToolExecutor,
                          ResearchToolSession)
from .client import Generation, OpenAICompatibleClient
from .prompts import image_data_url
from .vision_tools import (TOOL_DEFINITIONS, TOOL_MODE, VisionToolSession,
                           extract_tool_call, strip_response_tag, tools_prompt)


@dataclass
class ToolAgentResult:
    content: str
    reasoning: str
    raw: dict[str, Any]
    latency_seconds: float
    usage: dict[str, Any]
    finish_reason: str | None
    trajectory: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _token_count(usage: dict[str, Any], generation: Generation) -> int:
    for key in ("completion_tokens", "output_tokens"):
        if isinstance(usage.get(key), int):
            return usage[key]
    return max(1, (len(generation.reasoning) + len(generation.content) + 3) // 4)


def _sum_usage(target: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value


async def run_tool_agent(*, client: OpenAICompatibleClient, model: str,
                         initial_content: list[dict[str, Any]], image_paths: list[Path],
                         artifact_dir: Path, max_tokens: int, max_turns: int,
                         temperature: float, seed: int,
                         extra_body: dict[str, Any], tool_mode: str = "local-vision",
                         research_session: ResearchToolSession | None = None,
                         python_executor: PythonToolExecutor | None = None) -> ToolAgentResult:
    if tool_mode not in {"local-vision", "python", "local-vision-python", "agentic"}:
        raise ValueError(f"unsupported tool mode {tool_mode!r}")
    search_enabled = tool_mode == "agentic"
    python_enabled = tool_mode in {"python", "local-vision-python", "agentic"}
    vision_enabled = tool_mode in {"local-vision", "local-vision-python", "agentic"} and bool(image_paths)
    if search_enabled and research_session is None:
        raise ValueError("agentic mode requires a research tool session")
    if python_enabled and python_executor is None:
        raise ValueError(f"{tool_mode} mode requires a Python tool executor")
    session = VisionToolSession(image_paths, artifact_dir)
    extra_definitions = []
    if search_enabled:
        extra_definitions.extend(RESEARCH_TOOL_DEFINITIONS)
    if python_enabled:
        extra_definitions.append(PYTHON_TOOL_DEFINITION)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": tools_prompt(
            image_paths, extra_definitions, include_vision=vision_enabled)},
        {"role": "user", "content": initial_content},
    ]
    trajectory: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    raw_turns: list[dict[str, Any]] = []
    total_usage: dict[str, Any] = {}
    total_latency = 0.0
    remaining = max_tokens
    last_call: str | None = None
    consecutive_call_count = 0
    tool_loop = False
    exhausted = False
    max_turns_reached = False
    final_content = ""
    finish_reason = None

    for turn in range(max_turns + 1):
        generation = await client.generate_messages(
            model=model, messages=messages, max_tokens=max(1, remaining),
            temperature=temperature, seed=seed, extra_body=extra_body,
        )
        total_latency += generation.latency_seconds
        _sum_usage(total_usage, generation.usage)
        used = _token_count(generation.usage, generation)
        remaining = max(0, remaining - used)
        finish_reason = generation.finish_reason
        raw_turns.append(generation.raw)
        if generation.reasoning:
            reasoning_parts.append(f"[tool turn {turn}]\n{generation.reasoning}")
        call = extract_tool_call(generation.content)
        turn_record: dict[str, Any] = {
            "turn": turn, "reasoning": generation.reasoning, "assistant": generation.content,
            "finish_reason": generation.finish_reason, "usage": generation.usage,
            "latency_seconds": generation.latency_seconds, "remaining_output_tokens": remaining,
        }
        trajectory.append(turn_record)
        final_content = generation.content

        if remaining <= 0:
            exhausted = True
            break
        if not call:
            break
        if call.get("parse_error"):
            turn_record["tool_result"] = {"ok": False, "error": call["parse_error"]}
            break

        turn_record["tool_call"] = call
        canonical = json.dumps(call, ensure_ascii=False, sort_keys=True)
        if canonical == last_call:
            consecutive_call_count += 1
        else:
            last_call = canonical
            consecutive_call_count = 1
        if consecutive_call_count >= 3:
            tool_loop = True
            turn_record["tool_result"] = {
                "name": call["name"], "arguments": call["arguments"], "ok": False,
                "error": "same tool call repeated three times; agent loop stopped"}
            break
        if turn >= max_turns:
            max_turns_reached = True
            turn_record["tool_result"] = {
                "name": call["name"], "arguments": call["arguments"], "ok": False,
                "error": "tool turn limit reached before execution"}
            break

        tool_name = str(call.get("name", ""))
        if vision_enabled and tool_name in {item["name"] for item in TOOL_DEFINITIONS}:
            result, output_path = await asyncio.to_thread(session.execute, call)
        elif python_enabled and tool_name == "python_interpreter":
            assert python_executor is not None
            result, output_path = await python_executor.execute_async(call, artifact_dir)
            if output_path and result.get("ok"):
                image_id = session.register_image(output_path)
                result["output_image"] = image_id
                result["observation"] += f"\nThe first generated plot is attached as {image_id}."
        elif search_enabled and tool_name in {"web_search", "text_search", "image_search", "visit"}:
            assert research_session is not None
            result, output_path = await research_session.execute(call)
        else:
            result, output_path = ({
                "name": tool_name, "arguments": call.get("arguments", {}), "ok": False,
                "error": f"unknown or disabled tool {tool_name!r}",
                "observation": f"Tool execution error: unknown or disabled tool {tool_name!r}",
            }, None)
        turn_record["tool_result"] = result
        messages.append({"role": "assistant", "content": generation.content})
        observation = f"<observation>\n{result['observation']}\n</observation>"
        if output_path and result.get("ok"):
            observation_content: list[dict[str, Any]] = [
                {"type": "text", "text": observation},
                {"type": "image_url", "image_url": {"url": image_data_url(output_path)}},
            ]
            messages.append({"role": "user", "content": observation_content})
        else:
            messages.append({"role": "user", "content": observation})
    else:  # pragma: no cover - loop always exits through max_turn guard
        max_turns_reached = True

    tool_calls = [turn["tool_result"] for turn in trajectory if "tool_result" in turn]
    protocol_mode = (AGENTIC_TOOL_MODE if search_enabled else
                     LOCAL_VISION_PYTHON_TOOL_MODE if tool_mode == "local-vision-python" else
                     OFFLINE_PYTHON_TOOL_MODE if tool_mode == "python" else TOOL_MODE)
    diagnostics = {
        "tool_mode": protocol_mode,
        "tool_profile": tool_mode,
        "network_access": search_enabled,
        "evaluation_track": "open_book_agentic" if search_enabled else "closed_book_tool_augmented",
        "max_tool_turns": max_turns,
        "tool_calls": len(tool_calls),
        "successful_tool_calls": sum(bool(item.get("ok")) for item in tool_calls),
        "failed_tool_calls": sum(not bool(item.get("ok")) for item in tool_calls),
        "tool_loop_detected": tool_loop,
        "tool_turn_limit_reached": max_turns_reached,
        "output_token_budget_exhausted": exhausted,
        "remaining_output_tokens": remaining,
        "tool_counts": dict(Counter(item.get("name", "unknown") for item in tool_calls)),
    }
    return ToolAgentResult(
        content=strip_response_tag(final_content),
        reasoning="\n\n".join(reasoning_parts), raw={"turns": raw_turns},
        latency_seconds=total_latency, usage=total_usage, finish_reason=finish_reason,
        trajectory=trajectory, diagnostics=diagnostics,
    )


def safe_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)
