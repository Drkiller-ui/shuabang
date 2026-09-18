#!/usr/bin/env python3
"""End-to-end text, vision, and DSpark activation gate for SGLang."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def request_json(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def wait_ready(api_base: str, wait_seconds: int, server_pid: int | None) -> dict:
    deadline = time.monotonic() + wait_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if server_pid is not None:
            try:
                os.kill(server_pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError(f"SGLang process {server_pid} exited before readiness") from exc
        try:
            return request_json(f"{api_base.rstrip('/')}/models", timeout=10)
        except Exception as exc:  # server may still be downloading/loading weights
            last_error = exc
            time.sleep(5)
    raise TimeoutError(f"SGLang was not ready after {wait_seconds}s: {last_error}")


def red_png_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def message_text(message: dict) -> str:
    return "\n".join(
        str(message.get(key) or "")
        for key in ("reasoning_content", "reasoning", "content")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--wait-seconds", type=int, default=3600)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--min-accept-length", type=float, default=1.01)
    parser.add_argument("--report", default="dspark-smoke-report.json")
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    root_base = api_base[:-3] if api_base.endswith("/v1") else api_base
    report: dict = {"passed": False, "model": args.model, "api_base": api_base}
    try:
        report["models"] = wait_ready(api_base, args.wait_seconds, args.server_pid)

        text_response = request_json(
            f"{api_base}/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {"role": "user", "content": "Return exactly the marker DSPARK_OK."}
                ],
                "temperature": 0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        report["openai_text"] = text_response
        text_message = text_response["choices"][0]["message"]
        if "DSPARK_OK" not in message_text(text_message):
            raise RuntimeError("OpenAI text response did not contain DSPARK_OK")

        vision_response = request_json(
            f"{api_base}/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Name the dominant image color in one word."},
                            {"type": "image_url", "image_url": {"url": red_png_data_url()}},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        report["openai_vision"] = vision_response
        vision_message = vision_response["choices"][0]["message"]
        if not message_text(vision_message).strip():
            raise RuntimeError("Multimodal request returned an empty response")

        native_response = request_json(
            f"{root_base}/generate",
            {
                "text": (
                    "List the integers from 1 through 80 in order, one integer per line. "
                    "Do not skip any integer and do not add commentary."
                ),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 128,
                    "ignore_eos": True,
                },
            },
        )
        report["native_generate"] = native_response
        meta = native_response.get("meta_info") or {}
        verify_count = int(meta.get("spec_verify_ct") or 0)
        accept_length = float(meta.get("spec_accept_length") or 0)
        if verify_count <= 0:
            raise RuntimeError(f"DSpark was not active: spec_verify_ct={verify_count}")
        if accept_length < args.min_accept_length:
            raise RuntimeError(
                f"DSpark acceptance length {accept_length:.3f} is below "
                f"the smoke threshold {args.min_accept_length:.3f}"
            )
        report["dspark"] = {
            "spec_verify_ct": verify_count,
            "spec_accept_rate": meta.get("spec_accept_rate"),
            "spec_accept_length": accept_length,
            "spec_cap_length": meta.get("spec_cap_length"),
        }

        try:
            report["server_info"] = request_json(f"{root_base}/server_info", timeout=30)
        except Exception as exc:
            report["server_info_error"] = repr(exc)
        report["passed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Wrote validation report: {output.resolve()}")


if __name__ == "__main__":
    main()
