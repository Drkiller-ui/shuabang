from __future__ import annotations

import argparse
import asyncio

from .infer import run_inference
from .report import make_report
from .scoring import score_run
from .common import DEFAULT_BENCHMARKS


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="data/mini_eval_v1")
    parser.add_argument("--run", required=True)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS),
                        help="comma-separated benchmark names; default: gpqa,mmmu,multimodalqa; all selects all three")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mmeval")
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer", help="Generate model responses with resume support")
    add_common(infer)
    infer.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    infer.add_argument("--api-key-env", default="OPENAI_API_KEY")
    infer.add_argument("--model", default="Qwen/Qwen3.5-4B")
    infer.add_argument("--concurrency", type=int, default=8)
    infer.add_argument("--temperature", type=float, default=0.0)
    infer.add_argument("--seed", type=int, default=20260916)
    infer.add_argument("--max-tokens", type=int, help="cumulative output cap; hard maximum is 32768")
    infer.add_argument("--timeout", type=float, default=900.0)
    infer.add_argument("--retries", type=int, default=5)
    infer.add_argument("--limit", type=int, help="debug: maximum items per benchmark")
    infer.add_argument("--force", action="store_true", help="regenerate even when the request fingerprint matches")
    infer.add_argument("--tools", choices=("none", "local-vision", "agentic"), default="none",
                       help="none, offline visual tools, or the fixed per-benchmark routed tool policy")
    infer.add_argument("--max-tool-turns", type=int, default=6)
    infer.add_argument("--tool-executor", choices=("docker", "podman", "local"), default="docker",
                       help="Python tool executor; use Docker or Podman for real evaluation")
    infer.add_argument("--tool-container-image", default="qwen35-eval-sandbox:latest")
    infer.add_argument("--tool-python-timeout", type=float, default=10.0)
    infer.add_argument("--extra-body-json", help="extra OpenAI request fields as a JSON object")

    score = sub.add_parser("score", help="Parse answers and execute code in a sandbox")
    add_common(score)
    score.add_argument("--workers", type=int, default=4)
    score.add_argument("--force", action="store_true")

    report = sub.add_parser("report", help="Aggregate scores and export bad cases")
    add_common(report)

    args = parser.parse_args()
    if args.command == "infer":
        asyncio.run(run_inference(args))
    elif args.command == "score":
        score_run(args)
    else:
        make_report(args)


if __name__ == "__main__":
    main()
