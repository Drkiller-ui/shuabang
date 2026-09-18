#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv pip install -r requirements-eval.txt
# Qwen3.5 support currently requires the vLLM nightly wheels recommended by Qwen.
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

docker build -t qwen35-eval-sandbox:latest -f docker/Dockerfile .
python -m unittest discover -s tests -v

echo "Environment ready. Start the model with: scripts/launch_qwen35_4b_vllm.sh"

