#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.5-4B-DFlash}"
PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"

export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --tp-size "$TP_SIZE" \
  --dp-size "$DP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --cuda-graph-max-bs-decode "$MAX_RUNNING_REQUESTS" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --mamba-scheduler-strategy extra_buffer \
  --reasoning-parser qwen3 \
  --limit-mm-data-per-request '{"image":16}' \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-dflash-block-size "$DFLASH_BLOCK_SIZE"
