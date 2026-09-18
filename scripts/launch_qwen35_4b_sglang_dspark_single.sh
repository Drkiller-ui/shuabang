#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${VENV_DIR:-.venv-dspark}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-shanjiaz/qwen3_5_4b_perfectblend_regen_dspark}"
REVISION_LOCK="${REVISION_LOCK:-environment-dspark-model-revisions.json}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-40000}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
EXPECTED_GPU_NAME_SUBSTRING="${EXPECTED_GPU_NAME_SUBSTRING:-4090}"
ALLOW_LOW_VRAM="${ALLOW_LOW_VRAM:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing $VENV_DIR. Run scripts/setup_uv_sglang_dspark_single.sh first." >&2
  exit 1
fi

if [[ -f "$REVISION_LOCK" ]]; then
  readarray -t LOCKED_VALUES < <(
    "$VENV_DIR/bin/python" - "$REVISION_LOCK" "$MODEL" "$DRAFT_MODEL" <<'PY'
import json
import re
import sys

path, model, draft = sys.argv[1:]
lock = json.load(open(path, encoding="utf-8"))
if lock.get("model") != model or lock.get("draft_model") != draft:
    raise SystemExit(f"Revision lock {path} does not match the requested model IDs")
values = (lock.get("model_revision"), lock.get("draft_revision"))
if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) for value in values):
    raise SystemExit(f"Revision lock {path} does not contain two full commit SHAs")
print(*values, sep="\n")
PY
  )
  MODEL_REVISION="${MODEL_REVISION:-${LOCKED_VALUES[0]}}"
  DRAFT_MODEL_REVISION="${DRAFT_MODEL_REVISION:-${LOCKED_VALUES[1]}}"
else
  MODEL_REVISION="${MODEL_REVISION:-main}"
  DRAFT_MODEL_REVISION="${DRAFT_MODEL_REVISION:-main}"
fi

export TP_SIZE DP_SIZE MIN_VRAM_MIB EXPECTED_GPU_NAME_SUBSTRING ALLOW_LOW_VRAM
"$VENV_DIR/bin/python" - <<'PY'
import os
import torch

expected_count = int(os.environ["TP_SIZE"]) * int(os.environ["DP_SIZE"])
actual_count = torch.cuda.device_count()
if actual_count != expected_count:
    raise SystemExit(
        f"Expected {expected_count} visible GPU(s) for TP={os.environ['TP_SIZE']} "
        f"and DP={os.environ['DP_SIZE']}, found {actual_count}."
    )
minimum_mib = int(os.environ["MIN_VRAM_MIB"])
expected_name = os.environ["EXPECTED_GPU_NAME_SUBSTRING"].lower()
allow_low_vram = os.environ.get("ALLOW_LOW_VRAM", "0") == "1"
for index in range(actual_count):
    props = torch.cuda.get_device_properties(index)
    memory_mib = props.total_memory // 1024**2
    print(f"GPU {index}: {props.name}, {memory_mib} MiB")
    if expected_name not in props.name.lower():
        raise SystemExit(f"GPU {index} is not an RTX 4090: {props.name}")
    if memory_mib < minimum_mib and not allow_low_vram:
        raise SystemExit(
            f"GPU {index} has {memory_mib} MiB; target+draft requires "
            f"at least {minimum_mib} MiB per GPU."
        )
PY

"$VENV_DIR/bin/python" scripts/patch_sglang_qwen35_last_layer.py --check

export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-compact}"

EXTRA_ARGS=()
if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
  EXTRA_ARGS+=(--disable-cuda-graph)
fi

exec "$VENV_DIR/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --revision "$MODEL_REVISION" \
  --served-model-name "$MODEL" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --tp-size "$TP_SIZE" \
  --dp-size "$DP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --attention-backend triton \
  --enable-metrics \
  --reasoning-parser qwen3 \
  --limit-mm-data-per-request '{"image":16}' \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-draft-model-revision "$DRAFT_MODEL_REVISION" \
  --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE" \
  "${EXTRA_ARGS[@]}"
