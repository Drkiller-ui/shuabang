#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Keep large installation and model caches on the project data volume.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/hf-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv-cache}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

VENV_DIR="${VENV_DIR:-.venv-dspark}"
PORT="${PORT:-8000}"
RUN_DIR="${RUN_DIR:-runs/dspark-4090-install-validation}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"
RECOMMENDED_VRAM_GIB="${RECOMMENDED_VRAM_GIB:-40}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
GPU_LABEL="${GPU_LABEL:-RTX 4090 48GB}"
mkdir -p "$RUN_DIR"
# A failed retry must never leave a previous run's PASS marker.
rm -f "$RUN_DIR/PASS.txt"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  bash scripts/setup_uv_sglang_dspark_single.sh 2>&1 | tee "$RUN_DIR/setup.log"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing $VENV_DIR after setup." >&2
  exit 1
fi

nvidia-smi -i 0 -q > "$RUN_DIR/nvidia-smi.txt"
"$VENV_DIR/bin/python" scripts/preflight_sglang_dspark.py \
  --recommended-vram-gib "$RECOMMENDED_VRAM_GIB" \
  --revision-lock "$RUN_DIR/model-revisions.json" \
  --report "$RUN_DIR/preflight.json"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PORT="$PORT" \
CONTEXT_LENGTH="$CONTEXT_LENGTH" \
MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" \
MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC" \
DISABLE_CUDA_GRAPH=1 \
REVISION_LOCK="$RUN_DIR/model-revisions.json" \
  setsid bash scripts/launch_qwen35_4b_sglang_dspark_single.sh \
  > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_DIR/server.pid"

set +e
"$VENV_DIR/bin/python" scripts/smoke_sglang_dspark.py \
  --api-base "http://127.0.0.1:${PORT}/v1" \
  --server-pid "$SERVER_PID" \
  --wait-seconds "$STARTUP_TIMEOUT_SECONDS" \
  --min-accept-length 1.01 \
  --report "$RUN_DIR/validation.json"
VALIDATION_STATUS=$?
set -e

if (( VALIDATION_STATUS != 0 )); then
  echo "Validation failed. Last 200 server log lines:" >&2
  tail -n 200 "$RUN_DIR/server.log" >&2 || true
  exit "$VALIDATION_STATUS"
fi

export VALIDATION_RUN_DIR="$RUN_DIR" GPU_LABEL
"$VENV_DIR/bin/python" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["VALIDATION_RUN_DIR"])
report = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
revisions = json.loads((run_dir / "model-revisions.json").read_text(encoding="utf-8"))
dspark = report["dspark"]
summary = (
    f"PASS: single {os.environ['GPU_LABEL']}, Qwen3.5-4B + SGLang + DSpark\n"
    f"sglang={preflight['packages']['sglang']}\n"
    f"target_revision={revisions['model_revision']}\n"
    f"draft_revision={revisions['draft_revision']}\n"
    f"spec_verify_ct={dspark['spec_verify_ct']}\n"
    f"spec_accept_length={dspark['spec_accept_length']}\n"
    f"spec_accept_rate={dspark['spec_accept_rate']}\n"
)
(run_dir / "PASS.txt").write_text(summary, encoding="utf-8")
print(summary)
PY

echo "Validation artifacts: $RUN_DIR"
echo "The validation server will now stop."
