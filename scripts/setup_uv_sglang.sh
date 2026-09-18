#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
SGLANG_WHEEL_INDEX="${SGLANG_WHEEL_INDEX:-https://docs.sglang.ai/whl/cu130/}"
export EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-2}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --upgrade uv
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "$PYTHON_VERSION" .venv
uv pip install --python .venv/bin/python -r requirements-eval.txt
uv pip install --python .venv/bin/python --upgrade \
  --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url "$SGLANG_WHEEL_INDEX" \
  "sglang[all]"

.venv/bin/python - <<'PY'
import os
import torch

print(f"torch={torch.__version__}")
print(f"torch CUDA runtime={torch.version.cuda}")
print(f"CUDA available={torch.cuda.is_available()}")
if torch.version.cuda != "13.0":
    raise SystemExit(f"Expected a cu130 PyTorch wheel, got CUDA runtime {torch.version.cuda!r}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access an NVIDIA GPU")
print(f"GPU count={torch.cuda.device_count()}")
expected = int(os.environ["EXPECTED_GPU_COUNT"])
if torch.cuda.device_count() < expected:
    raise SystemExit(f"Expected at least {expected} GPUs, found {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    gib = props.total_memory / 1024**3
    print(f"GPU {index}: {props.name}, {gib:.1f} GiB, compute capability {props.major}.{props.minor}")
PY

.venv/bin/python -m compileall -q mmeval
.venv/bin/python -m unittest discover -s tests -v
uv pip freeze --python .venv/bin/python > environment-cu130.lock.txt

echo "uv environment ready."
echo "Start Qwen3.5-4B + DFlash with: bash scripts/launch_qwen35_4b_sglang_dflash.sh"
