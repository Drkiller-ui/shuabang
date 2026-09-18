#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv-dspark}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.19}"
EXPECTED_TORCH_VERSION="${EXPECTED_TORCH_VERSION:-2.13.0}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-13.0}"
MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-40}"
EXPECTED_COMPUTE_CAPABILITY="${EXPECTED_COMPUTE_CAPABILITY:-8.9}"
EXPECTED_GPU_NAME_SUBSTRING="${EXPECTED_GPU_NAME_SUBSTRING:-4090}"
export EXPECTED_TORCH_VERSION EXPECTED_TORCH_CUDA MIN_GPU_MEMORY_GIB EXPECTED_COMPUTE_CAPABILITY EXPECTED_GPU_NAME_SUBSTRING SGLANG_VERSION
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --upgrade uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
  "$VENV_DIR/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
fi
uv pip install --python "$VENV_DIR/bin/python" -r requirements-eval.txt
uv pip install --python "$VENV_DIR/bin/python" --upgrade \
  --prerelease=allow \
  -r requirements-server.txt

"$VENV_DIR/bin/python" - <<'PY'
import os
import importlib.metadata
import torch

expected_sglang = os.environ["SGLANG_VERSION"]
actual_sglang = importlib.metadata.version("sglang")
print(f"sglang={actual_sglang}")
if actual_sglang != expected_sglang:
    raise SystemExit(f"Expected SGLang {expected_sglang}, got {actual_sglang}")
print(f"torch={torch.__version__}")
print(f"torch CUDA runtime={torch.version.cuda}")
print(f"CUDA available={torch.cuda.is_available()}")
expected_torch = os.environ["EXPECTED_TORCH_VERSION"]
actual_torch = importlib.metadata.version("torch")
if actual_torch != expected_torch:
    raise SystemExit(f"Expected PyTorch {expected_torch}, got {actual_torch}")
expected = os.environ["EXPECTED_TORCH_CUDA"]
if torch.version.cuda != expected:
    raise SystemExit(f"Expected a cu{expected.replace('.', '')} PyTorch wheel, got {torch.version.cuda!r}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access an NVIDIA GPU")
if torch.cuda.device_count() < 1:
    raise SystemExit("Expected at least one NVIDIA GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"This profile requires exactly one visible GPU, found {torch.cuda.device_count()}. "
        "Set CUDA_VISIBLE_DEVICES to one device."
    )
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    gib = props.total_memory / 1024**3
    capability = f"{props.major}.{props.minor}"
    print(
        f"GPU {index}: {props.name}, {gib:.1f} GiB, "
        f"compute capability {capability}"
    )
    minimum = float(os.environ["MIN_GPU_MEMORY_GIB"])
    if gib < minimum:
        raise SystemExit(f"Expected at least {minimum:.0f} GiB VRAM, found {gib:.1f} GiB")
    expected_cc = os.environ["EXPECTED_COMPUTE_CAPABILITY"]
    if capability != expected_cc:
        raise SystemExit(
            f"Expected RTX 4090 compute capability {expected_cc}, found {capability} ({props.name})"
        )
    expected_name = os.environ["EXPECTED_GPU_NAME_SUBSTRING"].lower()
    if expected_name not in props.name.lower():
        raise SystemExit(
            f"Expected GPU name containing {expected_name!r}, found {props.name!r}"
        )
PY

# Temporary, narrow fix for Qwen3.5 checkpoints whose DSpark taps include the
# final target layer. The patcher is idempotent and refuses unknown layouts.
"$VENV_DIR/bin/python" scripts/patch_sglang_qwen35_last_layer.py
"$VENV_DIR/bin/python" scripts/patch_sglang_qwen35_last_layer.py --check

"$VENV_DIR/bin/python" -m compileall -q mmeval scripts/preflight_sglang_dspark.py
"$VENV_DIR/bin/python" -m unittest discover -s tests -v
uv pip freeze --python "$VENV_DIR/bin/python" > environment-dspark-cu130.lock.txt

"$VENV_DIR/bin/python" scripts/preflight_sglang_dspark.py \
  --recommended-vram-gib "$MIN_GPU_MEMORY_GIB" \
  --revision-lock environment-dspark-model-revisions.json \
  --report environment-dspark-single.json

echo "Single ${EXPECTED_GPU_NAME_SUBSTRING} GPU DSpark environment is ready."
echo "Run the complete startup and DSpark validation gate with:"
if [[ "$EXPECTED_GPU_NAME_SUBSTRING" == "4080" ]]; then
  echo "  SKIP_SETUP=1 bash scripts/validate_qwen35_4b_sglang_dspark_4080super.sh"
else
  echo "  SKIP_SETUP=1 bash scripts/validate_qwen35_4b_sglang_dspark_4090.sh"
fi
