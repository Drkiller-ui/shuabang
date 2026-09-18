#!/usr/bin/env bash
set -euo pipefail

# Single 32 GB RTX 4080 SUPER environment smoke test. Production uses 2x4090.
export GPU_LABEL="${GPU_LABEL:-RTX 4080 SUPER 32GB}"
export EXPECTED_GPU_NAME_SUBSTRING="${EXPECTED_GPU_NAME_SUBSTRING:-4080}"
export MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-30}"
export MIN_VRAM_MIB="${MIN_VRAM_MIB:-30000}"
export RECOMMENDED_VRAM_GIB="${RECOMMENDED_VRAM_GIB:-30}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
export RUN_DIR="${RUN_DIR:-runs/dspark-4080super-install-validation}"

exec bash "$(dirname "$0")/validate_qwen35_4b_sglang_dspark_4090.sh"
