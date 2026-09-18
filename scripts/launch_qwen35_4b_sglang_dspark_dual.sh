#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Production profile: two independent target+draft replicas. This raises
# throughput without splitting the small 4B target across both GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TP_SIZE="${TP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-2}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"

exec bash scripts/launch_qwen35_4b_sglang_dspark_single.sh
