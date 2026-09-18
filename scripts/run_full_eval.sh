#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
VENV_DIR="${VENV_DIR:-.venv-dspark}"
source "$VENV_DIR/bin/activate"

RUN_DIR="${1:-runs/qwen35-4b-mini-v1}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
BENCHMARKS="${BENCHMARKS:-all}"
CONCURRENCY="${CONCURRENCY:-4}"
WORKERS="${WORKERS:-4}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TOOLS="${TOOLS:-agentic}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-8}"
TOOL_EXECUTOR="${TOOL_EXECUTOR:-docker}"
SCORE_EXECUTOR="${SCORE_EXECUTOR:-$TOOL_EXECUTOR}"

if [[ "$BENCHMARKS" == "all" ]]; then
  REQUIRED_BENCHMARKS=(mathvision mmmu mmlu_pro livecodebench multimodalqa gpqa)
else
  IFS=',' read -r -a REQUIRED_BENCHMARKS <<< "$BENCHMARKS"
fi

MISSING_DATA=()
for benchmark in "${REQUIRED_BENCHMARKS[@]}"; do
  benchmark="${benchmark//[[:space:]]/}"
  [[ -z "$benchmark" ]] && continue
  for kind in questions references; do
    path="data/mini_eval_v1/$kind/$benchmark.jsonl"
    [[ -f "$path" ]] || MISSING_DATA+=("$path")
  done
done
if (( ${#MISSING_DATA[@]} > 0 )); then
  echo "Cannot start evaluation: required dataset files are missing:" >&2
  printf '  - %s\n' "${MISSING_DATA[@]}" >&2
  echo "The public repository excludes GPQA and aggregate files. Rebuild authorized private data before running the affected benchmarks." >&2
  exit 2
fi

python -m mmeval infer \
  --run "$RUN_DIR" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --benchmarks "$BENCHMARKS" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --tools "$TOOLS" \
  --max-tool-turns "$MAX_TOOL_TURNS" \
  --tool-executor "$TOOL_EXECUTOR" \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":true}}'

SCORE_ARGS=(
  --run "$RUN_DIR"
  --benchmarks "$BENCHMARKS"
  --executor "$SCORE_EXECUTOR"
  --workers "$WORKERS"
)
python -m mmeval score "${SCORE_ARGS[@]}"
python -m mmeval report --run "$RUN_DIR" --benchmarks "$BENCHMARKS"

echo "Complete report: $RUN_DIR/summary.md"
echo "Bad cases:       $RUN_DIR/badcases/"
echo "Thinking flags:  $RUN_DIR/thinking_cases/"
echo "Full traces:     $RUN_DIR/responses/"
