#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-gpt-5.4-mini}"
DATA_ROOT="${DATA_ROOT:-data/application/mcq/by_world}"
WORLDS_STR="${WORLDS:-seen_baseline never_seen_baseline}"
WORKERS="${WORKERS:-10}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
OVERWRITE="${OVERWRITE:-0}"
API_KEY_FILE="${API_KEY_FILE:-keys/openrouter_key.txt}"

read -r -a WORLDS <<< "$WORLDS_STR"

for world in "${WORLDS[@]}"; do
  input="$DATA_ROOT/$world.json"
  if [[ ! -f "$input" ]]; then
    echo "ERROR: missing application world file: $input" >&2
    exit 1
  fi

  args=(
    -m memory_control_tests.evaluation.application
    --input "$input"
    --model "$MODEL"
    --api_key_file "$API_KEY_FILE"
    --workers "$WORKERS"
    --request_timeout "$REQUEST_TIMEOUT"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  fi

  echo "RUN application plain api $world model=$MODEL"
  "$PYTHON_BIN" "${args[@]}"
done
