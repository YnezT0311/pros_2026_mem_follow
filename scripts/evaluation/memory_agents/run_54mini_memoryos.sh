#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${MEMORYOS_PY:-python}"
DATA_ROOT="${DATA_ROOT:-data/recall/mcq_work}"
TOPIC="${TOPIC:-travelPlanning}"
MODEL="${MODEL:-gpt-5.4-mini}"
WORLDS_STR="${WORLDS:-baseline no_store forget}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
STAGE_IDS_STR="${STAGE_IDS:-auto}"
SHORT_TERM_CAPACITY="${SHORT_TERM_CAPACITY:-20}"
METHOD_CONFIG="${METHOD_CONFIG:-tmp/method_configs/memoryos_short_${SHORT_TERM_CAPACITY}.json}"

mkdir -p "$(dirname "$METHOD_CONFIG")"
printf '{"memoryos_short_term_capacity": %s}\n' "$SHORT_TERM_CAPACITY" >"$METHOD_CONFIG"

read -r -a WORLDS <<< "$WORLDS_STR"
mapfile -t RENDERED_FILES < <(find "$DATA_ROOT/$TOPIC" -mindepth 2 -maxdepth 2 -name 'mcq_questions.json' | sort)

if [[ "${#RENDERED_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no MCQ files found under $DATA_ROOT/$TOPIC"
  exit 1
fi

count=0
for rendered in "${RENDERED_FILES[@]}"; do
  count=$((count + 1))
  if [[ "$LIMIT" -gt 0 && "$count" -gt "$LIMIT" ]]; then
    break
  fi
  if [[ "$STAGE_IDS_STR" == "auto" ]]; then
    mapfile -t STAGE_LIST < <(jq -r '[.whole_recall_set[], .slot_recall_set[]] | .[] | .timestamp | capture("(?<stage>stage_[0-9]+)_").stage' "$rendered" | sort -u -V)
  else
    read -r -a STAGE_LIST <<< "$STAGE_IDS_STR"
  fi
  for stage_id in "${STAGE_LIST[@]}"; do
    for world in "${WORLDS[@]}"; do
    mapfile -t OUT_INFO < <("$PYTHON_BIN" - "$rendered" "$world" "$MODEL" "$stage_id" <<'PY'
from memory_control_tests.evaluation.paths import default_output_path
import sys
print(default_output_path(sys.argv[1], sys.argv[2], "all_stages", "memoryos", sys.argv[3], stage_id=sys.argv[4]))
PY
)
    out="${OUT_INFO[0]}"
    if [[ "$OVERWRITE" != "1" && -f "$out" ]]; then
      echo "SKIP MemoryOS $world $stage_id -> $out"
      continue
    fi
    echo "RUN MemoryOS $world $stage_id -> $out"
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      "$PYTHON_BIN" -m memory_control_tests.evaluation.mem_evals \
        --method memoryos \
        --rendered "$rendered" \
        --model "$MODEL" \
        --world "$world" \
        --ask_period all_stages \
        --stage_id "$stage_id" \
        --method_config "$METHOD_CONFIG" \
        --output "$out"
    done
  done
done
