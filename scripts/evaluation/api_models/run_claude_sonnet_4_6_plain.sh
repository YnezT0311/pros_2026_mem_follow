#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/recall/mcq_work}"
TOPIC="${TOPIC:-travelPlanning}"
MODEL="${MODEL:-anthropic/claude-sonnet-4.6}"
WORLDS_STR="${WORLDS:-baseline no_store forget no_use_active no_use_release}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
STAGE_IDS_STR="${STAGE_IDS:-auto}"

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
path = default_output_path(sys.argv[1], sys.argv[2], "all_stages", "plain", sys.argv[3], stage_id=sys.argv[4])
print(path)
PY
)
    out="${OUT_INFO[0]}"
    if [[ "$OVERWRITE" != "1" && -f "$out" ]]; then
      echo "SKIP plain $world $stage_id -> $out"
      continue
    fi
    echo "RUN plain $world $stage_id -> $out"
    "$PYTHON_BIN" -m memory_control_tests.evaluation.mem_evals \
      --method plain \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world "$world" \
      --ask_period all_stages \
      --stage_id "$stage_id" \
      --output "$out"
    done
  done
done
