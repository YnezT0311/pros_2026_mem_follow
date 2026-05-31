#!/usr/bin/env bash
# Single-item application baseline runner. Filters by_world JSON to one item,
# rewrites the 'world' tag so output doesn't clobber the full baseline, runs eval.
# Usage: ITEM_ID=<id> MODEL=<model> ./run_single_item.sh
# Example:
#   ITEM_ID=financialConsultation_persona0_sample0_stage_01 MODEL=gpt-5.4-mini \
#     ./scripts/evaluation/api_models/run_single_item.sh
# Output:
#   eval_results/application/<world>__single_<topic>_<stage>/<model>/...json
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-gpt-5.4-mini}"
ITEM_ID="${ITEM_ID:?ITEM_ID required}"
DATA_ROOT="${DATA_ROOT:-data/application/mcq/by_world}"
WORLDS_STR="${WORLDS:-seen_baseline never_seen_baseline}"
WORKERS="${WORKERS:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
TMP_DIR="${TMP_DIR:-tmp/single_item_inputs}"
API_KEY_FILE="${API_KEY_FILE:-keys/openrouter_key.txt}"
mkdir -p "$TMP_DIR"

# Stable tag including topic and stage. Stage-only tags clobber results across
# topics, e.g. travelPlanning_stage_06 vs medicalConsultation_stage_06.
TAG=$($PYTHON_BIN - "$ITEM_ID" <<'PY'
import re
import sys

item_id = sys.argv[1]
topic = item_id.split("_persona", 1)[0]
match = re.search(r"(stage_[0-9]+)", item_id)
stage = match.group(1) if match else item_id[-12:]
safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{topic}_{stage}")
print(safe)
PY
)

read -r -a WORLDS <<< "$WORLDS_STR"
for world in "${WORLDS[@]}"; do
  src="$DATA_ROOT/$world.json"
  new_world="${world}__single_${TAG}"
  dst="$TMP_DIR/${new_world}.json"
  $PYTHON_BIN -c "
import json,sys
d=json.load(open('$src'))
items=[i for i in d['items'] if i['id']=='$ITEM_ID']
if not items:
    print(f'ERR: no item $ITEM_ID in $src',file=sys.stderr); sys.exit(1)
for it in items: it['world']='$new_world'
d['world']='$new_world'
d['items']=items
json.dump(d,open('$dst','w'),ensure_ascii=False,indent=2)
print(f'  wrote 1 item to $dst (world=$new_world)')
"
  echo "RUN single-item world=$new_world model=$MODEL"
  $PYTHON_BIN -m memory_control_tests.evaluation.application \
    --input "$dst" --model "$MODEL" --api_key_file "$API_KEY_FILE" \
    --workers "$WORKERS" --request_timeout "$REQUEST_TIMEOUT" --overwrite
done
