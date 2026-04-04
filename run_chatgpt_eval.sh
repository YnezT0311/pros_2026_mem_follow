#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

TIMING="human_timing.json"
DATA="./data"
RESULTS="./results"
LIMIT="${LIMIT:-4}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="./chatgpt_session"

# Default worlds to run. Override with:
#   WORLDS="baseline forget no_store" ./run_chatgpt_eval.sh
WORLDS_STR="${WORLDS:-baseline forget no_store}"
read -r -a WORLDS <<< "$WORLDS_STR"

mkdir -p "$RESULTS"

echo "============================================"
echo "Starting ChatGPT web evaluation"
echo "Personas per world: $LIMIT"
echo "Worlds: ${WORLDS[*]}"
echo "============================================"

echo ""
echo "[0/${#WORLDS[@]}] LOGIN — complete Cloudflare verification in the browser, then press Enter"
python evaluate_chatgpt_web.py \
  --login \
  --session_dir "$SESSION_DIR"

idx=1
for WORLD in "${WORLDS[@]}"; do
  WORLD_LABEL="$(printf '%s' "$WORLD" | tr '[:lower:]' '[:upper:]')"
  echo ""
  echo "[$idx/${#WORLDS[@]}] ${WORLD_LABEL}"
  python evaluate_chatgpt_web.py \
    --topic travelPlanning \
    --world "$WORLD" \
    --limit "$LIMIT" \
    --timing_profile "$TIMING" \
    --data_dir "$DATA" \
    --history_rate "$HISTORY_RATE" \
    --session_dir "$SESSION_DIR" \
    --manual_cleanup \
    --output "$RESULTS/${WORLD}.jsonl"
  idx=$((idx + 1))
done

echo ""
echo "============================================"
echo "All done. Results in $RESULTS/"
echo "============================================"
