#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMING="${TIMING:-./human_timing.json}"
RESULTS="${RESULTS:-./results}"
LIMIT="${LIMIT:-1}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="${SESSION_DIR:-./chatgpt_session}"
SCRIPT_PATH="./evaluate_chatgpt_web.py"
TOPIC="${TOPIC:-financialConsultation}"
DATA_DIR="${DATA:-../data/benchmark_work_v2}"

WORLDS_STR="${WORLDS:-baseline no_store forget}"
read -r -a WORLDS <<< "$WORLDS_STR"

mkdir -p "$RESULTS"

echo "============================================"
echo "Starting ChatGPT web evaluation"
echo "Personas per world: $LIMIT"
echo "Topic: $TOPIC"
echo "Worlds: ${WORLDS[*]}"
echo "Timing profile: $TIMING"
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS"
echo "Session dir: $SESSION_DIR"
echo "============================================"

echo ""
echo "[0/${#WORLDS[@]}] LOGIN — complete ChatGPT login in the browser, then press Enter"
python "$SCRIPT_PATH" \
  --login \
  --session_dir "$SESSION_DIR"

idx=1
for WORLD in "${WORLDS[@]}"; do
  WORLD_LABEL="$(printf '%s' "$WORLD" | tr '[:lower:]' '[:upper:]')"
  echo ""
  echo "[$idx/${#WORLDS[@]}] ${WORLD_LABEL}"
  python "$SCRIPT_PATH" \
    --topic "$TOPIC" \
    --world "$WORLD" \
    --limit "$LIMIT" \
    --timing_profile "$TIMING" \
    --data_dir "$DATA_DIR" \
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
