#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMING="${TIMING:-./human_timing.json}"
RESULTS="${RESULTS:-./test_results}"
LIMIT="${LIMIT:-1}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="${SESSION_DIR:-./claude_session}"
SCRIPT_PATH="./evaluate_claude_web.py"
TOPIC="${TOPIC:-travelPlanning}"
DATA_DIR="${DATA:-../data/benchmark_work_v2}"
STAGE_ID="${STAGE_ID:-stage_01}"

WORLDS=(baseline no_store)

mkdir -p "$RESULTS"

echo "============================================"
echo "Starting Claude Stage 01 smoke test"
echo "Topic: $TOPIC"
echo "Worlds: ${WORLDS[*]}"
echo "MCQs: whole_recall only"
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS"
echo "============================================"

echo ""
echo "[0/${#WORLDS[@]}] LOGIN — complete Claude login in the browser, then press Enter"
python "$SCRIPT_PATH" \
  --login \
  --session_dir "$SESSION_DIR" \
  --timing_profile "$TIMING"

idx=1
for WORLD in "${WORLDS[@]}"; do
  WORLD_LABEL="$(printf '%s' "$WORLD" | tr '[:lower:]' '[:upper:]')"
  echo ""
  echo "[$idx/${#WORLDS[@]}] ${WORLD_LABEL} — ${STAGE_ID} whole_recall"
  python "$SCRIPT_PATH" \
    --topic "$TOPIC" \
    --world "$WORLD" \
    --limit "$LIMIT" \
    --timing_profile "$TIMING" \
    --data_dir "$DATA_DIR" \
    --history_rate "$HISTORY_RATE" \
    --session_dir "$SESSION_DIR" \
    --stage_id_filter "$STAGE_ID" \
    --qa_family_filter "whole_recall" \
    --overwrite \
    --output "$RESULTS/${WORLD}_${STAGE_ID}_whole_recall.jsonl"
  idx=$((idx + 1))
done

echo ""
echo "============================================"
echo "Stage 01 smoke test done. Results in $RESULTS/"
echo "============================================"
