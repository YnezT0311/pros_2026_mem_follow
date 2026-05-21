#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMING="${TIMING:-./human_timing.json}"
RESULTS="${RESULTS:-./results}"
LIMIT="${LIMIT:-0}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="${SESSION_DIR:-./chatgpt_session}"
SCRIPT_PATH="./evaluate_chatgpt_web.py"
DATA_DIR="${DATA:-../data/benchmark_work_v2}"

if [[ -n "${TOPICS:-}" ]]; then
  TOPICS_STR="$TOPICS"
elif [[ -n "${TOPIC:-}" ]]; then
  TOPICS_STR="$TOPIC"
else
  TOPICS_STR="travelPlanning financialConsultation medicalConsultation"
fi
read -r -a TOPIC_LIST <<< "$TOPICS_STR"

WORLDS_STR="${WORLDS:-baseline no_store forget}"
read -r -a WORLDS <<< "$WORLDS_STR"

mkdir -p "$RESULTS"

echo "============================================"
echo "Starting ChatGPT web evaluation"
echo "Personas per topic/world: $LIMIT (0 = all)"
echo "Topics: ${TOPIC_LIST[*]}"
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
total=$(( ${#TOPIC_LIST[@]} * ${#WORLDS[@]} ))
for TOPIC_NAME in "${TOPIC_LIST[@]}"; do
  for WORLD in "${WORLDS[@]}"; do
    WORLD_LABEL="$(printf '%s' "$WORLD" | tr '[:lower:]' '[:upper:]')"
    echo ""
    echo "[$idx/$total] ${TOPIC_NAME} / ${WORLD_LABEL}"
    python "$SCRIPT_PATH" \
      --topic "$TOPIC_NAME" \
      --world "$WORLD" \
      --limit "$LIMIT" \
      --timing_profile "$TIMING" \
      --data_dir "$DATA_DIR" \
      --history_rate "$HISTORY_RATE" \
      --session_dir "$SESSION_DIR" \
      --output "$RESULTS/${TOPIC_NAME}_${WORLD}.jsonl"
    idx=$((idx + 1))
  done
done

echo ""
echo "============================================"
echo "All done. Results in $RESULTS/"
echo "============================================"
