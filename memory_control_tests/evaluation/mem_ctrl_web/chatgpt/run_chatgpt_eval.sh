#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMING="${TIMING:-./human_timing.json}"
RESULTS="${RESULTS:-./results}"
LIMIT="${LIMIT:-0}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="${SESSION_DIR:-./chatgpt_session}"
SCRIPT_PATH="./evaluate_chatgpt_web.py"
DATA_DIR="${DATA:-../data/benchmark_work_v2}"

# Auto-restart configuration
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_DELAY="${RETRY_DELAY:-5}"
RETRY_COUNT=0
SCRIPT_SELF="${BASH_SOURCE[0]}"

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
echo "Auto-restart: enabled (max $MAX_RETRIES retries, $RETRY_DELAY sec delay)"
echo "============================================"

run_evaluation() {
  # Check if login is needed
  if [[ ! -d "$SESSION_DIR" ]]; then
    echo ""
    echo "[LOGIN] Complete ChatGPT login in the browser, then press Enter"
    python3 "$SCRIPT_PATH" \
      --login \
      --session_dir "$SESSION_DIR"
  else
    echo ""
    echo "[RESUME] Using existing session from $SESSION_DIR"
  fi

  idx=1
  total=$(( ${#TOPIC_LIST[@]} * ${#WORLDS[@]} ))
  for TOPIC_NAME in "${TOPIC_LIST[@]}"; do
    for WORLD in "${WORLDS[@]}"; do
      WORLD_LABEL="$(printf '%s' "$WORLD" | tr '[:lower:]' '[:upper:]')"
      echo ""
      echo "[$idx/$total] ${TOPIC_NAME} / ${WORLD_LABEL}"
      python3 "$SCRIPT_PATH" \
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
}

# Run with auto-restart on failure
while true; do
  if run_evaluation; then
    echo "[SUCCESS] Evaluation completed successfully"
    exit 0
  else
    EXIT_CODE=$?
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
      echo "[FAILED] Evaluation failed after $MAX_RETRIES retries"
      echo "Check the errors above and fix the issue manually"
      exit $EXIT_CODE
    fi
    echo ""
    echo "[ERROR] Evaluation failed with exit code $EXIT_CODE"
    echo "[RETRY $RETRY_COUNT/$MAX_RETRIES] Restarting in $RETRY_DELAY seconds..."
    sleep "$RETRY_DELAY"
    RETRY_DELAY=$((RETRY_DELAY + 2))  # Increase delay each retry
  fi
done
