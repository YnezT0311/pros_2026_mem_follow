#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMING="${TIMING:-./human_timing.json}"
RESULTS="${RESULTS:-./results}"
LIMIT="${LIMIT:-4}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="${SESSION_DIR:-./claude_session}"
SCRIPT_PATH="./evaluate_claude_web.py"
TOPIC="${TOPIC:-financialConsultation}"

if [[ -n "${DATA:-}" ]]; then
  DATA_DIR="$DATA"
elif [[ -d "../data/benchmark_work_v2" ]]; then
  DATA_DIR="../data/benchmark_work_v2"
else
  echo "ERROR: could not find a data directory."
  echo "Looked for ../data/benchmark_work_v2 relative to: $SCRIPT_DIR"
  echo "Set DATA=/path/to/data and rerun."
  exit 1
fi

# Operational notes:
# - login-first / eval-second:
#   this script always runs the login step first, then runs evaluation worlds.
# - automatic cleanup:
#   the Claude evaluator performs pre-session delete-all-chat-history,
#   pre-session clear-memory, deletion of the temporary clear-memory chat,
#   and post-session delete-current-chat automatically.
# - readiness:
#   evaluate_claude_web.py waits for the input box, then pauses for your Enter
#   before automated actions begin.
# - outputs:
#   results are stored per persona and per session under:
#   results/claude_web_results/<topic>/<sample_id>/test_type_<world>/
# - resume:
#   a session is skipped only when session_result.json exists and
#   status == "completed"; sessions with status == "error" are rerun.
# - click recorder:
#   use record_claude_web_clicks.py separately for selector debugging.
#
# Common usage:
#   TOPIC=financialConsultation LIMIT=1 HISTORY_RATE=0.2 WORLDS="baseline forget no_store" ./run_claude_eval.sh
#
# Default worlds to run. Override with:
#   WORLDS="baseline forget no_store" ./run_claude_eval.sh
WORLDS_STR="${WORLDS:-baseline forget no_store}"
read -r -a WORLDS <<< "$WORLDS_STR"

mkdir -p "$RESULTS"

echo "============================================"
echo "Starting Claude web evaluation"
echo "Personas per world: $LIMIT"
echo "Topic: $TOPIC"
echo "Worlds: ${WORLDS[*]}"
echo "Timing profile: $TIMING"
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS"
echo "Session dir: $SESSION_DIR"
echo "============================================"

echo ""
echo "[0/${#WORLDS[@]}] LOGIN — complete Claude login in the browser, then press Enter"
python "$SCRIPT_PATH" \
  --login \
  --session_dir "$SESSION_DIR" \
  --timing_profile "$TIMING"

echo ""
echo "Login step finished. The evaluation step will open Claude again"
echo "and begin automatically once the chat UI is ready."

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
    --output "$RESULTS/${WORLD}.jsonl"
  idx=$((idx + 1))
done

echo ""
echo "============================================"
echo "All done. Results in $RESULTS/"
echo "============================================"
