#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

TIMING="human_timing.json"
DATA="./data"
RESULTS="./results"
LIMIT="${LIMIT:-4}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="./chatgpt_session"
SCRIPT_PATH="memory_control_tests/evaluation/chatgpt/evaluate_chatgpt_web.py"

# Operational notes:
# - login-first / eval-second:
#   this script always runs the login step first, then runs evaluation worlds.
# - manual cleanup:
#   evaluation runs pass --manual_cleanup so the user can delete memory and
#   delete the current chat manually between sessions.
# - readiness:
#   evaluate_chatgpt_web.py waits for the input box, then waits an additional
#   grace period controlled by --ready_grace_sec before starting.
# - outputs:
#   results are stored per persona and per session under:
#   results/chatgpt_web_results/<topic>/<sample_id>/test_type_<world>/
# - resume:
#   a session is skipped only when session_result.json exists and
#   status == "completed"; sessions with status == "error" are rerun.
# - click recorder:
#   use memory_control_tests/evaluation/chatgpt/record_chatgpt_web_clicks.py separately for selector debugging.

# Default worlds to run. Override with:
#   WORLDS="baseline forget no_store" memory_control_tests/evaluation/chatgpt/run_chatgpt_eval.sh
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
python "$SCRIPT_PATH" \
  --login \
  --session_dir "$SESSION_DIR"

idx=1
for WORLD in "${WORLDS[@]}"; do
  WORLD_LABEL="$(printf '%s' "$WORLD" | tr '[:lower:]' '[:upper:]')"
  echo ""
  echo "[$idx/${#WORLDS[@]}] ${WORLD_LABEL}"
  python "$SCRIPT_PATH" \
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
