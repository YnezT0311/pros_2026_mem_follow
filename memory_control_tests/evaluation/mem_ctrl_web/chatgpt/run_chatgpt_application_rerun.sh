#!/bin/bash
# Rerun ONLY the failed/broken ChatGPT web application sessions, then rebuild clean results.jsonl.
#
# Why this exists: evaluate_chatgpt_web.py resumes by skipping sessions whose
# session_result.json has status=="completed". But some failures are marked
# "completed" while their answer is actually "Conversation has been deleted"
# (chat deleted mid-run) — plain resume would never retry those. This script:
#   1. scans every session_result.json and DELETES any bad session dir
#      (status!=completed, empty/garbage response, or "conversation deleted" /
#       "unusual activity" markers) so resume treats it as missing,
#   2. re-runs (only the missing/non-completed sessions are executed),
#   3. rebuilds each test_type/results.jsonl from the per-session records
#      (results.jsonl is append-only, so this de-dups it into the authoritative set).
#
# Idempotent + self-correcting: if the browser dies again and re-breaks a session,
# just run this script again — step 1 will catch and delete it next time.
#
# Paths/defaults mirror run_chatgpt_application_eval.sh. Override via env, e.g.:
#   RESULTS=/abs/path/to/application_results ./run_chatgpt_application_rerun.sh
#   TOPICS="medicalConsultation" WORLDS="forget no_store" ./run_chatgpt_application_rerun.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMING="${TIMING:-./human_timing.json}"
RESULTS="${RESULTS:-./application_results}"
LIMIT="${LIMIT:-0}"
HISTORY_RATE="${HISTORY_RATE:-0.2}"
SESSION_DIR="${SESSION_DIR:-./chatgpt_application_session}"
SCRIPT_PATH="./evaluate_chatgpt_web.py"
DATA_DIR="${DATA:-../data/application/mcq}"
ROOT="$RESULTS/chatgpt_web_results"

MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_DELAY="${RETRY_DELAY:-5}"

if [[ -n "${TOPICS:-}" ]]; then TOPICS_STR="$TOPICS"
elif [[ -n "${TOPIC:-}" ]]; then TOPICS_STR="$TOPIC"
else TOPICS_STR="travelPlanning financialConsultation medicalConsultation"; fi
read -r -a TOPIC_LIST <<< "$TOPICS_STR"
WORLDS_STR="${WORLDS:-seen_baseline never_seen_baseline no_store forget no_use_active no_use_release}"
read -r -a WORLDS <<< "$WORLDS_STR"

echo "============================================"
echo "ChatGPT web application — RERUN failed sessions"
echo "Topics:  ${TOPIC_LIST[*]}"
echo "Worlds:  ${WORLDS[*]}"
echo "Results: $ROOT"
echo "============================================"

if [[ ! -d "$ROOT" ]]; then
  echo "[error] results root not found: $ROOT"
  echo "        point RESULTS at the dir that CONTAINS chatgpt_web_results/"
  exit 1
fi

# ---- step 1: delete bad session dirs so resume re-runs them ----
echo ""
echo "[scan] deleting failed/broken sessions ..."
python3 - "$ROOT" "${TOPIC_LIST[*]}" "${WORLDS[*]}" <<'PY'
import json, sys, glob, os, shutil
root, topics, worlds = sys.argv[1], set(sys.argv[2].split()), set(sys.argv[3].split())
BAD = ("conversation has been deleted", "unusual activity", "start new chat")
def is_bad(sr):
    try: d = json.load(open(sr))
    except Exception: return True
    if d.get("status") != "completed": return True
    recs = d.get("records") or []
    if not recs: return True
    for r in recs:
        resp = (r.get("model_response") or "").strip()
        if len(resp) < 3 or any(m in resp.lower() for m in BAD): return True
    return False
n = 0
for sr in glob.glob(f"{root}/*/*/test_type_*/session_*/session_result.json"):
    # path: .../<topic>/<sample>/test_type_<world>/session_x/session_result.json
    parts = sr.split("/")
    topic = parts[-5]; world = parts[-3].replace("test_type_", "")
    if topics and topic not in topics: continue
    if worlds and world not in worlds: continue
    if is_bad(sr):
        d = os.path.dirname(sr); shutil.rmtree(d); n += 1
        print("  deleted", os.path.relpath(d, root))
print(f"[scan] removed {n} bad session dir(s)")
PY

# ---- step 2: resume run (only missing/non-completed sessions execute) ----
run_pass() {
  if [[ ! -d "$SESSION_DIR" ]]; then
    echo ""
    echo "[LOGIN] complete ChatGPT login in the browser, then press Enter"
    python3 "$SCRIPT_PATH" --login --session_dir "$SESSION_DIR"
  fi
  for TOPIC_NAME in "${TOPIC_LIST[@]}"; do
    for WORLD in "${WORLDS[@]}"; do
      echo ""
      echo "[rerun] ${TOPIC_NAME} / ${WORLD}"
      python3 "$SCRIPT_PATH" \
        --dataset application --topic "$TOPIC_NAME" --world "$WORLD" \
        --limit "$LIMIT" --timing_profile "$TIMING" --data_dir "$DATA_DIR" \
        --history_rate "$HISTORY_RATE" --session_dir "$SESSION_DIR" \
        --output "$RESULTS/${TOPIC_NAME}_${WORLD}.jsonl"
    done
  done
}

RETRY_COUNT=0
while true; do
  if run_pass; then break; fi
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
    echo "[FAILED] rerun aborted after $MAX_RETRIES retries"; break
  fi
  echo "[RETRY $RETRY_COUNT/$MAX_RETRIES] restarting in $RETRY_DELAY s ..."
  sleep "$RETRY_DELAY"; RETRY_DELAY=$((RETRY_DELAY + 2))
done

# ---- step 3: rebuild clean results.jsonl from session records ----
echo ""
echo "[rebuild] regenerating results.jsonl from session_result.json ..."
python3 - "$ROOT" "${TOPIC_LIST[*]}" "${WORLDS[*]}" <<'PY'
import json, sys, glob, os
root, topics, worlds = sys.argv[1], set(sys.argv[2].split()), set(sys.argv[3].split())
for td in sorted(glob.glob(f"{root}/*/*/test_type_*")):
    parts = td.split("/"); topic = parts[-3]; world = parts[-1].replace("test_type_", "")
    if topics and topic not in topics: continue
    if worlds and world not in worlds: continue
    recs = []
    for sr in sorted(glob.glob(f"{td}/session_*/session_result.json")):
        try: d = json.load(open(sr))
        except Exception: continue
        recs += d.get("records") or []
    with open(os.path.join(td, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {os.path.relpath(td, root)}: {len(recs)} records")
PY

echo ""
echo "[done] rerun + rebuild complete."
