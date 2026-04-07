#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOPICS="${TOPICS:-legalConsultation financialConsultation medicalConsultation}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
TMP_DIR="${TMP_DIR:-tmp}"

cd "${ROOT_DIR}"
mkdir -p "${TMP_DIR}"

is_topic_complete() {
  local topic="$1"
  python - "$topic" <<'PY'
import json
import sys
from pathlib import Path

topic = sys.argv[1]
topic_dir = Path("data/output") / topic
required = [
    "General Personal History Late Stage",
    "Contextual Personal History Late Stage",
    "Conversation Late Stage",
]
files = sorted(topic_dir.glob("conversation_*.json"))
if len(files) != 10:
    print("0")
    raise SystemExit(0)

for path in files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        print("0")
        raise SystemExit(0)
    if not all(k in data for k in required):
        print("0")
        raise SystemExit(0)

print("1")
PY
}

while true; do
  all_ready=1
  echo "[$(date '+%F %T')] checking topics: ${TOPICS}"
  for topic in ${TOPICS}; do
    ready="$(is_topic_complete "${topic}")"
    if [[ "${ready}" == "1" ]]; then
      echo "  [ready] ${topic}"
    else
      echo "  [wait ] ${topic}"
      all_ready=0
    fi
  done

  if [[ "${all_ready}" == "1" ]]; then
    echo "[$(date '+%F %T')] all topics complete; launching construct_world_and_mcq.sh"
    exec bash "${SCRIPT_DIR}/construct_world_and_mcq.sh"
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done
