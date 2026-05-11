#!/usr/bin/env bash

set -euo pipefail

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
SESSIONS_DIR="$CODEX_HOME/sessions"
SESSION_INDEX="$CODEX_HOME/session_index.jsonl"
PROJECT_PATH="${1:-$PWD}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required."
  exit 1
fi

if [[ ! -d "$SESSIONS_DIR" ]]; then
  echo "Codex sessions directory not found: $SESSIONS_DIR"
  exit 1
fi

TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

build_session_list() {
  python3 - "$SESSIONS_DIR" "$SESSION_INDEX" "$PROJECT_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

sessions_dir = Path(sys.argv[1]).expanduser()
session_index = Path(sys.argv[2]).expanduser()
project_path = os.path.abspath(os.path.expanduser(sys.argv[3]))

index_titles = {}
index_updated = {}
if session_index.exists():
    with session_index.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("id")
            if not sid:
                continue
            index_titles[sid] = obj.get("thread_name", "")
            index_updated[sid] = obj.get("updated_at", "")


def shorten(text, limit):
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_user_request(text):
    if not text:
        return ""
    markers = [
        "## My request for Codex:",
        "My request for Codex:",
        "User request:",
    ]
    for marker in markers:
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            if tail:
                return tail
    return text.strip()


rows = []
for path in sessions_dir.rglob("*.jsonl"):
    session_id = None
    cwd = ""
    first_user_prompt = ""
    first_request_prompt = ""
    first_user_prompt_short = ""
    first_ts = ""
    last_ts = ""

    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                ts = obj.get("timestamp", "")
                if ts:
                    if not first_ts:
                        first_ts = ts
                    last_ts = ts

                typ = obj.get("type")
                payload = obj.get("payload", {})

                if typ == "session_meta":
                    session_id = payload.get("id", session_id)
                    cwd = payload.get("cwd", cwd)

                candidate = ""
                if typ == "event_msg" and payload.get("type") == "user_message":
                    candidate = payload.get("message", "")
                elif typ == "response_item" and payload.get("type") == "message" and payload.get("role") == "user":
                    content = payload.get("content", [])
                    parts = []
                    for item in content:
                        text = item.get("text") or item.get("input_text") or ""
                        if text:
                            parts.append(text)
                    candidate = "\n".join(parts).strip()

                if candidate and not first_user_prompt:
                    first_user_prompt = candidate

                if candidate and not first_request_prompt:
                    extracted = extract_user_request(candidate)
                    if extracted and not extracted.startswith("# AGENTS.md instructions"):
                        first_request_prompt = extracted
    except OSError:
        continue

    if not session_id or not cwd:
        continue

    abs_cwd = os.path.abspath(os.path.expanduser(cwd))
    if abs_cwd != project_path and not abs_cwd.startswith(project_path.rstrip(os.sep) + os.sep):
        continue

    cleaned_prompt = first_request_prompt or extract_user_request(first_user_prompt)
    title = index_titles.get(session_id, "").strip()
    prompt = shorten(cleaned_prompt, 140)
    if not title:
        title = prompt or path.stem

    rows.append(
        {
            "session_id": session_id,
            "title": shorten(title, 90),
            "prompt": prompt or "-",
            "cwd": abs_cwd,
            "updated_at": index_updated.get(session_id, "") or last_ts or first_ts,
            "path": str(path),
        }
    )


rows.sort(key=lambda row: row["updated_at"], reverse=True)

for row in rows:
    print(
        "\t".join(
            [
                row["session_id"],
                row["updated_at"],
                row["title"],
                row["prompt"],
                row["cwd"],
                row["path"],
            ]
        )
    )
PY
}

draw_header() {
  printf '\n'
  printf '===============================================================\n'
  printf ' Codex Thread Manager\n'
  printf ' Project: %s\n' "$PROJECT_PATH"
  printf ' Source : %s\n' "$SESSIONS_DIR"
  printf '===============================================================\n'
}

render_list() {
  local i=1
  while IFS=$'\t' read -r session_id updated_at title prompt cwd path; do
    printf '[%02d] %s\n' "$i" "$title"
    printf '     updated: %s\n' "${updated_at:-unknown}"
    printf '     prompt : %s\n' "$prompt"
    printf '     id     : %s\n' "$session_id"
    printf '     file   : %s\n' "$path"
    printf '\n'
    i=$((i + 1))
  done <"$TMP_LIST"
}

delete_sessions() {
  local selected_file="$1"
  local backup_index
  backup_index="$(mktemp)"

  if [[ -f "$SESSION_INDEX" ]]; then
    cp "$SESSION_INDEX" "$backup_index"
  else
    : >"$backup_index"
  fi

  while IFS=$'\t' read -r session_id updated_at title prompt cwd path; do
    if [[ -f "$path" ]]; then
      rm -f "$path"
      printf 'Deleted session file: %s\n' "$path"
    fi
  done <"$selected_file"

  if [[ -f "$SESSION_INDEX" ]]; then
    python3 - "$backup_index" "$selected_file" "$SESSION_INDEX" <<'PY'
import json
import sys

backup_index, selected_file, session_index = sys.argv[1:4]
ids = set()
with open(selected_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ids.add(line.split("\t", 1)[0])

with open(backup_index, "r", encoding="utf-8") as src, open(session_index, "w", encoding="utf-8") as dst:
    for raw in src:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            dst.write(raw)
            continue
        if obj.get("id") in ids:
            continue
        dst.write(raw)
PY
    printf 'Updated session index: %s\n' "$SESSION_INDEX"
  fi

  rm -f "$backup_index"
}

build_session_list >"$TMP_LIST"

if [[ ! -s "$TMP_LIST" ]]; then
  draw_header
  echo "No Codex threads found for this project."
  exit 0
fi

draw_header
render_list

TOTAL="$(wc -l <"$TMP_LIST" | tr -d ' ')"
printf 'Select thread numbers to delete (example: 1 3 5), or q to quit: '
read -r selection

if [[ "$selection" == "q" || "$selection" == "Q" || -z "$selection" ]]; then
  echo "No changes made."
  exit 0
fi

SELECTED_TMP="$(mktemp)"
trap 'rm -f "$TMP_LIST" "$SELECTED_TMP"' EXIT

for token in $selection; do
  if [[ ! "$token" =~ ^[0-9]+$ ]]; then
    echo "Invalid selection: $token"
    exit 1
  fi
  if (( token < 1 || token > TOTAL )); then
    echo "Selection out of range: $token"
    exit 1
  fi
  sed -n "${token}p" "$TMP_LIST" >>"$SELECTED_TMP"
done

if [[ ! -s "$SELECTED_TMP" ]]; then
  echo "No valid threads selected."
  exit 1
fi

printf '\nThreads selected for deletion:\n\n'
render_selected() {
  local i=1
  while IFS=$'\t' read -r session_id updated_at title prompt cwd path; do
    printf '(%d) %s\n' "$i" "$title"
    printf '    prompt: %s\n' "$prompt"
    printf '    id    : %s\n' "$session_id"
    printf '    file  : %s\n' "$path"
    printf '\n'
    i=$((i + 1))
  done <"$SELECTED_TMP"
}
render_selected

printf "Type DELETE to permanently remove these thread files: "
read -r confirm
if [[ "$confirm" != "DELETE" ]]; then
  echo "Deletion cancelled."
  exit 0
fi

delete_sessions "$SELECTED_TMP"
echo "Done."
