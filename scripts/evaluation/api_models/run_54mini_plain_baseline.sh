#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/recall/mcq_work}"
TOPICS_STR="${TOPICS:-travelPlanning financialConsultation medicalConsultation}"
MODEL="${MODEL:-gpt-5.4-mini}"
WORLDS_STR="${WORLDS:-baseline}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
WORKERS="${WORKERS:-10}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
ASK_PERIOD="${ASK_PERIOD:-all_stages}"
STAGE_IDS_STR="${STAGE_IDS:-auto}"
RUN_LOG_MD="${RUN_LOG_MD:-eval_results/plain_api_run_log.md}"
RUN_LOG_TSV="${RUN_LOG_TSV:-eval_results/eval_run_log.tsv}"

read -r -a TOPICS <<< "$TOPICS_STR"
read -r -a WORLDS <<< "$WORLDS_STR"

format_duration() {
  local seconds="$1"
  printf '%02d:%02d:%02d' $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

ensure_run_log() {
  if [[ ! -f "$RUN_LOG_MD" ]]; then
    mkdir -p "$(dirname "$RUN_LOG_MD")"
    {
      echo "| model | api | topic | rendered | world | time | cost | output | status |"
      echo "|---|---|---|---|---|---:|---:|---|---|"
    } > "$RUN_LOG_MD"
  fi
  if [[ ! -f "$RUN_LOG_TSV" ]]; then
    mkdir -p "$(dirname "$RUN_LOG_TSV")"
    printf 'api_model\tmode\ttopic\trendered\tworld\task_period\tworkers\trequest_timeout\tdata_root\tnum_mcq\ttime\ttime_seconds\ttime_per_mcq_seconds\tcost\toutput\tstatus\n' > "$RUN_LOG_TSV"
  fi
}

find_rendered_files() {
  local topic="$1"
  if [[ -d "$DATA_ROOT/test/$topic/specs" ]]; then
    find "$DATA_ROOT/test/$topic/specs" -maxdepth 1 -name '*.mcq_questions.json' | sort
    return
  fi
  if [[ -d "$DATA_ROOT/$topic" ]]; then
    mapfile -t rendered_files < <(find "$DATA_ROOT/$topic" -mindepth 2 -maxdepth 2 -name 'mcq_questions.json' | sort)
    if [[ "${#rendered_files[@]}" -gt 0 ]]; then
      printf '%s\n' "${rendered_files[@]}"
      return
    fi
    find "$DATA_ROOT/$topic" -mindepth 2 -maxdepth 2 -name 'whole_recall.json' | sort
    return
  fi
}

for topic in "${TOPICS[@]}"; do
  mapfile -t RENDERED_FILES < <(find_rendered_files "$topic")
  if [[ "${#RENDERED_FILES[@]}" -eq 0 ]]; then
    echo "ERROR: no MCQ files found for topic '$topic' under $DATA_ROOT"
    exit 1
  fi

  count=0
  for rendered in "${RENDERED_FILES[@]}"; do
    count=$((count + 1))
    if [[ "$LIMIT" -gt 0 && "$count" -gt "$LIMIT" ]]; then
      break
    fi

    if [[ "$STAGE_IDS_STR" == "auto" ]]; then
      if [[ "$(basename "$rendered")" == "whole_recall.json" ]]; then
        slot_rendered="$(dirname "$rendered")/slot_recall.json"
        if [[ -f "$slot_rendered" ]]; then
          mapfile -t STAGE_LIST < <(jq -r '.items[] | .timestamp | capture("(?<stage>stage_[0-9]+)_").stage' "$slot_rendered" "$rendered" | sort -u -V)
        else
          mapfile -t STAGE_LIST < <(jq -r '.items[] | .timestamp | capture("(?<stage>stage_[0-9]+)_").stage' "$rendered" | sort -u -V)
        fi
      else
        mapfile -t STAGE_LIST < <(jq -r '[.whole_recall_set[], .slot_recall_set[]] | .[] | .timestamp | capture("(?<stage>stage_[0-9]+)_").stage' "$rendered" | sort -u -V)
      fi
    else
      read -r -a STAGE_LIST <<< "$STAGE_IDS_STR"
    fi

    for stage_id in "${STAGE_LIST[@]}"; do
      for world in "${WORLDS[@]}"; do
      mapfile -t OUT_INFO < <("$PYTHON_BIN" - "$rendered" "$world" "$MODEL" "$stage_id" <<'PY'
from memory_control_tests.evaluation.paths import default_output_path
import sys

print(default_output_path(sys.argv[1], sys.argv[2], "all_stages", "plain", sys.argv[3], stage_id=sys.argv[4]))
PY
)
      out="${OUT_INFO[0]}"
      if [[ "$OVERWRITE" != "1" && -f "$out" ]]; then
        echo "SKIP plain $world $stage_id -> $out"
        continue
      fi

      echo "RUN plain $world $stage_id -> $out"
      ensure_run_log
      start_epoch="$(date +%s)"
      tmp_log="$(mktemp)"
      set +e
      "$PYTHON_BIN" -m memory_control_tests.evaluation.mem_evals \
        --method plain \
        --rendered "$rendered" \
        --model "$MODEL" \
        --world "$world" \
        --ask_period "$ASK_PERIOD" \
        --stage_id "$stage_id" \
        --workers "$WORKERS" \
        --request_timeout "$REQUEST_TIMEOUT" \
        --output "$out" > >(tee "$tmp_log") 2> >(tee -a "$tmp_log" >&2)
      status="$?"
      set -e
      end_epoch="$(date +%s)"
      duration_seconds="$((end_epoch - start_epoch))"
      duration="$(format_duration "$duration_seconds")"
      cost="$(sed -n 's/.*delta=\$\([0-9.][0-9.]*\).*/\1/p' "$tmp_log" | tail -1)"
      if [[ -z "$cost" ]]; then
        cost="n/a"
      fi
      num_mcq="$("$PYTHON_BIN" - "$out" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
print(len(data.get("whole_recall_results", []) or []) + len(data.get("slot_recall_results", []) or []))
PY
)"
      if [[ "$num_mcq" -gt 0 ]]; then
        time_per_mcq="$("$PYTHON_BIN" - "$duration_seconds" "$num_mcq" <<'PY'
import sys
print(f"{int(sys.argv[1]) / int(sys.argv[2]):.3f}")
PY
)"
      else
        time_per_mcq="n/a"
      fi
      rendered_name="$(basename "$(dirname "$rendered")")"
      printf '| %s | plain api | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$MODEL" "$topic" "$rendered_name" "$world" "$duration" "$cost" "$out" "$status" >> "$RUN_LOG_MD"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$MODEL" "plain" "$topic" "$rendered_name" "$world" "$stage_id" "$WORKERS" "$REQUEST_TIMEOUT" "$DATA_ROOT" \
        "$num_mcq" "$duration" "$duration_seconds" "$time_per_mcq" "$cost" "$out" "$status" >> "$RUN_LOG_TSV"
      rm -f "$tmp_log"
      if [[ "$status" -ne 0 ]]; then
        exit "$status"
      fi
      done
    done
  done
done
