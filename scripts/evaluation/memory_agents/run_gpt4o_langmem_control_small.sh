#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

LANGMEM_PY="/home/yao/.conda/envs/langmem311/bin/python"
MODEL="gpt-4o"
TOPIC="travelPlanning"
OVERWRITE="${OVERWRITE-0}"
RUN_TAG="${RUN_TAG-$(date +%Y%m%d_%H%M%S)}"
CASE_LOG_DIR="tmp/langmem_control_small_${RUN_TAG}"
mkdir -p "$CASE_LOG_DIR"

FAILURES=0
TOTAL=0
DONE=0

period_tag() {
  case "$1" in
    "Conversation Early Stage") echo "early" ;;
    "Conversation Intermediate Stage") echo "intermediate" ;;
    "Conversation Late Stage") echo "late" ;;
    *) echo "unknown" ;;
  esac
}

rendered_for_persona() {
  local persona="$1"
  echo "data/test/${TOPIC}/specs/conversation_${TOPIC}_persona${persona}_sample0.recall_rendered.json"
}

memory_output() {
  local rendered="$1" world="$2" ask_period="$3"
  local stem out_dir filename
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/${TOPIC}/${world}/${MODEL}+LangMem"
  mkdir -p "$out_dir"
  filename="${stem}.${world}"
  if [[ "$ask_period" != "Conversation Late Stage" ]]; then
    filename="${filename}.$(period_tag "$ask_period")"
  fi
  filename="${filename}.langmem_retrieval_eval_${MODEL}.json"
  echo "${out_dir}/${filename}"
}

run_case() {
  local rendered="$1" world="$2" ask_period="$3"
  local out stem ask_tag case_log status
  out="$(memory_output "$rendered" "$world" "$ask_period")"
  stem="$(basename "$rendered" .recall_rendered.json)"
  ask_tag="$(period_tag "$ask_period")"
  case_log="${CASE_LOG_DIR}/${stem}.${world}.${ask_tag}.log"

  if [[ "$OVERWRITE" != "1" && -f "$out" ]]; then
    echo "SKIP LangMem $world $ask_period -> $out"
    return
  fi

  TOTAL=$((TOTAL + 1))
  echo "RUN LangMem $world $ask_period -> $out"
  {
    echo "[start] $(date '+%F %T')"
    echo "[rendered] $rendered"
    echo "[world] $world"
    echo "[ask_period] $ask_period"
    echo "[output] $out"
    "$LANGMEM_PY" -m memory_control_tests.evaluation.evaluate_langmem_recall_mcqs \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world "$world" \
      --ask_period "$ask_period" \
      --api_key_file "keys/openrouter_key.txt" \
      --preload_batch_size 2 \
      --output "$out"
  } >"$case_log" 2>&1
  status=$?

  if [[ $status -eq 0 ]]; then
    DONE=$((DONE + 1))
    echo "DONE LangMem $world $ask_period -> $out"
  else
    FAILURES=$((FAILURES + 1))
    echo "FAIL LangMem $world $ask_period -> $out"
    echo "  log: $case_log"
  fi
}

for persona in 0 1 2 3; do
  rendered="$(rendered_for_persona "$persona")"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_case "$rendered" no_store "$ask"
    run_case "$rendered" forget "$ask"
  done
done

echo "SUMMARY LangMem no_store+forget small"
echo "  run_tag: $RUN_TAG"
echo "  case_logs: $CASE_LOG_DIR"
echo "  launched: $TOTAL"
echo "  completed: $DONE"
echo "  failed: $FAILURES"

if [[ $FAILURES -gt 0 ]]; then
  exit 1
fi
