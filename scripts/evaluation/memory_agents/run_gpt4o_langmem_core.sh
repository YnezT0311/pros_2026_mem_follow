#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

LANGMEM_PY="/home/yao/.conda/envs/langmem311/bin/python"
MODEL="gpt-4o"
TOPIC="travelPlanning"
OVERWRITE="${OVERWRITE-0}"
MAX_JOBS="${MAX_JOBS-4}"

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
  local out
  out="$(memory_output "$rendered" "$world" "$ask_period")"
  if [[ "$OVERWRITE" != "1" && -f "$out" ]]; then
    echo "SKIP LangMem $world $ask_period -> $out"
    return
  fi
  echo "RUN LangMem $world $ask_period -> $out"
  "$LANGMEM_PY" -m memory_control_tests.evaluation.evaluate_langmem_recall_mcqs \
    --rendered "$rendered" \
    --model "$MODEL" \
    --world "$world" \
    --ask_period "$ask_period" \
    --api_key_file "keys/openrouter_key.txt" \
    --preload_batch_size 2 \
    --output "$out"
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    sleep 2
  done
}

for persona in 0 1 2 3; do
  rendered="$(rendered_for_persona "$persona")"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    wait_for_slot
    run_case "$rendered" baseline "$ask" &
    wait_for_slot
    run_case "$rendered" no_store "$ask" &
  done
done

for persona in 0 1 2 3 4 5 6 7 8 9; do
  rendered="$(rendered_for_persona "$persona")"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    wait_for_slot
    run_case "$rendered" forget "$ask" &
  done
done

wait
echo "DONE gpt-4o LangMem core worlds"
