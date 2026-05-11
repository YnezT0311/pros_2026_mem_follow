#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

AMEM_PY="/home/yao/.conda/envs/amem/bin/python"
MODEL="gpt-4o"
TOPIC="travelPlanning"

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
  out_dir="eval_results/${TOPIC}/${world}/${MODEL}+A-Mem"
  mkdir -p "$out_dir"
  filename="${stem}.${world}"
  if [[ "$ask_period" != "Conversation Late Stage" ]]; then
    filename="${filename}.$(period_tag "$ask_period")"
  fi
  filename="${filename}.a_mem_retrieval_eval_${MODEL}.json"
  echo "${out_dir}/${filename}"
}

run_amem_case() {
  local rendered="$1" world="$2" ask_period="$3"
  local out
  out="$(memory_output "$rendered" "$world" "$ask_period")"
  echo "[run] A-Mem $world $ask_period -> $out"
  HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
    "$AMEM_PY" -m memory_control_tests.evaluation.evaluate_amem_recall_mcqs \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world "$world" \
      --ask_period "$ask_period" \
      --api_key_file keys/openrouter_key.txt \
      --output "$out"
}

echo "== A-Mem refresh run: baseline persona0-3 =="
for persona in 0 1 2 3; do
  rendered="$(rendered_for_persona "$persona")"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_amem_case "$rendered" baseline "$ask"
  done
done

echo "== A-Mem refresh run: no_store persona0-3 =="
for persona in 0 1 2 3; do
  rendered="$(rendered_for_persona "$persona")"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_amem_case "$rendered" no_store "$ask"
  done
done

echo "== A-Mem refresh run: forget persona0-9 =="
for persona in 0 1 2 3 4 5 6 7 8 9; do
  rendered="$(rendered_for_persona "$persona")"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_amem_case "$rendered" forget "$ask"
  done
done

