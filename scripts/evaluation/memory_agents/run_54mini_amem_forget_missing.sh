#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

AMEM_PY="${AMEM_PY:-python}"
MODEL="gpt-5.4-mini"
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

forget_output() {
  local rendered="$1" ask_period="$2"
  local stem filename out_dir
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/${TOPIC}/forget/${MODEL}+A-Mem"
  mkdir -p "$out_dir"
  filename="${stem}.forget"
  if [[ "$ask_period" != "Conversation Late Stage" ]]; then
    filename="${filename}.$(period_tag "$ask_period")"
  fi
  filename="${filename}.a_mem_retrieval_eval_${MODEL}.json"
  echo "${out_dir}/${filename}"
}

run_forget_case() {
  local rendered="$1" ask_period="$2"
  local out
  out="$(forget_output "$rendered" "$ask_period")"
  if [[ -f "$out" ]]; then
    echo "SKIP A-Mem forget $ask_period -> $out"
    return
  fi
  echo "RUN A-Mem forget $ask_period -> $out"
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$AMEM_PY" -m memory_control_tests.evaluation.mem_evals \
      --method amem \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world forget \
      --ask_period "$ask_period" \
      --output "$out"
}

for persona in 5 6 7 8 9; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN gpt-5.4-mini A-Mem forget-missing persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_forget_case "$rendered" "$ask"
  done
done

echo "DONE gpt54mini_amem_forget_missing"
