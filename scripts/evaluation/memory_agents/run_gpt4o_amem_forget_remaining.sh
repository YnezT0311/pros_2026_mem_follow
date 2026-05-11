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

run_case() {
  local persona="$1"
  local ask="$2"
  local rendered="data/test/${TOPIC}/specs/conversation_${TOPIC}_persona${persona}_sample0.recall_rendered.json"
  local stem
  local suffix=".forget"
  stem="$(basename "$rendered" .recall_rendered.json)"
  if [[ "$ask" != "Conversation Late Stage" ]]; then
    suffix="${suffix}.$(period_tag "$ask")"
  fi
  local out="eval_results/${TOPIC}/forget/${MODEL}+A-Mem/${stem}${suffix}.a_mem_retrieval_eval_${MODEL}.json"
  echo "[forget-remaining] persona=${persona} ask=${ask} -> ${out}"
  HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
    "$AMEM_PY" -m memory_control_tests.evaluation.evaluate_amem_recall_mcqs \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world forget \
      --ask_period "$ask" \
      --api_key_file keys/openrouter_key.txt \
      --output "$out"
}

# Resume from the known breakpoint: persona0 late, then all persona1-9.
run_case 0 "Conversation Late Stage"

for persona in 1 2 3 4 5 6 7 8 9; do
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_case "$persona" "$ask"
  done
done

