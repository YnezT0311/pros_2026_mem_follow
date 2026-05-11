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

for persona in 0 1 2 3; do
  rendered="data/test/${TOPIC}/specs/conversation_${TOPIC}_persona${persona}_sample0.recall_rendered.json"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    stem="$(basename "$rendered" .recall_rendered.json)"
    suffix=".baseline"
    if [[ "$ask" != "Conversation Late Stage" ]]; then
      suffix="${suffix}.$(period_tag "$ask")"
    fi
    out="eval_results/${TOPIC}/baseline/${MODEL}+A-Mem/${stem}${suffix}.a_mem_retrieval_eval_${MODEL}.json"
    echo "[baseline] persona=${persona} ask=${ask} -> ${out}"
    HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
      "$AMEM_PY" -m memory_control_tests.evaluation.evaluate_amem_recall_mcqs \
        --rendered "$rendered" \
        --model "$MODEL" \
        --world baseline \
        --ask_period "$ask" \
        --api_key_file keys/openrouter_key.txt \
        --output "$out"
  done
done
