#!/usr/bin/env bash
set -euo pipefail

cd /mnt/yao_data/proj_2026_agent/MemoryCtrl || exit 1

PYTHON_BIN="/home/yao/.conda/envs/mem0/bin/python"
RENDERED_TEMPLATE="data/test/travelPlanning/specs/conversation_travelPlanning_persona%s_sample0.recall_rendered.json"
OUTPUT_TEMPLATE="eval_results/travelPlanning/%s/gpt-4o+mem0/conversation_travelPlanning_persona%s_sample0.%s.%s.mem0_retrieval_eval_gpt-4o.json"

for world in no_store forget; do
  for i in 0 1 2 3 4 5 6 7 8 9; do
    for period in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
      tag=$(echo "$period" | sed 's/Conversation //; s/ Stage//; s/ /_/g' | tr '[:upper:]' '[:lower:]')
      rendered=$(printf "$RENDERED_TEMPLATE" "$i")
      output=$(printf "$OUTPUT_TEMPLATE" "$world" "$i" "$world" "$tag")
      echo "[$(date '+%F %T')] world=$world persona=$i period=$period"
      "$PYTHON_BIN" -u -m memory_control_tests.evaluation.evaluate_mem0_recall_mcqs \
        --rendered "$rendered" \
        --model gpt-4o \
        --backend retrieval \
        --world "$world" \
        --ask_period "$period" \
        --api_key_file keys/openrouter_key.txt \
        --output "$output"
      status=$?
      echo "[$(date '+%F %T')] done world=$world persona=$i period=$period status=$status"
    done
  done
done
