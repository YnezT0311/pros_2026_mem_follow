#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

rendered_for_persona() {
  local persona="$1"
  echo "data/test/travelPlanning/specs/conversation_travelPlanning_persona${persona}_sample0.recall_rendered.json"
}

run_plain() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  conda run -n agent python -m memory_control_tests.evaluation.evaluate_recall_mcqs \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    "$@"
}

run_mem0() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  conda run -n mem0 python -m memory_control_tests.evaluation.evaluate_mem0_recall_mcqs \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    --backend retrieval \
    "$@"
}

run_amem() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  conda run -n amem python -m memory_control_tests.evaluation.evaluate_amem_recall_mcqs \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    "$@"
}

run_langmem() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  conda run -n langmem311 python -m memory_control_tests.evaluation.evaluate_langmem_recall_mcqs \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    "$@"
}

annotate_slot_types() {
  local model="$1"
  shift
  for eval_json in "$@"; do
    if [[ -f "$eval_json" ]]; then
      conda run -n agent python -m memory_control_tests.evaluation.annotate_slot_types_llm \
        --input "$eval_json" \
        --model "$model"
    fi
  done
}

run_no_use_family() {
  local runner="$1" model="$2" rendered="$3"
  "$runner" "$model" "$rendered" no_use "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage"
  "$runner" "$model" "$rendered" no_use "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Intermediate Stage"
  "$runner" "$model" "$rendered" no_use "Conversation Late Stage" --no_use_restrict_period "Conversation Late Stage"
  "$runner" "$model" "$rendered" no_use "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage"
  "$runner" "$model" "$rendered" no_use "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage"
  "$runner" "$model" "$rendered" no_use "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage"
  "$runner" "$model" "$rendered" no_use "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage"
  "$runner" "$model" "$rendered" no_use "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage"
}

run_stage_family() {
  local runner="$1" model="$2" rendered="$3" world="$4"
  "$runner" "$model" "$rendered" "$world" "Conversation Early Stage"
  "$runner" "$model" "$rendered" "$world" "Conversation Intermediate Stage"
  "$runner" "$model" "$rendered" "$world" "Conversation Late Stage"
}

for model in gpt-5.4-mini gpt-4o; do
  for persona in 0 1 2 3; do
    rendered="$(rendered_for_persona "$persona")"
    run_stage_family run_plain "$model" "$rendered" baseline
    run_stage_family run_plain "$model" "$rendered" no_store
    run_no_use_family run_plain "$model" "$rendered"
  done

  for persona in 0 1 2 3 4 5 6 7 8 9; do
    rendered="$(rendered_for_persona "$persona")"
    run_stage_family run_plain "$model" "$rendered" forget
  done

  for persona in 0 1 2 3; do
    rendered="$(rendered_for_persona "$persona")"
    run_stage_family run_mem0 "$model" "$rendered" baseline
    run_stage_family run_mem0 "$model" "$rendered" no_store
    run_no_use_family run_mem0 "$model" "$rendered"

    run_stage_family run_amem "$model" "$rendered" baseline
    run_stage_family run_amem "$model" "$rendered" no_store
    run_no_use_family run_amem "$model" "$rendered"

    run_stage_family run_langmem "$model" "$rendered" baseline
    run_stage_family run_langmem "$model" "$rendered" no_store
    run_no_use_family run_langmem "$model" "$rendered"
  done

  for persona in 0 1 2 3 4 5 6 7 8 9; do
    rendered="$(rendered_for_persona "$persona")"
    run_stage_family run_mem0 "$model" "$rendered" forget
    run_stage_family run_amem "$model" "$rendered" forget
    run_stage_family run_langmem "$model" "$rendered" forget
  done
done

slot_eval_paths=()
while IFS= read -r path; do
  slot_eval_paths+=("$path")
done < <(find data/test/travelPlanning/specs eval_results/travelPlanning -type f -name "*.json" \
  | grep -E "(recall_eval|mem0_retrieval_eval|a_mem_retrieval_eval|langmem_retrieval_eval)")

annotate_slot_types gpt-5-mini "${slot_eval_paths[@]}"

conda run -n agent python -m memory_control_tests.evaluation.summarize_instruction_control_results
