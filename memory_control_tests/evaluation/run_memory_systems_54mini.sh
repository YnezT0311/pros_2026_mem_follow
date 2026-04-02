#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

period_tag() {
  case "$1" in
    "Conversation Early Stage") echo "early" ;;
    "Conversation Intermediate Stage") echo "intermediate" ;;
    "Conversation Late Stage") echo "late" ;;
    *) echo "unknown" ;;
  esac
}

memory_output_path() {
  local backend="$1" rendered="$2" world="$3" ask_period="$4" restrict="${5:-}" release="${6:-}"
  local stem filename topic out_dir model_suffix="gpt-5.4-mini"
  stem="$(basename "$rendered" .recall_rendered.json)"
  topic="travelPlanning"
  out_dir="eval_results/${topic}/${world}/gpt-5.4-mini+${backend}"
  mkdir -p "$out_dir"
  if [[ "$world" == "no_use" ]]; then
    filename="${stem}.${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      filename="${filename}.release_$(period_tag "$release")"
    fi
    case "$backend" in
      mem0) filename="${filename}.test_$(period_tag "$ask_period").mem0_retrieval_eval_${model_suffix}.json" ;;
      A-Mem) filename="${filename}.test_$(period_tag "$ask_period").a_mem_retrieval_eval_${model_suffix}.json" ;;
      LangMem) filename="${filename}.test_$(period_tag "$ask_period").langmem_retrieval_eval_${model_suffix}.json" ;;
    esac
  else
    filename="${stem}.${world}"
    if [[ "$ask_period" != "Conversation Late Stage" ]]; then
      filename="${filename}.$(period_tag "$ask_period")"
    fi
    case "$backend" in
      mem0) filename="${filename}.mem0_retrieval_eval_${model_suffix}.json" ;;
      A-Mem) filename="${filename}.a_mem_retrieval_eval_${model_suffix}.json" ;;
      LangMem) filename="${filename}.langmem_retrieval_eval_${model_suffix}.json" ;;
    esac
  fi
  echo "${out_dir}/${filename}"
}

run_mem0_case() {
  /home/yao/.conda/envs/mem0/bin/python -m memory_control_tests.evaluation.evaluate_mem0_recall_mcqs "$@"
}

run_amem_case() {
  HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    /home/yao/.conda/envs/amem/bin/python -m memory_control_tests.evaluation.evaluate_amem_recall_mcqs "$@"
}

run_langmem_case() {
  /home/yao/.conda/envs/langmem311/bin/python -m memory_control_tests.evaluation.evaluate_langmem_recall_mcqs "$@"
}

run_other_worlds_for_persona() {
  local persona="$1"
  local rendered="data/test/travelPlanning/specs/conversation_travelPlanning_persona${persona}_sample0.recall_rendered.json"

  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world baseline --ask_period "$ask" --output "$(memory_output_path mem0 "$rendered" baseline "$ask")"
    run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_store --ask_period "$ask" --output "$(memory_output_path mem0 "$rendered" no_store "$ask")"
    run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world baseline --ask_period "$ask" --output "$(memory_output_path A-Mem "$rendered" baseline "$ask")"
    run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_store --ask_period "$ask" --output "$(memory_output_path A-Mem "$rendered" no_store "$ask")"
    run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world baseline --ask_period "$ask" --output "$(memory_output_path LangMem "$rendered" baseline "$ask")"
    run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_store --ask_period "$ask" --output "$(memory_output_path LangMem "$rendered" no_store "$ask")"
  done

  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage")"
  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Intermediate Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Intermediate Stage" "Conversation Intermediate Stage")"
  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Late Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Late Stage" "Conversation Late Stage")"
  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage")"
  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage")"
  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage" "Conversation Early Stage")"
  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage" "Conversation Early Stage")"
  run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path mem0 "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage" "Conversation Early Stage")"

  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage")"
  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Intermediate Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Intermediate Stage" "Conversation Intermediate Stage")"
  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Late Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Late Stage" "Conversation Late Stage")"
  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage")"
  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage")"
  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage" "Conversation Early Stage")"
  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage" "Conversation Early Stage")"
  run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path A-Mem "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage" "Conversation Early Stage")"

  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage")"
  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Intermediate Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Intermediate Stage" "Conversation Intermediate Stage")"
  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Late Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Late Stage" "Conversation Late Stage")"
  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage")"
  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage")"
  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Early Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage" "Conversation Early Stage")"
  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Intermediate Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage" "Conversation Early Stage")"
  run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world no_use --ask_period "Conversation Late Stage" --no_use_restrict_period "Conversation Early Stage" --no_use_release_period "Conversation Early Stage" --output "$(memory_output_path LangMem "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage" "Conversation Early Stage")"
}

run_forget_for_persona() {
  local persona="$1"
  local rendered="data/test/travelPlanning/specs/conversation_travelPlanning_persona${persona}_sample0.recall_rendered.json"

  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_mem0_case --rendered "$rendered" --model gpt-5.4-mini --world forget --ask_period "$ask" --output "$(memory_output_path mem0 "$rendered" forget "$ask")"
    run_amem_case --rendered "$rendered" --model gpt-5.4-mini --world forget --ask_period "$ask" --output "$(memory_output_path A-Mem "$rendered" forget "$ask")"
    run_langmem_case --rendered "$rendered" --model gpt-5.4-mini --world forget --ask_period "$ask" --output "$(memory_output_path LangMem "$rendered" forget "$ask")"
  done
}

for persona in 0 1 2 3; do
  echo "RUN other-worlds persona${persona}"
  run_other_worlds_for_persona "$persona"
done

for persona in 0 1 2 3 4 5 6 7 8 9; do
  echo "RUN forget persona${persona}"
  run_forget_for_persona "$persona"
done
