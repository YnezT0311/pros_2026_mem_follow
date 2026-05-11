#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

rendered_for_persona() {
  local persona="$1"
  echo "data/test/travelPlanning/specs/conversation_travelPlanning_persona${persona}_sample0.recall_rendered.json"
}

period_tag() {
  case "$1" in
    "Conversation Early Stage") echo "early" ;;
    "Conversation Intermediate Stage") echo "intermediate" ;;
    "Conversation Late Stage") echo "late" ;;
    *) echo "unknown" ;;
  esac
}

expected_plain_output() {
  local rendered="$1" model="$2" world="$3" ask_period="$4" restrict="${5:-}" release="${6:-}"
  local stem out_dir filename
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/travelPlanning/${world}/${model}"
  mkdir -p "$out_dir"
  if [[ "$world" == "no_use" ]]; then
    filename="${stem}.${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      filename="${filename}.release_$(period_tag "$release")"
    fi
    filename="${filename}.test_$(period_tag "$ask_period").recall_eval_${model}.json"
  else
    filename="${stem}.${world}.recall_eval_${model}.json"
    if [[ "$ask_period" != "Conversation Late Stage" ]]; then
      filename="${stem}.${world}.$(period_tag "$ask_period").recall_eval_${model}.json"
    fi
  fi
  echo "${out_dir}/${filename}"
}

expected_mem0_output() {
  local rendered="$1" model="$2" world="$3" ask_period="$4" backend="$5" restrict="${6:-}" release="${7:-}"
  local suffix=""
  if [[ "$world" == "no_use" ]]; then
    suffix=".${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      suffix="${suffix}.release_$(period_tag "$release")"
    fi
    suffix="${suffix}.test_$(period_tag "$ask_period").mem0_${backend}_eval_${model}.json"
  else
    suffix=".${world}.mem0_${backend}_eval_${model}.json"
    if [[ "$ask_period" != "Conversation Late Stage" ]]; then
      suffix=".${world}.$(period_tag "$ask_period").mem0_${backend}_eval_${model}.json"
    fi
  fi
  echo "${rendered/.recall_rendered.json/$suffix}"
}

expected_amem_output() {
  local rendered="$1" model="$2" world="$3" ask_period="$4" restrict="${5:-}" release="${6:-}"
  local suffix=""
  if [[ "$world" == "no_use" ]]; then
    suffix=".${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      suffix="${suffix}.release_$(period_tag "$release")"
    fi
    suffix="${suffix}.test_$(period_tag "$ask_period").a_mem_retrieval_eval_${model}.json"
  else
    suffix=".${world}.a_mem_retrieval_eval_${model}.json"
    if [[ "$ask_period" != "Conversation Late Stage" ]]; then
      suffix=".${world}.$(period_tag "$ask_period").a_mem_retrieval_eval_${model}.json"
    fi
  fi
  echo "${rendered/.recall_rendered.json/$suffix}"
}

expected_langmem_output() {
  local rendered="$1" model="$2" world="$3" ask_period="$4" restrict="${5:-}" release="${6:-}"
  local suffix=""
  if [[ "$world" == "no_use" ]]; then
    suffix=".${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      suffix="${suffix}.release_$(period_tag "$release")"
    fi
    suffix="${suffix}.test_$(period_tag "$ask_period").langmem_retrieval_eval_${model}.json"
  else
    suffix=".${world}.langmem_retrieval_eval_${model}.json"
    if [[ "$ask_period" != "Conversation Late Stage" ]]; then
      suffix=".${world}.$(period_tag "$ask_period").langmem_retrieval_eval_${model}.json"
    fi
  fi
  echo "${rendered/.recall_rendered.json/$suffix}"
}

run_plain() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  local extra_args=("$@")
  local output_path
  local restrict="" release=""
  local i=0
  while [[ $i -lt ${#extra_args[@]} ]]; do
    case "${extra_args[$i]}" in
      --no_use_restrict_period) restrict="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      --no_use_release_period) release="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      *) i=$((i+1)) ;;
    esac
  done
  output_path="$(expected_plain_output "$rendered" "$model" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$output_path" ]]; then
    echo "SKIP plain $model $world $ask_period -> $output_path"
    return
  fi
  echo "RUN plain $model $world $ask_period -> $output_path"
  conda run -n agent python -m memory_control_tests.evaluation.mem_evals \
    --method plain \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    "${extra_args[@]}" \
    --output "$output_path"
}

run_mem0() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  local extra_args=("$@")
  local output_path backend="retrieval" restrict="" release=""
  local i=0
  while [[ $i -lt ${#extra_args[@]} ]]; do
    case "${extra_args[$i]}" in
      --backend) backend="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      --no_use_restrict_period) restrict="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      --no_use_release_period) release="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      *) i=$((i+1)) ;;
    esac
  done
  output_path="$(expected_mem0_output "$rendered" "$model" "$world" "$ask_period" "$backend" "$restrict" "$release")"
  if [[ -f "$output_path" ]]; then
    echo "SKIP mem0 $model $world $ask_period -> $output_path"
    return
  fi
  echo "RUN mem0 $model $world $ask_period -> $output_path"
  conda run -n mem0 python -m memory_control_tests.evaluation.evaluate_mem0_recall_mcqs \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    --backend retrieval \
    "${extra_args[@]}" \
    --output "$output_path"
}

run_amem() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  local extra_args=("$@")
  local output_path restrict="" release=""
  local i=0
  while [[ $i -lt ${#extra_args[@]} ]]; do
    case "${extra_args[$i]}" in
      --no_use_restrict_period) restrict="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      --no_use_release_period) release="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      *) i=$((i+1)) ;;
    esac
  done
  output_path="$(expected_amem_output "$rendered" "$model" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$output_path" ]]; then
    echo "SKIP A-Mem $model $world $ask_period -> $output_path"
    return
  fi
  echo "RUN A-Mem $model $world $ask_period -> $output_path"
  HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  conda run -n amem python -m memory_control_tests.evaluation.evaluate_amem_recall_mcqs \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    "${extra_args[@]}" \
    --output "$output_path"
}

run_langmem() {
  local model="$1" rendered="$2" world="$3" ask_period="$4"
  shift 4
  local extra_args=("$@")
  local output_path restrict="" release=""
  local i=0
  while [[ $i -lt ${#extra_args[@]} ]]; do
    case "${extra_args[$i]}" in
      --no_use_restrict_period) restrict="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      --no_use_release_period) release="${extra_args[$((i+1))]}"; i=$((i+2)) ;;
      *) i=$((i+1)) ;;
    esac
  done
  output_path="$(expected_langmem_output "$rendered" "$model" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$output_path" ]]; then
    echo "SKIP LangMem $model $world $ask_period -> $output_path"
    return
  fi
  echo "RUN LangMem $model $world $ask_period -> $output_path"
  conda run -n langmem311 python -m memory_control_tests.evaluation.evaluate_langmem_recall_mcqs \
    --rendered "$rendered" \
    --model "$model" \
    --world "$world" \
    --ask_period "$ask_period" \
    "${extra_args[@]}" \
    --output "$output_path"
}

annotate_slot_types() {
  local model="$1"
  shift
  for eval_json in "$@"; do
    if [[ -f "$eval_json" ]]; then
      conda run -n agent python -m memory_control_tests.analysis.annotate_slot_types_llm \
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

conda run -n agent python -m memory_control_tests.analysis.summarize_instruction_control_results
