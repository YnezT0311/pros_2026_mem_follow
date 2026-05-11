#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

AGENT_PY="/home/yao/.conda/envs/agent/bin/python"
MEM0_PY="/home/yao/.conda/envs/mem0/bin/python"
AMEM_PY="/home/yao/.conda/envs/amem/bin/python"
LANGMEM_PY="/home/yao/.conda/envs/langmem311/bin/python"
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

plain_output() {
  local rendered="$1" world="$2" ask_period="$3" restrict="${4:-}" release="${5:-}"
  local stem out_dir filename
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/${TOPIC}/${world}/${MODEL}"
  mkdir -p "$out_dir"
  if [[ "$world" == "no_use" ]]; then
    filename="${stem}.${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      filename="${filename}.release_$(period_tag "$release")"
    fi
    filename="${filename}.test_$(period_tag "$ask_period").recall_eval_${MODEL}.json"
    echo "${out_dir}/${filename}"
    return
  fi
  filename="${stem}.${world}.recall_eval_${MODEL}.json"
  if [[ "$ask_period" != "Conversation Late Stage" ]]; then
    filename="${stem}.${world}.$(period_tag "$ask_period").recall_eval_${MODEL}.json"
  fi
  echo "${out_dir}/${filename}"
}

memory_output() {
  local backend="$1" rendered="$2" world="$3" ask_period="$4" restrict="${5:-}" release="${6:-}"
  local stem out_dir filename
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/${TOPIC}/${world}/${MODEL}+${backend}"
  mkdir -p "$out_dir"
  if [[ "$world" == "no_use" ]]; then
    filename="${stem}.${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      filename="${filename}.release_$(period_tag "$release")"
    fi
    case "$backend" in
      mem0) filename="${filename}.test_$(period_tag "$ask_period").mem0_retrieval_eval_${MODEL}.json" ;;
      A-Mem) filename="${filename}.test_$(period_tag "$ask_period").a_mem_retrieval_eval_${MODEL}.json" ;;
      LangMem) filename="${filename}.test_$(period_tag "$ask_period").langmem_retrieval_eval_${MODEL}.json" ;;
    esac
  else
    filename="${stem}.${world}"
    if [[ "$ask_period" != "Conversation Late Stage" ]]; then
      filename="${filename}.$(period_tag "$ask_period")"
    fi
    case "$backend" in
      mem0) filename="${filename}.mem0_retrieval_eval_${MODEL}.json" ;;
      A-Mem) filename="${filename}.a_mem_retrieval_eval_${MODEL}.json" ;;
      LangMem) filename="${filename}.langmem_retrieval_eval_${MODEL}.json" ;;
    esac
  fi
  echo "${out_dir}/${filename}"
}

run_plain_case() {
  local rendered="$1" world="$2" ask_period="$3" restrict="${4:-}" release="${5:-}"
  local out
  local extra_args=()
  out="$(plain_output "$rendered" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$out" ]]; then
    echo "SKIP plain $world $ask_period -> $out"
    return
  fi
  echo "RUN plain $world $ask_period -> $out"
  if [[ -n "$restrict" ]]; then
    extra_args+=(--no_use_restrict_period "$restrict")
  fi
  if [[ -n "$release" ]]; then
    extra_args+=(--no_use_release_period "$release")
  fi
  "$AGENT_PY" -m memory_control_tests.evaluation.mem_evals \
    --method plain \
    --rendered "$rendered" \
    --model "$MODEL" \
    --world "$world" \
    --ask_period "$ask_period" \
    "${extra_args[@]}" \
    --output "$out"
}

run_mem0_case() {
  local rendered="$1" world="$2" ask_period="$3" restrict="${4:-}" release="${5:-}"
  local out
  local extra_args=()
  out="$(memory_output mem0 "$rendered" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$out" ]]; then
    echo "SKIP mem0 $world $ask_period -> $out"
    return
  fi
  echo "RUN mem0 $world $ask_period -> $out"
  if [[ -n "$restrict" ]]; then
    extra_args+=(--no_use_restrict_period "$restrict")
  fi
  if [[ -n "$release" ]]; then
    extra_args+=(--no_use_release_period "$release")
  fi
  "$MEM0_PY" -m memory_control_tests.evaluation.evaluate_mem0_recall_mcqs \
    --rendered "$rendered" \
    --model "$MODEL" \
    --world "$world" \
    --ask_period "$ask_period" \
    "${extra_args[@]}" \
    --output "$out"
}

run_amem_case() {
  local rendered="$1" world="$2" ask_period="$3" restrict="${4:-}" release="${5:-}"
  local out
  local extra_args=()
  out="$(memory_output A-Mem "$rendered" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$out" ]]; then
    echo "SKIP A-Mem $world $ask_period -> $out"
    return
  fi
  echo "RUN A-Mem $world $ask_period -> $out"
  if [[ -n "$restrict" ]]; then
    extra_args+=(--no_use_restrict_period "$restrict")
  fi
  if [[ -n "$release" ]]; then
    extra_args+=(--no_use_release_period "$release")
  fi
  HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$AMEM_PY" -m memory_control_tests.evaluation.evaluate_amem_recall_mcqs \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world "$world" \
      --ask_period "$ask_period" \
      "${extra_args[@]}" \
      --output "$out"
}

run_langmem_case() {
  local rendered="$1" world="$2" ask_period="$3" restrict="${4:-}" release="${5:-}"
  local out
  local extra_args=()
  out="$(memory_output LangMem "$rendered" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$out" ]]; then
    echo "SKIP LangMem $world $ask_period -> $out"
    return
  fi
  echo "RUN LangMem $world $ask_period -> $out"
  if [[ -n "$restrict" ]]; then
    extra_args+=(--no_use_restrict_period "$restrict")
  fi
  if [[ -n "$release" ]]; then
    extra_args+=(--no_use_release_period "$release")
  fi
  "$LANGMEM_PY" -m memory_control_tests.evaluation.evaluate_langmem_recall_mcqs \
    --rendered "$rendered" \
    --model "$MODEL" \
    --world "$world" \
    --ask_period "$ask_period" \
    "${extra_args[@]}" \
    --output "$out"
}

run_no_use_family() {
  local fn="$1" rendered="$2"
  "$fn" "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage"
  "$fn" "$rendered" no_use "Conversation Intermediate Stage" "Conversation Intermediate Stage"
  "$fn" "$rendered" no_use "Conversation Late Stage" "Conversation Late Stage"
  "$fn" "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage"
  "$fn" "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage"
  "$fn" "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage" "Conversation Early Stage"
  "$fn" "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage" "Conversation Early Stage"
  "$fn" "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage" "Conversation Early Stage"
}

for persona in 0 1 2 3; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN gpt-4o other-worlds persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_plain_case "$rendered" baseline "$ask"
    run_plain_case "$rendered" no_store "$ask"
    run_mem0_case "$rendered" baseline "$ask"
    run_mem0_case "$rendered" no_store "$ask"
    run_amem_case "$rendered" baseline "$ask"
    run_amem_case "$rendered" no_store "$ask"
    run_langmem_case "$rendered" baseline "$ask"
    run_langmem_case "$rendered" no_store "$ask"
  done
  run_no_use_family run_plain_case "$rendered"
  run_no_use_family run_mem0_case "$rendered"
  run_no_use_family run_amem_case "$rendered"
  run_no_use_family run_langmem_case "$rendered"
done

for persona in 0 1 2 3 4 5 6 7 8 9; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN gpt-4o forget persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_plain_case "$rendered" forget "$ask"
    run_mem0_case "$rendered" forget "$ask"
    run_amem_case "$rendered" forget "$ask"
    run_langmem_case "$rendered" forget "$ask"
  done
done
