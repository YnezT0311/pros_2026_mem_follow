#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Reuse the amem env — it already has sentence-transformers + openai, and we
# `pip install`-ed faiss-cpu / pymilvus into it for the new adapters.
MEMORYOS_PY="${MEMORYOS_PY:-python}"
MODEL="gpt-5.4-mini"
TOPIC="travelPlanning"
SHORT_TERM_CAPACITY=20  # validated on persona0 baseline; whole 1.00 / slot 0.71
METHOD_CONFIG="tmp/method_configs/memoryos_short_${SHORT_TERM_CAPACITY}.json"
mkdir -p "$(dirname "$METHOD_CONFIG")"
printf '{"memoryos_short_term_capacity": %s}\n' "$SHORT_TERM_CAPACITY" >"$METHOD_CONFIG"

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
  local rendered="$1" world="$2" ask_period="$3" restrict="${4:-}" release="${5:-}"
  local stem out_dir filename
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/${TOPIC}/${world}/${MODEL}+MemoryOS"
  mkdir -p "$out_dir"
  if [[ "$world" == "no_use" ]]; then
    filename="${stem}.${world}.restrict_$(period_tag "$restrict")"
    if [[ -n "$release" ]]; then
      filename="${filename}.release_$(period_tag "$release")"
    fi
    filename="${filename}.test_$(period_tag "$ask_period").memoryos_retrieval_eval_${MODEL}.json"
  else
    filename="${stem}.${world}"
    if [[ "$ask_period" != "Conversation Late Stage" ]]; then
      filename="${filename}.$(period_tag "$ask_period")"
    fi
    filename="${filename}.memoryos_retrieval_eval_${MODEL}.json"
  fi
  echo "${out_dir}/${filename}"
}

run_memoryos_case() {
  local rendered="$1" world="$2" ask_period="$3" restrict="${4:-}" release="${5:-}"
  local out
  local extra_args=()
  out="$(memory_output "$rendered" "$world" "$ask_period" "$restrict" "$release")"
  if [[ -f "$out" ]]; then
    echo "SKIP MemoryOS $world $ask_period -> $out"
    return
  fi
  echo "RUN MemoryOS $world $ask_period -> $out"
  if [[ -n "$restrict" ]]; then
    extra_args+=(--no_use_restrict_period "$restrict")
  fi
  if [[ -n "$release" ]]; then
    extra_args+=(--no_use_release_period "$release")
  fi
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$MEMORYOS_PY" -m memory_control_tests.evaluation.mem_evals \
      --method memoryos \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world "$world" \
      --ask_period "$ask_period" \
      --method_config "$METHOD_CONFIG" \
      "${extra_args[@]}" \
      --output "$out"
}

run_no_use_family() {
  local rendered="$1"
  run_memoryos_case "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage"
  run_memoryos_case "$rendered" no_use "Conversation Intermediate Stage" "Conversation Intermediate Stage"
  run_memoryos_case "$rendered" no_use "Conversation Late Stage" "Conversation Late Stage"
  run_memoryos_case "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage"
  run_memoryos_case "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage"
  run_memoryos_case "$rendered" no_use "Conversation Early Stage" "Conversation Early Stage" "Conversation Early Stage"
  run_memoryos_case "$rendered" no_use "Conversation Intermediate Stage" "Conversation Early Stage" "Conversation Early Stage"
  run_memoryos_case "$rendered" no_use "Conversation Late Stage" "Conversation Early Stage" "Conversation Early Stage"
}

for persona in 0 1 2 3; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN gpt-5.4-mini MemoryOS other-worlds persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_memoryos_case "$rendered" baseline "$ask"
    run_memoryos_case "$rendered" no_store "$ask"
  done
  run_no_use_family "$rendered"
done

for persona in 0 1 2 3 4 5 6 7 8 9; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN gpt-5.4-mini MemoryOS forget persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_memoryos_case "$rendered" forget "$ask"
  done
done
