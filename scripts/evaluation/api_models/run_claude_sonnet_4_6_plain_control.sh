#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="anthropic/claude-sonnet-4.6"
MODEL_TAG="anthropic_claude-sonnet-4.6"
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

raw_output_path() {
  local rendered="$1" world="$2" ask_period="$3"
  local stem out_dir filename
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/${TOPIC}/${world}/${MODEL_TAG}"
  mkdir -p "$out_dir"
  filename="${stem}.${world}.raw_eval_${MODEL_TAG}.json"
  if [[ "$ask_period" != "Conversation Late Stage" ]]; then
    filename="${stem}.${world}.$(period_tag "$ask_period").raw_eval_${MODEL_TAG}.json"
  fi
  echo "${out_dir}/${filename}"
}

scored_output_path() {
  local raw_path="$1"
  echo "${raw_path%.json}.scored.json"
}

run_plain_case() {
  local rendered="$1" world="$2" ask_period="$3"
  local raw_out scored_out
  raw_out="$(raw_output_path "$rendered" "$world" "$ask_period")"
  scored_out="$(scored_output_path "$raw_out")"

  if [[ -f "$scored_out" ]]; then
    echo "SKIP scored $world $ask_period -> $scored_out"
    return
  fi

  if [[ ! -f "$raw_out" ]]; then
    echo "RUN raw $world $ask_period -> $raw_out"
    "$PYTHON_BIN" -m memory_control_tests.evaluation.mem_evals \
      --method plain \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world "$world" \
      --ask_period "$ask_period" \
      --api_key_file "keys/openrouter_key.txt" \
      --output "$raw_out"
  else
    echo "SKIP raw $world $ask_period -> $raw_out"
  fi

  echo "RUN score $world $ask_period -> $scored_out"
  "$PYTHON_BIN" -m memory_control_tests.evaluation.scores \
    --input "$raw_out" \
    --output "$scored_out"
}

for persona in 0 1 2 3; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN Claude Sonnet 4.6 plain no_store persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_plain_case "$rendered" no_store "$ask"
  done
done

for persona in 0 1 2 3 4 5 6 7 8 9; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN Claude Sonnet 4.6 plain forget persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_plain_case "$rendered" forget "$ask"
  done
done
