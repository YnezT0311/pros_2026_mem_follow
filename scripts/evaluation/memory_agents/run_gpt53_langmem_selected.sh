#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

LANGMEM_PY="/home/yao/.conda/envs/langmem311/bin/python"
MODEL="${MODEL-gpt-5.3-chat}"
TOPIC="travelPlanning"
WORLD="${1:-}"
OVERWRITE="${OVERWRITE-0}"

if [[ -z "$WORLD" ]]; then
  echo "Usage: $0 <baseline|no_store|forget>" >&2
  exit 1
fi

case "$WORLD" in
  baseline|no_store) PERSONAS="${OTHER_PERSONAS-0 1 2 3}" ;;
  forget) PERSONAS="${FORGET_PERSONAS-0 1 2 3 4 5 6 7 8 9}" ;;
  *)
    echo "Unsupported world: $WORLD" >&2
    exit 1
    ;;
esac

model_tag() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

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
  local rendered="$1" world="$2" ask_period="$3"
  local stem out_dir filename model_safe
  stem="$(basename "$rendered" .recall_rendered.json)"
  model_safe="$(model_tag "$MODEL")"
  out_dir="eval_results/${TOPIC}/${world}/${model_safe}+LangMem"
  mkdir -p "$out_dir"
  filename="${stem}.${world}"
  if [[ "$ask_period" != "Conversation Late Stage" ]]; then
    filename="${filename}.$(period_tag "$ask_period")"
  fi
  filename="${filename}.langmem_retrieval_eval_${model_safe}.json"
  echo "${out_dir}/${filename}"
}

run_langmem_case() {
  local rendered="$1" world="$2" ask_period="$3"
  local out
  out="$(memory_output "$rendered" "$world" "$ask_period")"
  if [[ "$OVERWRITE" != "1" && -f "$out" ]]; then
    echo "SKIP LangMem $MODEL $world $ask_period -> $out"
    return
  fi
  echo "RUN LangMem $MODEL $world $ask_period -> $out"
  "$LANGMEM_PY" -m memory_control_tests.evaluation.evaluate_langmem_recall_mcqs \
    --rendered "$rendered" \
    --model "$MODEL" \
    --world "$world" \
    --ask_period "$ask_period" \
    --output "$out"
}

for persona in $PERSONAS; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN ${MODEL} LangMem ${WORLD} persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_langmem_case "$rendered" "$WORLD" "$ask"
  done
done

echo "DONE ${MODEL} LangMem ${WORLD}"
