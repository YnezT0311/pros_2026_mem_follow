#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <backend: mem0|A-Mem|LangMem> <model> <persona...>" >&2
  exit 1
fi

BACKEND="$1"
MODEL="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

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

backend_python() {
  case "$BACKEND" in
    mem0) echo "/home/yao/.conda/envs/mem0/bin/python" ;;
    A-Mem) echo "/home/yao/.conda/envs/amem/bin/python" ;;
    LangMem) echo "/home/yao/.conda/envs/langmem311/bin/python" ;;
    *) echo "Unsupported backend: $BACKEND" >&2; exit 1 ;;
  esac
}

backend_module() {
  case "$BACKEND" in
    mem0) echo "memory_control_tests.evaluation.evaluate_mem0_recall_mcqs" ;;
    A-Mem) echo "memory_control_tests.evaluation.evaluate_amem_recall_mcqs" ;;
    LangMem) echo "memory_control_tests.evaluation.evaluate_langmem_recall_mcqs" ;;
  esac
}

memory_output() {
  local rendered="$1" ask_period="$2"
  local stem out_dir suffix filename
  stem="$(basename "$rendered" .recall_rendered.json)"
  out_dir="eval_results/${TOPIC}/forget/${MODEL}+${BACKEND}"
  mkdir -p "$out_dir"
  case "$BACKEND" in
    mem0) suffix="mem0_retrieval_eval_${MODEL}.json" ;;
    A-Mem) suffix="a_mem_retrieval_eval_${MODEL}.json" ;;
    LangMem) suffix="langmem_retrieval_eval_${MODEL}.json" ;;
  esac
  filename="${stem}.forget"
  if [[ "$ask_period" != "Conversation Late Stage" ]]; then
    filename="${filename}.$(period_tag "$ask_period")"
  fi
  filename="${filename}.${suffix}"
  echo "${out_dir}/${filename}"
}

run_case() {
  local rendered="$1" ask_period="$2"
  local out py mod
  out="$(memory_output "$rendered" "$ask_period")"
  if [[ -f "$out" ]]; then
    echo "SKIP ${BACKEND} ${MODEL} forget $ask_period -> $out"
    return
  fi
  echo "RUN ${BACKEND} ${MODEL} forget $ask_period -> $out"
  py="$(backend_python)"
  mod="$(backend_module)"
  if [[ "$BACKEND" == "A-Mem" ]]; then
    HF_HOME=/home/yao/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      "$py" -m "$mod" \
        --rendered "$rendered" \
        --model "$MODEL" \
        --world forget \
        --ask_period "$ask_period" \
        --output "$out"
  else
    "$py" -m "$mod" \
      --rendered "$rendered" \
      --model "$MODEL" \
      --world forget \
      --ask_period "$ask_period" \
      --output "$out"
  fi
}

for persona in "$@"; do
  rendered="$(rendered_for_persona "$persona")"
  echo "RUN ${BACKEND} ${MODEL} forget persona${persona}"
  for ask in "Conversation Early Stage" "Conversation Intermediate Stage" "Conversation Late Stage"; do
    run_case "$rendered" "$ask"
  done
done
