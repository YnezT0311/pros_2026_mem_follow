#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOPICS="${TOPICS:-legalConsultation financialConsultation medicalConsultation}"
MODEL="${MODEL:-gpt-5-mini}"
API_KEY_FILE="${API_KEY_FILE:-keys/openrouter_key.txt}"
QA_FAMILY="${QA_FAMILY:-all}"
TMP_DIR="${TMP_DIR:-tmp}"

cd "${ROOT_DIR}"
mkdir -p "${TMP_DIR}"

echo "=== Block 1: Build baseline worlds ==="
for topic in ${TOPICS}; do
  echo "[baseline] ${topic}"
  python -m memory_control_tests.generation.build_baseline \
    --source_dir "data/output/${topic}" \
    --dest_dir "data/baseline/${topic}" \
    --spec_dest_dir "data/baseline"
done

echo "=== Block 2: Build MCQ specs ==="
STAGE_DIR="$(mktemp -d "${TMP_DIR}/construct_world_and_mcq.XXXXXX")"
cleanup() {
  rm -rf "${STAGE_DIR}"
}
trap cleanup EXIT

for topic in ${TOPICS}; do
  mkdir -p "${STAGE_DIR}/${topic}/specs"
  find "data/baseline/${topic}/specs" -maxdepth 1 -type f -name 'conversation_*.memory_control.json' -print0 \
    | while IFS= read -r -d '' path; do
        cp "${path}" "${STAGE_DIR}/${topic}/specs/"
      done
done

python -m memory_control_tests.generation.build_mcq_specs \
  --source_dir "${STAGE_DIR}" \
  --dest_dir "data/test"

echo "=== Block 3: Render and export MCQs ==="
for topic in ${TOPICS}; do
  echo "[render] ${topic}"
  find "data/baseline/${topic}/specs" -maxdepth 1 -type f -name 'conversation_*.memory_control.json' -print0 \
    | while IFS= read -r -d '' sidecar; do
        python -m memory_control_tests.generation.render_recall_mcqs \
          --sidecar "${sidecar}" \
          --model "${MODEL}" \
          --api_key_file "${API_KEY_FILE}" \
          --qa_family "${QA_FAMILY}"
      done

  echo "[export] ${topic}"
  python -m memory_control_tests.generation.export_test_benchmark \
    --source_dir "data/test/${topic}" \
    --dest_dir "data/test/${topic}"
done

echo "Done."
