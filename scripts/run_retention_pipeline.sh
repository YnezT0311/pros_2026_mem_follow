#!/usr/bin/env bash
set -euo pipefail

# Full retention test pipeline:
# 1) Generate baseline world data (prepare_data.py)
# 2) Build retention world with local conflict repair
# 3) Generate time-aware retention QA specs
# 4) Evaluate baseline and retention worlds separately

# ----------------------------
# Config (override by env vars)
# ----------------------------
CONDA_ENV="${CONDA_ENV:-agent}"
MODEL="${MODEL:-gpt-4o}"
TOPICS="${TOPICS:-travelPlanning financialConsultation medicalConsultation}"
N_PERSONA="${N_PERSONA:-20}"
N_SAMPLES="${N_SAMPLES:-1}"
S_PERSONA="${S_PERSONA:-0}"
S_SAMPLES="${S_SAMPLES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output/}"
WORKERS="${WORKERS:-12}"
EVAL_WORKERS="${EVAL_WORKERS:-20}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF="${RETRY_BACKOFF:-2}"
API_KEY_FILE="${API_KEY_FILE:-openai_key.txt}"
EVAL_PROVIDER="${EVAL_PROVIDER:-auto}"
TOKEN_PATH="${TOKEN_PATH:-.}"
GEMINI_KEY_FILE="${GEMINI_KEY_FILE:-gemini_key.txt}"
CLAUDE_KEY_FILE="${CLAUDE_KEY_FILE:-claude_key.txt}"
XAI_KEY_FILE="${XAI_KEY_FILE:-xai_key.txt}"
API_BASE_URL="${API_BASE_URL:-}"

# Retention artifacts
RETENTION_WORLD_DIR="${RETENTION_WORLD_DIR:-data/retention/world}"
RETENTION_META="${RETENTION_META:-data/retention/retention_meta.jsonl}"
RETENTION_OPS="${RETENTION_OPS:-data/retention/retention_ops.jsonl}"
RETENTION_SUMMARY="${RETENTION_SUMMARY:-data/retention/retention_summary.json}"
RETENTION_QA_SPECS="${RETENTION_QA_SPECS:-data/retention/retention_qa_specs.jsonl}"
RETENTION_QA_REPORT="${RETENTION_QA_REPORT:-data/retention/retention_qa_specs_report.json}"
RETENTION_EVAL_CSV="${RETENTION_EVAL_CSV:-data/retention/retention_eval_results.csv}"
RETENTION_EVAL_SUMMARY="${RETENTION_EVAL_SUMMARY:-data/retention/retention_eval_summary.json}"
ALLOW_PARTIAL_OUTPUT="${ALLOW_PARTIAL_OUTPUT:-0}"

format_duration() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02dh:%02dm:%02ds" "${h}" "${m}" "${s}"
}

run_step() {
  local label="$1"
  shift
  local start_ts
  local end_ts
  local elapsed
  start_ts="$(date +%s)"
  echo "${label} START $(date '+%F %T')"
  "$@"
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "${label} DONE  $(date '+%F %T') | elapsed=${elapsed}s ($(format_duration "${elapsed}"))"
}

PIPELINE_START_TS="$(date +%s)"

echo "=== Retention Pipeline Config ==="
echo "CONDA_ENV=${CONDA_ENV}"
echo "MODEL=${MODEL}"
echo "TOPICS=${TOPICS}"
echo "N_PERSONA=${N_PERSONA}"
echo "WORKERS=${WORKERS}"
echo "EVAL_WORKERS=${EVAL_WORKERS}"
echo "EVAL_PROVIDER=${EVAL_PROVIDER}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "================================="

run_step "[1/4] Generating/continuing baseline world data..." \
  conda run -n "${CONDA_ENV}" python prepare_data.py \
    --model "${MODEL}" \
    --topics ${TOPICS} \
    --n_persona "${N_PERSONA}" \
    --n_samples "${N_SAMPLES}" \
    --s_persona "${S_PERSONA}" \
    --s_samples "${S_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    --skip_existing \
    --max_retries "${MAX_RETRIES}" \
    --retry_backoff "${RETRY_BACKOFF}"

echo "[check] Validating baseline output completeness..."
CHECK_OUTPUT="$(cd "$(dirname "$0")/.." && python - <<'PY'
import json, os
root = "data/output"
missing = []
for d, _, fs in os.walk(root):
    for f in fs:
        if not (f.startswith("conversation_") and f.endswith(".json")):
            continue
        p = os.path.join(d, f)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception:
            missing.append(p)
            continue
        if "Conversation Next Year" not in obj:
            missing.append(p)
print(len(missing))
if missing:
    print("MISSING_FILES_BEGIN")
    for m in missing[:20]:
        print(m)
    print("MISSING_FILES_END")
PY
)"
MISSING_COUNT="$(echo "${CHECK_OUTPUT}" | head -n 1)"
echo "${CHECK_OUTPUT}" | tail -n +2

if [[ "${MISSING_COUNT}" != "0" ]]; then
  echo "[check] Incomplete output files detected (missing Conversation Next Year): ${MISSING_COUNT}"
  if [[ "${ALLOW_PARTIAL_OUTPUT}" != "1" ]]; then
    echo "[check] Set ALLOW_PARTIAL_OUTPUT=1 to continue anyway."
    exit 1
  fi
fi

run_step "[2/4] Building retention world (with local repair)..." \
  conda run -n "${CONDA_ENV}" python privacy_test/retention/build_retention_world.py \
    --source_dir "${OUTPUT_DIR%/}" \
    --target_dir "${RETENTION_WORLD_DIR}" \
    --meta_path "${RETENTION_META}" \
    --ops_path "${RETENTION_OPS}" \
    --summary_path "${RETENTION_SUMMARY}" \
    --rebuild_target \
    --enable_local_repair \
    --max_repair_rounds 5

run_step "[3/4] Generating retention QA specs..." \
  conda run -n "${CONDA_ENV}" python privacy_test/retention/generate_retention_qa_specs.py \
    --meta_path "${RETENTION_META}" \
    --out_path "${RETENTION_QA_SPECS}" \
    --report_path "${RETENTION_QA_REPORT}"

run_step "[4/4] Evaluating baseline and retention worlds..." \
  conda run -n "${CONDA_ENV}" python privacy_test/retention/evaluate_retention_worlds.py \
    --meta_path "${RETENTION_META}" \
    --spec_path "${RETENTION_QA_SPECS}" \
    --out_csv "${RETENTION_EVAL_CSV}" \
    --summary_path "${RETENTION_EVAL_SUMMARY}" \
    --model "${MODEL}" \
    --provider "${EVAL_PROVIDER}" \
    --token_path "${TOKEN_PATH}" \
    --api_key_file "${API_KEY_FILE}" \
    --gemini_key_file "${GEMINI_KEY_FILE}" \
    --claude_key_file "${CLAUDE_KEY_FILE}" \
    --xai_key_file "${XAI_KEY_FILE}" \
    --api_base_url "${API_BASE_URL}" \
    --world both \
    --workers "${EVAL_WORKERS}"

PIPELINE_END_TS="$(date +%s)"
PIPELINE_ELAPSED=$((PIPELINE_END_TS - PIPELINE_START_TS))

echo "Done."
echo "Summary: ${RETENTION_EVAL_SUMMARY}"
echo "QA option report: ${RETENTION_QA_REPORT}"
echo "Total elapsed: ${PIPELINE_ELAPSED}s ($(format_duration "${PIPELINE_ELAPSED}"))"
