# Memory Instruction Pipeline (Baseline vs Retention World)

Naming note:
- `retention` in legacy code/results corresponds to `no_store`.
- `scope` style controls correspond to `no_use` (future extension).

TODO:
- Implement `direct_explicit_no_store_selective` (span-level selective no-store).
- Add dedicated `no_use` world (scope control).
- Keep parity with deletion pipeline under `privacy_test/deletion/`.

This folder contains the full pipeline for memory-instruction evaluation:

1. Build **baseline world** (`data/output`).
2. Build **retention world** with instruction injection + optional model rewrite.
3. Generate retention QA specs (time-aware MCQ).
4. Evaluate **both worlds** and report **retention** / **utility** scores separately.

## Files

- `build_retention_world.py`: inject instruction into retention world, rewrite future contamination, and optionally run local conflict repair.
- `generate_retention_qa_specs.py`: create QA specs from retention metadata.
- `evaluate_retention_worlds.py`: run model evaluation on baseline/retention worlds.
- `../../scripts/run_retention_pipeline.sh`: one-command end-to-end pipeline.

## Recommended: One-Command Pipeline

From repo root (`PersonaMem-main`):

```bash
bash scripts/run_retention_pipeline.sh
```

You can override config by env vars:

```bash
MODEL=gpt-5-mini \
TOPICS="medicalConsultation financialConsultation travelPlanning therapy legalConsultation" \
N_PERSONA=10 \
WORKERS=10 \
EVAL_WORKERS=20 \
bash scripts/run_retention_pipeline.sh
```

The script does all 4 stages and writes:

- baseline data: `data/output/...`
- retention world: `data/retention/world/...`
- metadata: `data/retention/retention_meta.jsonl`
- operations: `data/retention/retention_ops.jsonl`
- QA specs: `data/retention/retention_qa_specs.jsonl`
- eval rows: `data/retention/retention_eval_results.csv`
- eval summary: `data/retention/retention_eval_summary.json`

## Manual 4-Step Workflow

### 1) Build baseline world

```bash
conda run --no-capture-output -n agent python -u prepare_data.py \
  --model gpt-5-mini \
  --topics medicalConsultation financialConsultation travelPlanning therapy legalConsultation \
  --n_persona 10 \
  --n_samples 1 \
  --s_persona 0 \
  --s_samples 0 \
  --workers 10 \
  --output_dir data/output/ \
  --skip_existing
```

### 2) Build retention world

Recommended quality-first build:

```bash
conda run --no-capture-output -n agent python -u privacy_test/retention/build_retention_world.py \
  --source_dir data/output \
  --target_dir data/retention/world \
  --meta_path data/retention/retention_meta.jsonl \
  --ops_path data/retention/retention_ops.jsonl \
  --summary_path data/retention/retention_summary.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --instruction_type direct_explicit_no_store_all \
  --enable_model_rewrite \
  --rebuild_target
```

Incremental rebuild for new baseline files only:

```bash
conda run --no-capture-output -n agent python -u privacy_test/retention/build_retention_world.py \
  --source_dir data/output \
  --target_dir data/retention/world \
  --meta_path data/retention/retention_meta.jsonl \
  --ops_path data/retention/retention_ops.jsonl \
  --summary_path data/retention/retention_summary.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --instruction_type direct_explicit_no_store_all \
  --enable_model_rewrite \
  --skip_existing
```

Notes:
- `--enable_model_rewrite` is the recommended high-quality path.
- `--enable_local_repair` is a lightweight fallback and cannot be combined with `--enable_model_rewrite`.
- `--skip_existing` only skips files whose `world + meta + ops` are already complete.

`instruction_type` options:

- `direct_explicit_no_store_all` (implemented, recommended now)
- `direct_explicit_no_store_selective` (reserved for future, not implemented)

### 3) Generate retention QA specs

```bash
conda run --no-capture-output -n agent python -u privacy_test/retention/generate_retention_qa_specs.py \
  --meta_path data/retention/retention_meta.jsonl \
  --out_path data/retention/retention_qa_specs.jsonl \
  --report_path data/retention/retention_qa_specs_report.json
```

### 4) Evaluate both worlds

```bash
conda run --no-capture-output -n agent python -u privacy_test/retention/evaluate_retention_worlds.py \
  --meta_path data/retention/retention_meta.jsonl \
  --spec_path data/retention/retention_qa_specs.jsonl \
  --out_csv data/retention/retention_eval_results.csv \
  --summary_path data/retention/retention_eval_summary.json \
  --model gpt-5.1 \
  --provider openai \
  --api_key_file openai_key.txt \
  --world both \
  --workers 8 \
  --max_questions 200 \
  --sampling stratified \
  --sample_seed 0 \
  --sample_manifest_out data/retention/sample_manifest_200.json
```

To reuse exactly the same sampled questions for another model:

```bash
conda run --no-capture-output -n agent python -u privacy_test/retention/evaluate_retention_worlds.py \
  --meta_path data/retention/retention_meta.jsonl \
  --spec_path data/retention/retention_qa_specs.jsonl \
  --out_csv data/retention/eval_other_model.csv \
  --summary_path data/retention/eval_other_model_summary.json \
  --model gpt-5.1 \
  --provider openai \
  --world both \
  --workers 8 \
  --sample_manifest_in data/retention/sample_manifest_200.json
```

## Score Interpretation (Separated)

`retention_eval_summary.json` now uses the unified metric schema:

- `constraint_score`
- `leakage`
- `clean_utility`
- `robust_utility`
- `clean_utility_drop`
- `robust_utility_drop`

Interpretation:

- `constraint_score`: how well the model suppresses restricted memory.
- `leakage`: `1 - constraint_score`, i.e. restricted-memory leakage.
- `clean_utility`: allowed-memory use on standard allowed-information questions.
- `robust_utility`: allowed-memory use when answer choices include misleading refusal-style distractors.
- `clean_utility_drop`: drop in `clean_utility` from baseline to retention world.
- `robust_utility_drop`: drop in `robust_utility` from baseline to retention world.

Interpretation:

- **Retention score**: correctness on `must_not_recall` questions.
- **Utility score**: correctness on allowed-memory questions.

Typical target behavior:

- Baseline: higher retention recall (no prohibition), good utility.
- Retention world: retention recall should drop (respect no-store), utility should remain as high as possible.

## Notes

- Baseline files are not modified by retention builder.
- Model rewrite is the preferred contamination-cleanup path for retention.
- Local repair is conservative and only patches conflicted assistant text spans.
- If baseline generation is partial, pipeline can stop early unless `ALLOW_PARTIAL_OUTPUT=1`.
