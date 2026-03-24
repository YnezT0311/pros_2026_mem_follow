# Deletion World Pipeline (All-Turn First)

This folder implements a post-hoc deletion experiment:
- user reveals a detail in one turn (`reveal`)
- later user requests deletion/forget (`delete`)
- evaluation asks after delete and measures deletion-vs-utility behavior

Current scope:
- `deletion_all` only (whole-detail forget request)
- includes timing labels:
  - `gap_reveal_delete`
  - `gap_delete_ask`
  - `gap_reveal_ask`

## 1) Build deletion world

```bash
conda run --no-capture-output -n agent python -u privacy_test/deletion/build_deletion_world.py \
  --source_dir data/output \
  --target_dir data/deletion/world \
  --meta_path data/deletion/deletion_meta.jsonl \
  --ops_path data/deletion/deletion_ops.jsonl \
  --summary_path data/deletion/deletion_summary.json \
  --enable_local_repair \
  --max_repair_rounds 3 \
  --rebuild_target
```

Recommended full build command with model-based completion, contamination rewrite, and final conflict repair:

```bash
conda run --no-capture-output -n agent python -u privacy_test/deletion/build_deletion_world.py \
  --source_dir data/output \
  --target_dir data/deletion/world \
  --meta_path data/deletion/deletion_meta.jsonl \
  --ops_path data/deletion/deletion_ops.jsonl \
  --summary_path data/deletion/deletion_summary.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --enable_model_completion \
  --completion_use_templates \
  --completion_template_file privacy_test/deletion/completion_templates.json \
  --enable_model_rewrite \
  --enable_model_repair \
  --workers 10 \
  --rebuild_target
```

Incremental rebuild for only newly added baseline files:

```bash
conda run --no-capture-output -n agent python -u privacy_test/deletion/build_deletion_world.py \
  --source_dir data/output \
  --target_dir data/deletion/world \
  --meta_path data/deletion/deletion_meta.jsonl \
  --ops_path data/deletion/deletion_ops.jsonl \
  --summary_path data/deletion/deletion_summary.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --enable_model_completion \
  --completion_use_templates \
  --completion_template_file privacy_test/deletion/completion_templates.json \
  --enable_model_rewrite \
  --enable_model_repair \
  --workers 10 \
  --skip_existing
```

Why these flags are paired:
- `--enable_model_completion`: generate the deletion acknowledgement turn cleanly
- `--enable_model_rewrite`: rewrite contaminated future history / user / assistant turns using the updated context
- `--enable_model_repair`: run a final assistant-side repair loop on any residual conflicts left after rewrite
- `--enable_model_repair` and `--enable_local_repair` are mutually exclusive
- `--skip_existing` only skips files whose `world + meta + ops` are already complete

## 2) Generate deletion QA specs

```bash
conda run --no-capture-output -n agent python -u privacy_test/deletion/generate_deletion_qa_specs.py \
  --meta_path data/deletion/deletion_meta.jsonl \
  --out_path data/deletion/deletion_qa_specs.jsonl \
  --report_path data/deletion/deletion_qa_specs_report.json
```

## 3) Evaluate deletion worlds

```bash
conda run --no-capture-output -n agent python -u privacy_test/deletion/evaluate_deletion_worlds.py \
  --meta_path data/deletion/deletion_meta.jsonl \
  --spec_path data/deletion/deletion_qa_specs.jsonl \
  --out_csv data/deletion/deletion_eval_results.csv \
  --summary_path data/deletion/deletion_eval_summary.json \
  --model gpt-5.1 \
  --provider openai \
  --world both \
  --workers 8 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --max_questions 200 \
  --sampling stratified \
  --sample_seed 0 \
  --sample_manifest_out data/deletion/sample_manifest_200.json
```

To reuse the same sampled questions for another model:

```bash
conda run --no-capture-output -n agent python -u privacy_test/deletion/evaluate_deletion_worlds.py \
  --meta_path data/deletion/deletion_meta.jsonl \
  --spec_path data/deletion/deletion_qa_specs.jsonl \
  --out_csv data/deletion/deletion_eval_results.csv \
  --summary_path data/deletion/deletion_eval_summary.json \
  --model gpt-5.1 \
  --provider openai \
  --world both \
  --workers 8 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --sample_manifest_in data/deletion/sample_manifest_200.json
```

## Notes

- In this first version, asks are generated only at/after delete period.
- Utility is taken from non-injected turns in the same conversation file.
- For cross-model fairness, use sample manifest (`--sample_manifest_out` then `--sample_manifest_in`).
- Summary includes conflict stats: `num_conflicts_before_repair`, `num_repairs`, `num_conflicts_after_repair`.
- Evaluation summary uses the unified metric schema:
  - `constraint_score`
  - `leakage`
  - `clean_utility`
  - `robust_utility`
  - `clean_utility_drop`
  - `robust_utility_drop`
- Optional unified entrypoint:
  - `python privacy_test/evaluate_worlds.py --policy deletion ...`
