# No-Use World Pipeline

Two modes:
- `scope`: only `no_use_on`
- `temporal_scope`: `no_use_on` then later `no_use_off`

## 1) Build world (run separately)

### 1.1 Scope world (`no_use_on` only)

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/build_no_use_world.py \
  --source_dir data/output \
  --target_dir data/no_use/world_scope \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --ops_path data/no_use/no_use_ops_scope.jsonl \
  --summary_path data/no_use/no_use_summary_scope.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --completion_use_templates \
  --completion_template_count 10 \
  --completion_template_cache_path data/no_use/completion_templates.json \
  --mode scope \
  --rebuild_target
```

### 1.2 Temporal Scope world (`no_use_on` then `no_use_off`)

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/build_no_use_world.py \
  --source_dir data/output \
  --target_dir data/no_use/world_temporal_scope \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --ops_path data/no_use/no_use_ops_temporal_scope.jsonl \
  --summary_path data/no_use/no_use_summary_temporal_scope.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --completion_use_templates \
  --completion_template_count 10 \
  --completion_template_cache_path data/no_use/completion_templates.json \
  --mode temporal_scope \
  --rebuild_target
```

Incremental rebuild examples:

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/build_no_use_world.py \
  --source_dir data/output \
  --target_dir data/no_use/world_scope \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --ops_path data/no_use/no_use_ops_scope.jsonl \
  --summary_path data/no_use/no_use_summary_scope.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --completion_use_templates \
  --completion_template_count 10 \
  --completion_template_cache_path data/no_use/completion_templates.json \
  --mode scope \
  --skip_existing
```

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/build_no_use_world.py \
  --source_dir data/output \
  --target_dir data/no_use/world_temporal_scope \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --ops_path data/no_use/no_use_ops_temporal_scope.jsonl \
  --summary_path data/no_use/no_use_summary_temporal_scope.json \
  --model gpt-5-mini \
  --api_key_file openai_key.txt \
  --completion_use_templates \
  --completion_template_count 10 \
  --completion_template_cache_path data/no_use/completion_templates.json \
  --mode temporal_scope \
  --skip_existing
```

`--skip_existing` only skips files whose `world + meta + ops` are already complete for the requested mode.

Default behavior with `--completion_use_templates`:
- if cache exists, reuse it;
- if cache is missing, auto-generate once and save.

If you want to force regenerate templates and overwrite cache:

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/build_no_use_world.py \
  ... \
  --completion_use_templates \
  --completion_templates_from_model
```

## 2) Generate QA specs

Current QA design:
- `direct_cue_blocked`
  - old direct-cue suppression probe
  - baseline world should pick the forbidden-memory content
  - `no_use` world should pick the direct-cue safe answer instead
- `reasoning_alternative`
  - asks how to respond to a rephrased version of the target turn's problem
  - baseline world should pick the memory-using solution
  - `no_use` world should pick a safe alternative solution
- `reasoning_recovery_use`
  - used in `temporal_scope`
  - asks the same use-time question after `off`
  - both baseline and `no_use` worlds should now pick the memory-using solution again
- `reasoning_insufficient`
  - reserved as a future placeholder
  - not generated or evaluated in the current pipeline
- utility questions remain:
  - `utility_recall`
  - `utility_policy_pressure`

### 2.1 Generate `direct_cue` QA only

`scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/generate_no_use_qa_specs.py \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --out_path data/no_use/no_use_qa_specs_scope_direct_cue.jsonl \
  --report_path data/no_use/no_use_qa_specs_report_scope_direct_cue.json \
  --generator rule_based \
  --qa_profile direct_cue
```

`temporal_scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/generate_no_use_qa_specs.py \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --out_path data/no_use/no_use_qa_specs_temporal_scope_direct_cue.jsonl \
  --report_path data/no_use/no_use_qa_specs_report_temporal_scope_direct_cue.json \
  --generator rule_based \
  --qa_profile direct_cue
```

### 2.2 Generate `reasoning` QA only

If you want to keep the existing reasoning files, do not rerun this step.

Current reasoning evaluation scope:
- only `reasoning_alternative` and `reasoning_recovery_use` are used
- `reasoning_insufficient` is intentionally excluded from the current pipeline
- if older reasoning spec files still contain `blocked_use_insufficient` / `reasoning_insufficient`, remove those rows before evaluation

`scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/generate_no_use_qa_specs.py \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --out_path data/no_use/no_use_qa_specs_scope.jsonl \
  --report_path data/no_use/no_use_qa_specs_report_scope.json \
  --generator llm_reasoning \
  --qa_profile reasoning \
  --llm_model gpt-5-mini \
  --llm_reasoning_count 50 \
  --num_workers 10 \
  --token_path . \
  --api_key_file openai_key.txt
```

`temporal_scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/generate_no_use_qa_specs.py \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --out_path data/no_use/no_use_qa_specs_temporal_scope.jsonl \
  --report_path data/no_use/no_use_qa_specs_report_temporal_scope.json \
  --generator llm_reasoning \
  --qa_profile reasoning \
  --llm_model gpt-5-mini \
  --llm_reasoning_count 50 \
  --num_workers 10 \
  --token_path . \
  --api_key_file openai_key.txt
```

### 2.3 Generate both branches together

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/generate_no_use_qa_specs.py \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --out_path data/no_use/no_use_qa_specs_scope.jsonl \
  --report_path data/no_use/no_use_qa_specs_report_scope.json \
  --generator rule_based \
  --qa_profile both
```

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/generate_no_use_qa_specs.py \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --out_path data/no_use/no_use_qa_specs_temporal_scope.jsonl \
  --report_path data/no_use/no_use_qa_specs_report_temporal_scope.json \
  --generator rule_based \
  --qa_profile both
```

## 3) Evaluate

### 3.1 Evaluate `direct_cue` branch

Use `200` sampled questions.

`GPT-5.1 scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_scope_direct_cue.jsonl \
  --out_csv data/no_use/no_use_eval_results_scope_direct_cue.csv \
  --summary_path data/no_use/no_use_eval_summary_scope_direct_cue.json \
  --model gpt-5.1 \
  --provider openai \
  --world both \
  --branch direct_cue \
  --workers 6 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --max_questions 200 \
  --sampling stratified \
  --sample_seed 0 \
  --sample_manifest_out data/no_use/sample_manifest_200_scope_direct_cue.json
```

`GPT-5.1 temporal_scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_temporal_scope_direct_cue.jsonl \
  --out_csv data/no_use/no_use_eval_results_temporal_scope_direct_cue.csv \
  --summary_path data/no_use/no_use_eval_summary_temporal_scope_direct_cue.json \
  --model gpt-5.1 \
  --provider openai \
  --world both \
  --branch direct_cue \
  --workers 6 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --max_questions 200 \
  --sampling stratified \
  --sample_seed 0 \
  --sample_manifest_out data/no_use/sample_manifest_200_temporal_scope_direct_cue.json
```

Reuse the same sampled questions for `GPT-5-mini`:

`scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_scope_direct_cue.jsonl \
  --out_csv data/no_use/no_use_eval_results_scope_direct_cue_gpt5mini.csv \
  --summary_path data/no_use/no_use_eval_summary_scope_direct_cue_gpt5mini.json \
  --model gpt-5-mini \
  --provider openai \
  --world both \
  --branch direct_cue \
  --workers 6 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --sample_manifest_in data/no_use/sample_manifest_200_scope_direct_cue.json
```

`temporal_scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_temporal_scope_direct_cue.jsonl \
  --out_csv data/no_use/no_use_eval_results_temporal_scope_direct_cue_gpt5mini.csv \
  --summary_path data/no_use/no_use_eval_summary_temporal_scope_direct_cue_gpt5mini.json \
  --model gpt-5-mini \
  --provider openai \
  --world both \
  --branch direct_cue \
  --workers 6 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --sample_manifest_in data/no_use/sample_manifest_200_temporal_scope_direct_cue.json
```

### 3.2 Evaluate `reasoning` branch

Current evaluation note:
- the active `reasoning` branch assumes the spec file contains only:
  - `reasoning_alternative` (or legacy `blocked_use_alternative`)
  - `reasoning_recovery_use` (or legacy `no_use_recovery_use`)
  - shared utility rows
- `reasoning_insufficient` is not part of the current reported results

Use `200` sampled questions.

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_scope.jsonl \
  --out_csv data/no_use/no_use_eval_results_scope.csv \
  --summary_path data/no_use/no_use_eval_summary_scope.json \
  --model gpt-5.1 \
  --provider openai \
  --world both \
  --branch reasoning \
  --workers 8 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --max_questions 200 \
  --sampling stratified \
  --sample_seed 0 \
  --sample_manifest_out data/no_use/sample_manifest_200_scope.json
```

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_temporal_scope.jsonl \
  --out_csv data/no_use/no_use_eval_results_temporal_scope.csv \
  --summary_path data/no_use/no_use_eval_summary_temporal_scope.json \
  --model gpt-5.1 \
  --provider openai \
  --world both \
  --branch reasoning \
  --workers 8 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --max_questions 200 \
  --sampling stratified \
  --sample_seed 0 \
  --sample_manifest_out data/no_use/sample_manifest_200_temporal_scope.json
```

Reuse the same sampled questions for `GPT-5-mini`:

`scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_scope.jsonl \
  --out_csv data/no_use/no_use_eval_results_scope_gpt5mini.csv \
  --summary_path data/no_use/no_use_eval_summary_scope_gpt5mini.json \
  --model gpt-5-mini \
  --provider openai \
  --world both \
  --branch reasoning \
  --workers 8 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --sample_manifest_in data/no_use/sample_manifest_200_scope.json
```

`temporal_scope`

```bash
conda run --no-capture-output -n agent python -u privacy_test/no_use/evaluate_no_use_worlds.py \
  --meta_path data/no_use/no_use_meta_temporal_scope.jsonl \
  --spec_path data/no_use/no_use_qa_specs_temporal_scope.jsonl \
  --out_csv data/no_use/no_use_eval_results_temporal_scope_gpt5mini.csv \
  --summary_path data/no_use/no_use_eval_summary_temporal_scope_gpt5mini.json \
  --model gpt-5-mini \
  --provider openai \
  --world both \
  --branch reasoning \
  --workers 8 \
  --retries 12 \
  --rate_limit_rounds 6 \
  --rate_limit_sleep 5 \
  --sample_manifest_in data/no_use/sample_manifest_200_temporal_scope.json
```

Evaluation summary uses the unified metric schema:

- `constraint_score`
- `forbidden_leakage`
- `constraint_follow_increase`
- `forbidden_leakage_drop`
- `clean_utility`
- `robust_utility`
- `clean_utility_drop`
- `robust_utility_drop`

`temporal_scope` additionally reports:

- `recovery_rate`
- `recovery_delta`
