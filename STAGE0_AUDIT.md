# Stage 0 Audit Memo

## Scope

This memo records the current `Stage 0` pass over:
- baseline data completeness
- persona and history sanity
- baseline conversation naturalness
- rebuilt `no_use` worlds and QA specs
- current rebuild status of `retention` and `deletion`

This is a current audit snapshot, not a final benchmark certification.

## Method

This pass used:
- direct file inspection over `data/output/*/*.json`
- rebuilt `no_use` worlds under:
  - `data/no_use/world_scope`
  - `data/no_use/world_temporal_scope`
- regenerated `no_use` QA specs under:
  - `data/no_use/no_use_qa_specs_scope.jsonl`
  - `data/no_use/no_use_qa_specs_temporal_scope.jsonl`
- lightweight statistics over completeness, turn lengths, QA counts, and obvious policy leakage

The audit prompts and contamination-rewrite prompts are recorded in [TODO.md](/mnt/yao_data/proj_2026_agent/PersonaMem-main/TODO.md).

## High-Level Conclusion

`Stage 0` is substantially improved, but not fully cleared.

Current status:
- baseline completeness is improved in handling because incomplete files are now explicitly skipped during world building
- baseline persona/history/conversation quality is mostly usable
- rebuilt `no_use` is currently the cleanest world
- rebuilt `deletion` QA is now clean, but rebuilt `deletion` world still has residual conflicts
- rebuilt `retention` world is still incomplete because the new full rebuild is still running

Main remaining blockers:
1. two baseline files are still incomplete
2. period labels still do not semantically behave like literal `next week / next month / next year`
3. rebuilt `deletion` world still has unresolved residual conflicts
4. rebuilt `retention` world is not complete yet

## Findings

### 1. Baseline Completeness Is Better Enforced, But Source Data Still Has 2 Incomplete Files

Current source status:
- total baseline files: `24`
- complete files: `22`
- incomplete files: `2`

Incomplete files:
- [conversation_medicalConsultation_persona5_sample0.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/output/medicalConsultation/conversation_medicalConsultation_persona5_sample0.json)
  - missing `Conversation Next Week / Next Month / Next Year`
- [conversation_travelPlanning_persona2_sample0.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/output/travelPlanning/conversation_travelPlanning_persona2_sample0.json)
  - missing `Conversation Next Week / Next Month / Next Year`

Interpretation:
- this is still a source-data quality issue
- however, builders now explicitly print and skip incomplete files instead of silently copying them into worlds

Action:
- either regenerate these two source files or permanently exclude them from benchmark splits

### 2. Persona Quality Looks Acceptable in the Current Pass

Observed signal:
- complete-file personas are fairly detailed
- `Expanded Persona` length is substantial: mean about `111` words, range `93-142`
- sampled personas remain specific enough to support personalized memory tests

Current judgment:
- no obvious new persona collapse or generic one-line summaries
- no immediate blocker found here

Caveat:
- stereotype/template reuse still needs broader manual sampling before this area can be considered fully closed

### 3. History Content Is Substantive, But Period Labels Are Still Semantically Misleading

Observed pattern:
- history entries are generally concrete and behaviorally meaningful
- however, the stage names do not act like literal calendar gaps

Example:
- in [conversation_legalConsultation_persona1_sample0.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/output/legalConsultation/conversation_legalConsultation_persona1_sample0.json), `Init` events are dated in May 1998, while `Next Week` events are dated in September 1998

Interpretation:
- the benchmark stages work as ordered temporal blocks
- they do not reliably behave like literal `init -> next week -> next month -> next year`

Action:
- for analysis, treat them as abstract `period 1 / 2 / 3 / 4`
- do not present them as literal week/month/year intervals unless the source histories are rewritten

### 4. Baseline Conversations Are Mostly Usable, Though Some Openers Still Feel Benchmark-Like

Current baseline stats on complete files:
- mean user length: `24.6` words
- mean assistant length: `30.7` words
- user turns shorter than 8 words: `0`
- assistant turns shorter than 8 words: `0`

Positive signal:
- baseline assistant replies are not collapsing into empty acknowledgements
- turn lengths are healthy

Remaining issue:
- some initial user openers still read like benchmark setup rather than natural chat, for example:
  - `Hi — I want to go through my financial activities one by one and get practical suggestions for each.`
  - `I'm preparing for a legal consultation and want to walk through a sequence of my recent activities so you can point out where I might need help.`

Interpretation:
- baseline dialogue is usable
- but opener phrasing still slightly increases synthetic feel

### 5. Rebuilt No-Use World Is in Good Shape

#### Scope world

From [no_use_summary_scope.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_summary_scope.json):
- processed files: `22`
- rows: `88`
- conflicts before repair: `8`
- conflicts after repair: `0`

Naturalness check:
- assistant lines scanned: `1278`
- obvious policy-style assistant leakage by heuristic: `0`
- assistant lines shorter than 8 words: `0`

Interpretation:
- the current `scope` rebuild looks clean
- the assistant-only in-scope rewrite strategy is working well

#### Temporal scope world

From [no_use_summary_temporal_scope.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_summary_temporal_scope.json):
- processed files: `22`
- rows: `39`
- conflicts before repair: `3`
- conflicts after repair: `0`

Naturalness check:
- assistant lines scanned: `1268`
- heuristic policy-style assistant lines: `17`
- assistant lines shorter than 8 words: `1`

The remaining policy-style lines are almost all explicit `off` acknowledgements such as:
- `Got it, the temporary no-use restriction is lifted; I'll reference the previous details now.`
- `All set — the restriction is lifted and previous details can be used.`

Interpretation:
- temporal no-use is usable
- the remaining explicitness is mostly concentrated in `off` acknowledgements, not in ordinary downstream assistance

### 6. No-Use QA Quality Is Stronger Than Before

Current rebuilt QA stats:

Scope:
- file: [no_use_qa_specs_scope.jsonl](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_qa_specs_scope.jsonl)
- total specs: `453`
- breakdown:
  - `no_use_blocked`: `151`
  - `utility_recall`: `151`
  - `utility_policy_pressure`: `151`
- opener-like target facts: `0`
- short gold answers under 6 words: `0`

Temporal scope:
- file: [no_use_qa_specs_temporal_scope.jsonl](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_qa_specs_temporal_scope.jsonl)
- total specs: `297`
- breakdown:
  - `no_use_blocked`: `45`
  - `utility_recall`: `99`
  - `utility_policy_pressure`: `99`
  - `no_use_recovery`: `54`
- opener-like target facts: `0`
- short gold answers under 6 words: `0`

Interpretation:
- the earlier opener/scaffold utility-target problem is no longer present in rebuilt `no_use`
- QA coverage is balanced within each no-use mode

### 7. Grid / Coverage Is Still Uneven

Current topic coverage:

Scope meta:
- financial: `20`
- legal: `28`
- medical: `24`
- travel: `16`

Temporal scope meta:
- financial: `7`
- legal: `15`
- medical: `9`
- travel: `8`

Interpretation:
- `no_use scope` is usable but still not topic-balanced
- `temporal_scope` remains a lower-N supplemental condition rather than a main pooled dataset

### 8. Deletion QA Is Now Clean, But Deletion World Is Not Yet Cleared

Current rebuilt QA stats:

- file: [deletion_qa_specs.jsonl](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/deletion/deletion_qa_specs.jsonl)
- total specs: `750`
- breakdown:
  - `deleted_direct`: `150`
  - `deleted_paraphrase`: `150`
  - `allowed_recall`: `150`
  - `allowed_policy_pressure`: `150`
  - `allowed_reasoning`: `150`
- opener-like target facts: `0`
- short gold answers under 6 words: `0`

Current rebuilt world stats:

- file: [deletion_summary.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/deletion/deletion_summary.json)
- processed files: `22`
- reveals / deletes: `88 / 88`
- conflicts before repair: `12`
- conflicts after repair: `12`

Naturalness scan:
- assistant lines scanned: `1278`
- policy-style assistant lines: `40`
- non-direct-ack policy lines found by heuristic: `0`

Interpretation:
- rebuilt deletion QA is in good shape
- rebuilt deletion world is not yet cleared because residual conflicts remain unresolved

### 9. Retention Rebuild Is Still Incomplete

Current status:
- the new full retention rebuild is still running
- current retention world directory is incomplete

At the time of this memo:
- rebuilt retention files present: `9`
- `build_retention_world.py` process is still alive

Interpretation:
- I do not yet trust any full-world claim about rebuilt `retention`
- retention QA should not be treated as final until the rebuild fully completes

## Preliminary Assessment by Audit Area

### Persona

Status:
- mostly acceptable

### History

Status:
- substantive but period labels are not trustworthy as literal calendar semantics

### Baseline Conversation

Status:
- usable

### No-Use World

Status:
- largely cleared for current Stage 0 use

### Retention World

Status:
- pending full rebuild and re-audit

### Deletion World

Status:
- rebuilt but not yet cleared

### QA

Status:
- rebuilt `no_use` QA looks good
- rebuilt `deletion` QA looks good
- `retention` QA is still pending because the rebuild is incomplete

## Immediate Next Steps

1. Finish the full `retention` rebuild under the new LLM contamination workflow.
2. Resolve the `12` residual conflicts in rebuilt `deletion` world.
3. Regenerate final `retention` QA specs after the retention rebuild completes.
4. Re-run Stage 0 conversation and QA audit on rebuilt `retention` and repaired `deletion`.
5. Either regenerate or permanently exclude the 2 incomplete baseline files.
6. Treat temporal stages as `period 1 / 2 / 3 / 4` in analysis until the source histories are rewritten.

## Stage 0 Status

- Persona audit: mostly passed in current sample
- History audit: partially passed, but period-label semantics remain a known issue
- Baseline conversation audit: mostly passed
- No-use world audit: passed for current use
- Retention world audit: pending full rebuild
- Deletion world audit: rebuilt but not passed
- QA audit: passed for rebuilt `no_use` and rebuilt `deletion`; pending for rebuilt `retention`
- Grid/quota audit: partially completed, imbalance still present
