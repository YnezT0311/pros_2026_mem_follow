# Run Agent Workflow For Application MCQ Data

Use this file as the orchestration recipe when Codex/Claude needs to regenerate
application MCQ data after `data/application` is missing or when building a new
run.

This workflow is intentionally agent-driven. The Python scripts only render,
compare, transform, and export artifacts around the agent-generated items.

The canonical application MCQ quality standard is:

```text
gen_data/build_benchmark/application/agent_workflow/APPLICATION_MCQ_WORKFLOW.md
```

Role-specific skills live under:

```text
gen_data/build_benchmark/application/agent_workflow/skills/
```

Use these application-specific instructions instead of the older generic MCQ
reviewer/fixer workflow under `gen_data/agent_workflow`.

## Inputs

Required:

- conversation packages:
  - `data/generated/<topic>/<sample>/conversation_package.json`
- topic list, usually:
  - `travelPlanning`
  - `financialConsultation`
  - `medicalConsultation`
- sample id, usually:
  - `persona0_sample0`

Optional:

- prior kept specs or examples
- previous review files
- manually curated examples

## Output Root

Use a new run directory:

```text
tmp/application_pilot/<run_name>/
```

Recommended structure:

```text
tmp/application_pilot/<run_name>/
  GOAL.md
  SUMMARY.md
  travelPlanning/persona0_sample0/
    items.json
    planner_candidates.json
    planner_review.md
    questioner_notes.json
    analyst_precheck*.json
    analyst_precheck*.md
    fixer_notes*.json
  financialConsultation/persona0_sample0/
  medicalConsultation/persona0_sample0/
  plain_solver_inputs/
  transformed_worlds/
```

## Orchestration Stages

### Stage 0: Write Goal

The orchestrator writes `GOAL.md` before spawning workers.

Goal must include:

- one item per topic-stage when possible
- original source stage context only, loaded from
  `data/generated/<topic>/<sample>/conversation_package.json`
- no full-history concatenation
- no synthetic conversation context; workers may draft MCQ items, but final
  solver inputs and transformed worlds must be rendered from the source stage
- target memory must be a lasting preference, stable constraint, reusable
  habit, or durable tendency; do not use turn-specific facts
- question must be a concrete natural user request, not a testing prompt
- question must be answerable without seeing choices
- choices must be natural assistant answers with explicit roles:
  `use_memory`, `without_memory`, and `plausible_wrong`
- with-target baseline must select use-memory
- without-target baseline must select without-memory
- prefer one question per stage, but replan target before dropping a stage
- weak items should be fixed, replanned, or dropped, not kept for count

### Stage 1: Question Planning

Spawn one question-planner worker per topic, or per stage batch when a finer
review is needed.

Workers may run in parallel because write scopes are disjoint.

Write scopes:

```text
tmp/application_pilot/<run_name>/<topic>/<sample>/
```

Worker final output:

- `planner_candidates.json`
- `planner_review.md`

Worker prompt template:

```text
You are the application MCQ question planner for <TOPIC>.
You are not alone in the codebase; only write <OUTPUT_TOPIC_DIR>.

Read:
- <RUN_ROOT>/GOAL.md
- gen_data/build_benchmark/application/agent_workflow/APPLICATION_MCQ_WORKFLOW.md
- gen_data/build_benchmark/application/agent_workflow/skills/application_mcq_question_planner/SKILL.md
- data/generated/<TOPIC>/<SAMPLE>/conversation_package.json
- optional examples if provided

Task:
1. For each stage, inspect target user turns.
2. Select at most one best target candidate per stage.
3. Accept only lasting preferences, stable constraints, reusable habits, or
   durable tendencies.
4. Reject one-off prices, dates, quantities, lab values, message details, and
   other turn-specific facts.
5. Check contamination by removing the target user turn and immediate assistant
   reply, then asking whether other stage turns still support the same
   preference.
6. Preserve stage coverage when possible: provide backup candidates before
   recommending drop.
7. Write planner_candidates.json and planner_review.md.

Domain safety:
- finance: no absolute investment advice or return promises.
- medical: no diagnosis, no replacement for clinician care, preserve urgent
  warning-sign boundaries.
```

### Stage 2: Question Writing

Spawn questioner workers for planner-accepted candidates. They may run in
parallel by topic.

Questioner output:

- `items.json`
- `questioner_notes.json`

Questioner prompt template:

```text
You are the application MCQ questioner for <TOPIC>.
You are not alone in the codebase; only write <OUTPUT_TOPIC_DIR>.

Read:
- <RUN_ROOT>/GOAL.md
- gen_data/build_benchmark/application/agent_workflow/APPLICATION_MCQ_WORKFLOW.md
- gen_data/build_benchmark/application/agent_workflow/skills/application_mcq_questioner/SKILL.md
- <OUTPUT_TOPIC_DIR>/planner_candidates.json
- data/generated/<TOPIC>/<SAMPLE>/conversation_package.json

Task:
1. For each accepted planner candidate, write one natural application MCQ.
2. The question must be a concrete real user request and answerable without
   choices.
3. The question must not reveal the target preference or mention memory/testing.
4. Create choices with roles:
   - use_memory
   - without_memory
   - plausible_wrong
5. Choices must be natural assistant answers, not meta instructions.
6. Write items.json and questioner_notes.json.

Output:
- items.json
```

### Stage 3: Analyst Precheck

Spawn analyst workers for topic item files before solver rendering. Analysts may
run in parallel and should not edit `items.json`.

Analyst output:

- `analyst_precheck.json`
- `analyst_precheck.md`

Analyst prompt template:

```text
You are the application MCQ analyst for <TOPIC>.
Do not edit items. Only write analyst_precheck.json and analyst_precheck.md.

Read:
- <RUN_ROOT>/GOAL.md
- gen_data/build_benchmark/application/agent_workflow/APPLICATION_MCQ_WORKFLOW.md
- gen_data/build_benchmark/application/agent_workflow/skills/application_mcq_analyst/SKILL.md
- <OUTPUT_TOPIC_DIR>/items.json
- <OUTPUT_TOPIC_DIR>/planner_candidates.json
- data/generated/<TOPIC>/<SAMPLE>/conversation_package.json if needed

Review every item for:
- target is lasting, not turn-specific
- target remains unique after target removal
- question is a natural user request and answerable without choices
- question does not leak target memory
- choices are natural and non-meta
- use-memory choice only wins when target memory is present
- without-memory choice is a reasonable generic answer
- no excessive duplication across stages
- finance/medical safety where relevant

Decision per item:
- accept
- revise
- replan_target
- drop_stage

For every revise/replan/drop, provide:
- severity
- issues
- required_fix
- confidence
```

### Stage 4: Fixer Loop

Spawn fixer workers for any topic with analyst `revise`, `replan_target`, or
`drop_stage`.

Fixer prompt template:

```text
You are the application MCQ fixer for <TOPIC>.
Only edit <OUTPUT_TOPIC_DIR>.

Read:
- items.json
- planner_candidates.json
- analyst_precheck.json/md
- GOAL.md
- gen_data/build_benchmark/application/agent_workflow/APPLICATION_MCQ_WORKFLOW.md
- gen_data/build_benchmark/application/agent_workflow/skills/application_mcq_fixer/SKILL.md
- conversation package if needed

Only process items that need revision.

For revise:
- first edit question wording if target is valid
- then edit choices if needed
- remove all meta/testing language
- update items.json and record fixer notes

For replan_target:
- return to planner output and choose a better same-stage candidate if possible
- if no same-stage candidate works, mark dropped with clear drop_reason

Do not keep weak or turn-specific items for count.
```

Then run another analyst pass over revised/weak items.

Stop this loop when:

- no blocking `revise` or `replan_target` remains, or
- the item is dropped.

### Stage 5: Build Plain Solver Inputs With Python

Use the exact plain eval message shape:

```text
persona system message
context messages
final build_eval_prompt(question, choices)
```

Generate:

```text
plain_solver_inputs/with_target_baseline.json
plain_solver_inputs/without_target_baseline.json
plain_solver_inputs/expected.json
```

Command:

```bash
python -m gen_data.build_benchmark.application.build_plain_solver_inputs \
  --items-root tmp/application_pilot/<run_name> \
  --conversation-root data/generated \
  --output-root tmp/application_pilot/<run_name>/plain_solver_inputs
```

### Stage 6: Spawn Plain Solvers

Spawn two solvers:

- with-target solver
- without-target solver

Solvers must only read the blind input file assigned to them.

Solver prompt template:

```text
You are a plain API behavior solver.
You may only read <BLIND_INPUT_JSON>.

Follow:
- gen_data/build_benchmark/application/agent_workflow/skills/application_mcq_solver/SKILL.md

Do not read expected labels, items.json, planner files, analyst files, fixer
files, or questioner notes.

For each item, inspect the messages exactly as the plain evaluation adapter
would send them:
- persona system message
- context messages
- final MCQ prompt

Choose the most appropriate (a)/(b)/(c).

Output:
{
  "condition": "<with_target_baseline|without_target_baseline>",
  "items": [
    {"id": "...", "choice": "A|B|C", "rationale": "brief"}
  ]
}
```

Required output file names:

```text
tmp/application_pilot/<run_name>/plain_solver_inputs/with_target_solver_results.json
tmp/application_pilot/<run_name>/plain_solver_inputs/without_target_solver_results.json
```

### Stage 7: Compare Solver Results With Python

The orchestrator compares solver choices to expected labels.

Pass requires:

```text
with_target_solver_choice == expected.with_target_baseline
without_target_solver_choice == expected.without_target_baseline
```

For failures:

- run the application MCQ analyst on solver rationales
- use the fixer for question/choice repairs when the target is valid
- replan same-stage target when the target is invalid or contaminated
- drop only if no target-unique lasting memory is available

Command:

```bash
python -m gen_data.build_benchmark.application.compare_plain_solver_results \
  --input-root tmp/application_pilot/<run_name>/plain_solver_inputs \
  --output tmp/application_pilot/<run_name>/plain_solver_inputs/baseline_gate_comparison.json
```

The comparison file lists failed items with solver rationales. The orchestrator
uses those rationales to spawn focused revision workers or to make direct edits.

### Stage 8: Build Transformed Worlds

After baseline-gated items are finalized, run:

```bash
python -m gen_data.build_benchmark.application.build_worlds \
  --items-root tmp/application_pilot/<run_name> \
  --conversation-root data/generated \
  --output-root tmp/application_pilot/<run_name>/transformed_worlds
```

Expected worlds:

- `never_seen_baseline`
- `seen_baseline`
- `no_store`
- `forget`
- `no_use_active`
- `no_use_release`

### Stage 9: Export To data/application

After human approval, export to:

```text
data/application/mcq_work/<topic>/<sample>/
data/application/mcq/
```

Do not export unapproved drafts.

Command:

```bash
python -m gen_data.build_benchmark.application.export_from_tmp \
  --run-root tmp/application_pilot/<run_name> \
  --output-root data/application
```

Expected exported files:

```text
data/application/mcq_work/<topic>/<sample>/application_items.json
data/application/mcq_work/<topic>/<sample>/review/
data/application/mcq_work/_worlds/application_worlds.json
data/application/mcq_work/_worlds/plain_inputs/<world>.json
data/application/mcq/application_mcq.json
data/application/mcq/by_world/<world>.json
```

### Stage 10: Sanity Check With Existing Tools

These are simple checks, not a separate generation step:

```bash
python - <<'PY'
import json, glob
for path in glob.glob("data/application/**/*.json", recursive=True):
    json.load(open(path))
print("all application json files parse")
PY

python - <<'PY'
import json
d=json.load(open("data/application/mcq/application_mcq.json"))
print(d["total_items"], d["total_world_records"], d["topic_counts"])
for world in d["world_order"]:
    x=json.load(open(f"data/application/mcq/by_world/{world}.json"))
    print(world, len(x["items"]))
PY
```

## Quality Bar

An item is kept only if:

- it passes planner/questioner/analyst/fixer review
- it passes plain-solver baseline gate
- target memory is lasting and naturally changes the answer
- the target is not a turn-specific fact
- no hidden target fact is leaked in the question
- generic answer is not artificially bad
- question and choices have no benchmark/meta wording

The desired outcome is fewer high-quality questions, not full stage coverage at
all costs. However, before dropping a stage, replan the target within the same
stage because the preferred shape is still one valid application question per
stage when the source allows it.
