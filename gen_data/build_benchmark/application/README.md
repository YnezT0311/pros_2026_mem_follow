# Application MCQ Data Generation

This directory contains the build helpers for application-style memory-control
MCQs.

Application data generation has two parts:

1. Agent-assisted item generation and validation.
2. Python scripts that render solver inputs, compare solver results, build
   transformed worlds, and export approved data.

## Files In This Directory

```text
README.md
```

Human-facing entry point. Start here.

```text
RUN_AGENT_WORKFLOW.md
```

Instructions for Codex/Claude when new MCQ items need to be generated or
repaired. This is the agent-group playbook: question planners, questioners,
solver agents, analysts, fixers, revision loops, and drop criteria.

```text
agent_workflow/APPLICATION_MCQ_WORKFLOW.md
```

Canonical quality standard for application MCQs. This defines the required
planner/questioner/solver/analyst/fixer workflow, the lasting-preference target
standard, natural-question requirements, and baseline pass criteria. Use this
application-specific workflow instead of the older generic MCQ workflow under
`gen_data/agent_workflow`.

```text
agent_workflow/skills/*/SKILL.md
```

Role-specific instructions for application MCQ agents:
question planner, questioner, solver, analyst, and fixer.

```text
build_plain_solver_inputs.py
```

Deterministically renders the with-target and without-target baseline prompts
that solver agents answer. This prevents the solver gate from drifting away from
the real plain-eval prompt format.

```text
compare_plain_solver_results.py
```

Deterministically compares blind solver-agent answers against expected baseline
labels. The agent group uses the failure list and rationales to revise or drop
items.

```text
build_worlds.py
```

Builds the six final transformed worlds from approved kept items.

```text
export_from_tmp.py
```

Copies approved run artifacts into `data/application`.

## When Starting From Scratch

If `data/application` is missing or a new application dataset is needed, the
user only needs to ask Codex/Claude to run the agent workflow:

```text
Follow gen_data/build_benchmark/application/RUN_AGENT_WORKFLOW.md to generate
application MCQ data for travelPlanning, financialConsultation, and
medicalConsultation.
```

The orchestrating agent handles planner/questioner/solver/analyst/fixer
subagents and runs the Python scripts below. The user should not manually run
each command unless they are debugging.

The agent workflow writes a run directory such as:

```text
tmp/application_pilot/<run_name>/
```

That directory should contain per-topic `items.json`, review files,
plain-solver validation outputs, and transformed worlds.

## User Responsibilities

The user is responsible for:

1. Starting the run by asking Codex/Claude to follow `RUN_AGENT_WORKFLOW.md`.
2. Reviewing planner, analyst, fixer summaries, and final `SUMMARY.md`.
3. Deciding whether the candidate data is acceptable.
4. Approving export to `data/application`.

The user is not expected to:

- manually spawn every planner/questioner/solver/analyst/fixer agent
- manually render plain solver prompts
- manually compare solver outputs
- manually build transformed worlds

## Execution Order

1. **Run the agent workflow**

   Give Codex/Claude this instruction:

   ```text
   Follow gen_data/build_benchmark/application/RUN_AGENT_WORKFLOW.md to generate
   application MCQ data.
   ```

   The agent should apply the application-specific standard in:

   ```text
   gen_data/build_benchmark/application/agent_workflow/APPLICATION_MCQ_WORKFLOW.md
   ```

   The agent creates and iterates on:

   ```text
   tmp/application_pilot/<run_name>/<topic>/<sample>/items.json
   tmp/application_pilot/<run_name>/<topic>/<sample>/planner_candidates.json
   tmp/application_pilot/<run_name>/<topic>/<sample>/questioner_notes.json
   tmp/application_pilot/<run_name>/<topic>/<sample>/analyst_precheck*.json
   tmp/application_pilot/<run_name>/<topic>/<sample>/fixer_notes*.json
   ```

2. **Build blind solver inputs**

   The orchestrating agent will run `build_plain_solver_inputs.py`. The solver
   agents must answer these generated files, not hand-written prompts.

3. **Run solver agents**

   The agent workflow spawns two blind solver agents:

   - one reads `with_target_baseline.json`
   - one reads `without_target_baseline.json`

   They write:

   ```text
   with_target_solver_results.json
   without_target_solver_results.json
   ```

4. **Compare solver results**

   The orchestrating agent will run `compare_plain_solver_results.py`. If
   failures remain, it returns to the agent workflow for targeted revision/drop
   and reruns steps 2-4.

5. **Build transformed worlds**

   Once baseline gate passes, the orchestrating agent will run
   `build_worlds.py`.

6. **Export approved data**

   After human approval, the orchestrating agent will run `export_from_tmp.py`.

## Scripts

### Build plain solver inputs

The orchestrating agent will run:

```bash
python -m gen_data.build_benchmark.application.build_plain_solver_inputs \
  --items-root tmp/application_pilot/<run_name> \
  --conversation-root data/generated \
  --output-root tmp/application_pilot/<run_name>/plain_solver_inputs
```

This creates:

```text
plain_solver_inputs/with_target_baseline.json
plain_solver_inputs/without_target_baseline.json
plain_solver_inputs/expected.json
```

### Compare plain solver outputs

After the agent writes:

```text
plain_solver_inputs/with_target_solver_results.json
plain_solver_inputs/without_target_solver_results.json
```

the orchestrating agent will run:

```bash
python -m gen_data.build_benchmark.application.compare_plain_solver_results \
  --input-root tmp/application_pilot/<run_name>/plain_solver_inputs \
  --output tmp/application_pilot/<run_name>/plain_solver_inputs/baseline_gate_comparison.json
```

Failed items should be revised by the agent or dropped.

### Build transformed worlds

The orchestrating agent will run:

```bash
python -m gen_data.build_benchmark.application.build_worlds \
  --items-root tmp/application_pilot/<run_name> \
  --conversation-root data/generated \
  --output-root tmp/application_pilot/<run_name>/transformed_worlds
```

Worlds:

- `never_seen_baseline`
- `seen_baseline`
- `no_store`
- `forget`
- `no_use_active`
- `no_use_release`

### Export approved data

Only export after the agent workflow has passed analyst review and the
plain-solver baseline gate. After user approval, the orchestrating agent will
run:

```bash
python -m gen_data.build_benchmark.application.export_from_tmp \
  --run-root tmp/application_pilot/<run_name> \
  --output-root data/application
```

## Current Data Location

Runtime-ready exported data:

```text
data/application/mcq/
```

Work/audit data:

```text
data/application/mcq_work/
```
