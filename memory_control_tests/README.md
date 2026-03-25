# Memory Control Tests

## Goal

This folder contains a new baseline-first pipeline for memory-control evaluation without modifying the legacy `privacy_test/` code.

The design is built around one principle: we first stabilize a clean baseline world, and only then apply memory-control interventions.

The intended workflow is:

1. start from `data/output/`
2. build a stricter `data/baseline/`
3. fix `Initial Stage` interaction turns into `key_turns` and protected `probe_turns`
4. resolve future conflicts for the selected keys if needed
5. treat that resolved conversation as the true baseline world
6. run evaluation-time conversation transforms for:
   - `no_store`
   - `forget`
   - `no_use`
7. evaluate with fixed MCQ sets over the transformed context

The main design choice is that we do **not** materialize a separate static world for every gap condition. Instead, we keep one baseline conversation and insert instruction/reply turns on the fly during evaluation.

## Baseline World

`build_baseline.py` copies each conversation file from `data/output/` into `data/baseline/` and writes a sidecar spec file:

- `conversation_...json`
- `conversation_....memory_control.json`

The copied conversation file is the candidate baseline world. The sidecar describes whether that copied conversation is already usable as-is or whether future targeted revision would still be needed.

The sidecar records:

- all `Initial Stage` interaction candidates
- selected `key_turns`
- selected `protected_probe_turns`
- future-conflict evidence for each candidate
- a `baseline_resolution` summary describing whether the selected keys are already conflict-free

When the selected key set is already conflict-free, the baseline conversation can be copied directly from `data/output/` without editing. That is the preferred path because it keeps the baseline world maximally close to the original generated conversation.

## Candidate Selection

Selection is currently restricted to `Initial Stage` interaction turns.

This is deliberate. We use the `Initial Stage` to establish memory, and then use later stages as the intervention and evaluation window.

The builder:

- reads `Conversation History Initial Stage`
- maps interaction timestamps to their `Conversation Initial Stage` blocks
- extracts:
  - `[Task Goal]`
  - `[Context Can Add]`
  - `[Sensitive Info]`
  - `source_event_id`
- checks future structured history for conflicts
- penalizes earlier exact duplicates of the same sensitive value
- prefers the last exact occurrence when a sensitive value is repeated

The default split is approximately half-and-half:

- half of the `Initial Stage` interaction turns become `key_turns`
- the rest become protected `probe_turns`

The current selection rule prioritizes conflict-free keys first. This means the baseline builder prefers:

- lower future conflict count
- the last exact occurrence of repeated sensitive values
- stable tiebreaking by `history_index`

## Conflict Detection

The offline conflict check intentionally uses only information already present in the generated structured history:

- `[Task Goal]`
- `[Context Can Add]`
- `[Sensitive Info]`
- `source_event_id`
- event lineage from `relations`

We do not use the final natural-language conversation text for conflict detection. The goal is to make the filtering step deterministic and traceable from the structured history alone.

Future conflicts are detected with hard rules:

- exact reuse of normalized `[Sensitive Info]` values in later interaction history
- exact reuse of normalized `[Context Can Add]` keys in later interaction history
- later events or interactions that belong to the candidate source event's descendant chain
- a boolean high-similarity check on later `[Task Goal]` text

If a candidate has no later-stage hits under these rules, it is treated as baseline-safe. If it has hits, the spec marks it as `needs_revision`.

## Conflict Resolution Policy

The current policy is conservative:

- first try to choose a key set that is already conflict-free
- only if that fails should we consider targeted revision

In other words, the baseline builder currently prefers:

- `detect + annotate + avoid`

over:

- `detect + rewrite`

This is intentional. For the memory-control benchmark, it is better to keep the baseline conversation as close as possible to the original generated conversation unless revision is genuinely necessary.

If revision becomes necessary later, it should be:

- targeted only to the selected keys
- limited to later conflicting turns
- minimal rather than full regeneration

## Test Worlds

All three test settings reuse the same baseline world and the same QA set.

### `no_store`

- modify the chosen key turn itself
- insert the control request in the same user turn
- add a templated assistant acknowledgement in the same block

For `no_store`, templates support either:

- front-loaded insertion (`user_prefix`)
- end-loaded insertion (`user_suffix`)

### `forget`

- keep the key turn unchanged
- append a standalone instruction/reply turn in a later stage
- the default insertion point is the end of `Conversation Early Stage`

### `no_use`

- keep the key turn unchanged
- append a standalone restriction turn in a later stage
- optionally append a later release turn

The concrete instruction/reply pools live in `templates.json`.

The current template pool uses direct instruction wording only.

## Evaluation-Time Transform

`transforms.py` provides:

- `apply_no_store(...)`
- `apply_forget(...)`
- `apply_no_use(...)`
- `build_context_messages(...)`

This lets evaluation:

1. load a baseline conversation
2. insert control turns for the chosen world and gap condition
3. build the exact conversation context up to a target `ask_period`
4. append an MCQ prompt

This preserves the same basic evaluation shape as the old `privacy_test` scripts, which also feed the model a truncated conversation context followed by a final multiple-choice query.

## Evaluation Pipeline

The intended evaluation unit is:

- one baseline world
- one selected key or probe turn
- one world condition
- one `ask_period`
- one MCQ

At evaluation time:

1. start from the baseline conversation
2. apply the world transform if needed
3. truncate the conversation at the chosen `ask_period`
4. convert the resulting lines into chat messages
5. append one MCQ prompt
6. score the model's choice

This setup makes it easy to vary:

- key-to-instruction gap
- instruction-to-test gap
- restriction-to-release gap for `no_use`

without storing a large number of static world files.

## MCQ Design

The MCQ layer should be fixed **after** the baseline world is fixed.

That order matters because:

- key turns are chosen from the finalized baseline
- probe turns are chosen from the same baseline
- any needed conflict resolution should happen before the QA set is frozen

### Current Planned MCQ Families

The current design discussion supports at least two families.

#### 1. Recall MCQs

These test whether the model still remembers the content itself.

For both `key_turns` and `probe_turns`, we want:

- whole-turn recall questions
- slot-level recall questions for each sensitive field

Whole-turn questions test whether the overall event or request is remembered.

Slot-level questions test whether a specific sensitive field is remembered, which makes it possible to compare which types of information are easier to retain or forget.

#### 2. Application / Reasoning MCQs

These test whether the model can still **use** the remembered content to answer a new question, recommendation request, or planning problem.

This mirrors the more reasoning-heavy QA patterns in the original `prepare_qa.py`, especially:

- recommendation-style questions
- generalization from earlier reasons or preferences to a new situation

These questions are harder than pure recall because the model must apply earlier information, not merely restate it.

### MCQ Structure

A likely default structure is:

- `A`: correct answer
- `B`: distractor drawn from another turn in the same sample
- `C`: same-type but wrong detail
- `D`: a natural unavailable / ask-again response

The `D` option should stay natural rather than sounding like a policy classifier. For example:

- `I don't have that detail available anymore. Could you share it again?`

This makes the restricted-world answers feel more like realistic assistant behavior.

The same MCQ options should ideally be reused across:

- baseline
- `no_store`
- `forget`
- `no_use`

with only the expected answer changing by world and by test period.

## Why This Design

This design keeps the smallest possible difference between conditions:

- baseline world is fixed
- key/probe selection is fixed
- QA is fixed
- only the inserted memory-control turns vary

That makes it easier to study:

- key-to-instruction gap
- instruction-to-test gap
- whether unrelated recall utility remains intact
- whether memory can still be applied to harder reasoning tasks
