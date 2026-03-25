# Memory Control Tests

## Goal

This folder contains a new baseline-first pipeline for memory-control evaluation without modifying the legacy `privacy_test/` code.

The workflow is:

1. start from `data/output/`
2. build a stricter `data/baseline/`
3. fix key turns and protected recall-utility turns offline
4. run evaluation-time conversation transforms for:
   - `no_store`
   - `forget`
   - `no_use`

The main design choice is that we do **not** materialize a separate static world for every gap condition. Instead, we keep one baseline conversation and insert instruction/reply turns on the fly during evaluation.

## Baseline Construction

`build_baseline.py` copies each conversation file from `data/output/` into `data/baseline/` and writes a sidecar spec file:

- `conversation_...json`
- `conversation_....memory_control.json`

The sidecar spec records:

- all `Initial Stage` interaction candidates
- suggested `key_turns`
- suggested `protected_probe_turns`
- future-mention evidence for each candidate
- a `baseline_resolution` summary describing whether the selected keys are already conflict-free or would need later targeted revision

## Candidate Selection

Selection is currently restricted to `Initial Stage` interaction turns.

The builder:

- reads `Conversation History Initial Stage`
- maps interaction timestamps to their `Conversation Initial Stage` blocks
- searches later-stage conversation lines for likely future mentions
- penalizes earlier exact duplicates of the same sensitive value
- prefers the last exact occurrence when a sensitive value is repeated

Right now the duplicate-sensitive rule is a heuristic, not a hard filter. This keeps the builder usable even when there are not enough perfectly clean candidates.

## Conflict Detection

The offline conflict check intentionally uses only information already present in the generated data:

- `[Task Goal]`
- `[Context Can Add]`
- `[Sensitive Info]`
- the source `[Prev Event]`

This is important because the current history pipeline already exposes a traceable chain and explicit sensitive anchors. We do not need to guess conflicts from scratch.

Future conflicts are now detected with hard rules instead of line-level scoring:

- exact reuse of normalized `[Sensitive Info]` values in later interaction history
- exact reuse of normalized `[Context Can Add]` keys in later interaction history
- later events or interactions that belong to the candidate source event's descendant chain
- a boolean high-similarity check on later `[Task Goal]` text

If a candidate has no later-stage hits under these rules, it is treated as baseline-safe. If it has hits, the spec marks it as `needs_revision`.

When the selected key set is already conflict-free, the baseline conversation can be copied directly from `data/output/` without editing. That is the preferred path because it keeps the baseline world maximally close to the original generated conversation.

## World Semantics

All three test settings reuse the same baseline and the same QA.

### `no_store`

- modify the chosen key turn itself
- append a user suffix such as “do not remember this later”
- prepend a templated assistant acknowledgement to the same reply

### `forget`

- append a standalone instruction/reply turn in a later stage
- default use case is the end of `Conversation Early Stage`

### `no_use`

- append a standalone restriction turn in a later stage
- optionally append a later release turn

The concrete instruction/reply pools live in `templates.json`.

The template pool uses direct instruction wording. For `no_store`, templates support either front-loaded insertion or end-loaded insertion inside the key turn.

## Evaluation-Time Transform

`transforms.py` provides:

- `apply_no_store(...)`
- `apply_forget(...)`
- `apply_no_use(...)`
- `build_context_messages(...)`

This lets evaluation load a baseline conversation, insert control turns for the desired gap condition, and then build the exact conversation context to feed to the model.

## Why This Design

This design keeps the smallest possible difference between conditions:

- baseline is fixed
- key/probe selection is fixed
- QA is fixed
- only the inserted memory-control turns vary

That makes it easier to study:

- key-to-instruction gap
- instruction-to-test gap
- whether unrelated recall utility remains intact
