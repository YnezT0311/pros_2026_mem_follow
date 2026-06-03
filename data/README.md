# MemoryCtrl data layout & per-world data sources

This explains where each evaluation reads its data, how each control world's
control instruction is produced, and **which file is authoritative when you
update data**. It exists because the data flow is non-obvious and has bitten us:
the same control world is fed from *different* files depending on which runner
you launch, and several derived copies can silently drift from the master.

## 1. The one source of truth

```
data/generated/<topic>/persona0_sample0/conversation_package.json   <-- MASTER conversation
```

The master is the clean, control-free conversation (all `Conversation Stage NN`
arrays). **Every** other artifact (recall sidecar, recall MCQs, application
MCQs, all the web bundles) is derived from it. Edit the master first, then
propagate (see §5).

Recall also has two *source* artifacts that live next to the old work dir and
are referenced by pointer (NOT regenerated per run):

```
data/recall/mcq_work/<topic>/persona0_sample0/memory_targets.json   <-- recall SIDECAR (key_turns, each with user_turn/timestamp/key_phrase)
data/recall/mcq_work/<topic>/persona0_sample0/mcq_questions.json    <-- recall rendered MCQ source
```

`memory_targets.json` (the sidecar) must stay consistent with the master:
the transforms match each key turn by its `user_turn` text against the master
conversation. If the master is reworded but the sidecar `user_turn` is not, the
match fails and that turn gets **no control instruction** (this is exactly the
travelPlanning stage_13/stage_20 drift bug we hit).

## 2. Derived artifacts (regenerable — do NOT hand-edit)

- `*.transformed_history.json` — the conversation with the control instruction
  inserted, per (world, stage). **Regenerable caches.** gitignored under `data/`.
  The runner regenerates them via a freshness guard; delete them to force a
  clean rebuild. Never treat them as a source of truth.
- `data/recall/mcq/<topic>/persona0_sample0/<stage>/whole_recall.json` +
  `slot_recall.json` — the per-stage rendered recall MCQs (current layout).
- `data/application/mcq/by_world/<world>.json` — pre-rendered application MCQ
  records (context baked in, built with `application_shared_stage_v1` /
  `shared_transform`).

## 3. Who reads what (CRITICAL — pick the right data dir)

| Runner | Reads | Layout | forget transform |
|---|---|---|---|
| `scripts/evaluation/memory_agents/run_54mini_memory_method.py` (local memory-agent + api) | `data/recall/mcq/<topic>/persona0_sample0/<stage>/whole_recall.json` | per-stage subdir | `apply_stage_local_forget` (per-stage) |
| web full eval `mem_ctrl_web/chatgpt|claude/run_*_eval.sh` | **`mem_ctrl_web/data/benchmark_work_v2`** (default `DATA`) | flat, whole-history file + runtime per-stage regen | `apply_stage_local_forget` (per-stage, regenerated at run time) |
| web quick test `run_test.sh` | `mem_ctrl_web/data/recall/mcq` | flat | per-stage |
| retired `.sh` runners (memtree/memoryos/gpt4o/amem.sh…) | `data/recall/mcq_work/.../mcq_questions.json` | OLD flat (`persona0_sample0.<world>.stage_NN…`) | per-stage |

The web full eval (what produced the archived ChatGPT/Claude results) reads
**`benchmark_work_v2`**, which has its *own* nested copy of the master and
sidecar:

```
mem_ctrl_web/data/benchmark_work_v2/baseline/<topic>/persona0_sample0/conversation_package.json  <-- web master copy
mem_ctrl_web/data/benchmark_work_v2/<topic>/persona0_sample0/memory_targets.json                 <-- web sidecar copy
mem_ctrl_web/data/benchmark_work_v2/<topic>/persona0_sample0/mcq_questions.json                  <-- web rendered (source_conversation/source_sidecar pointers resolve relative to the bundle)
```

These web copies are a **derived export** and can drift from `data/`. As long as
each copy's master and sidecar are internally consistent (key turns match), the
transforms work — even if the text is older than `data/`. They were verified
consistent (key match 30/30 per topic) but carry pre-contamination wording for
travelPlanning stage_13/20.

## 4. Per control-world: where the instruction comes from

| World | Transform | Instruction | Placement | Coverage |
|---|---|---|---|---|
| `baseline` | none | none | — | uses master directly |
| `no_store` | `apply_no_store` | suffix appended to the **key turn** (`templates.json["no_store"]`) | same turn as the key turn | one per key turn |
| `forget` | `apply_stage_local_forget` (per-stage) or `apply_randomized_forget` (whole-history) | separate User+Assistant turn (`templates.json["forget"]`) | after the key turn, same stage | one per key turn |
| `no_use_active` / `no_use_release` | `apply_no_use` | restrict (+ release) instruction (`templates.json["no_use"]`) | median of key+probe turns, per stage | one per key-turn stage; `release` adds a later release turn |

Matching is by `user_turn` text + `stage_id`/`timestamp` (see `_find_key_position`
in `transforms.py`). Drift in the sidecar `user_turn` breaks matching → missing
instruction.

Known by-design: `no_use` inserts once per **stage that has a key turn**, so a
stage with no key turn legitimately has no `no_use` instruction (e.g.
travelPlanning stage_02/stage_04). Verified full coverage of key-turn stages
against the archived web logs.

## 5. When you UPDATE data — update order (authoritative = master)

1. Edit **`data/generated/<topic>/persona0_sample0/conversation_package.json`** (master). This is the only authoritative edit point.
2. Re-derive / re-align the recall **sidecar** `data/recall/mcq_work/<topic>/persona0_sample0/memory_targets.json` so every `key_turns[].user_turn` still matches the master verbatim.
3. Re-render the recall MCQs (`data/recall/mcq/<topic>/persona0_sample0/<stage>/whole_recall.json` + `slot_recall.json`) — their `items[].user_turn` is metadata, but keep it aligned.
4. Rebuild application `data/application/mcq/by_world/*.json` from the master.
5. **Delete all `*.transformed_history.json` caches** under `data/recall/` so the runner regenerates them from the new master (the freshness guard catches forget/no_store/no_use count changes, but NOT silent text drift — so delete to be safe).
6. Re-export the web bundle `mem_ctrl_web/data/` (both `benchmark_work_v2` and `recall/mcq`, plus `application/mcq/by_world`) from the updated `data/`, then commit + push so web downloads stay in sync.

Exception (application-only): `gen_data/build_benchmark/application/strip_contamination.py`
sanitises leaks **only** in the per-world application JSON and must never touch
the master. If you see the master reworded, it was a master-level edit and must
be propagated through steps 2–6.

## 6. Gotchas this README exists to prevent

- "I fixed forget but the web is still broken" → the web full eval reads
  `benchmark_work_v2`, not `data/recall/mcq`. Fix/regenerate the layout the
  runner you launch actually reads.
- "The transformed_history says 0 insertions" → stale cache; the runner
  regenerates it, or delete it. Judge correctness from `add_traces` (memory
  agent) / `session_trace.jsonl` `history_turn.user_input` (web), not the cache.
- "Only stage_01 has a forget instruction" → was a real bug in
  `apply_stage_local_forget` (full key_turns zipped against stage-filtered refs);
  fixed. If you see this pattern again, check the zip alignment.
- Verifying control-instruction presence in logs: match against the **full**
  `templates.json` template set, not hand-picked substrings (a short list gives
  false negatives).
