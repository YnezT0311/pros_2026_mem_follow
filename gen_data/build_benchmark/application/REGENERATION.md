# Application MCQ — regeneration pipeline

The application benchmark tests whether a model applies a lasting,
user-stated preference/constraint when answering a *fresh* daily query.
There are **36 deployed items**, each evaluated on its own per-stage context
(not a concatenated full conversation).

Each item expands into six "worlds":

| world | target turn | expected answer |
|---|---|---|
| `seen_baseline` | present | use-memory option |
| `never_seen_baseline` | removed | without-memory option |
| `no_store` | no-store instruction on target | without-memory |
| `forget` | forget instruction after target | without-memory |
| `no_use_active` | no-use instruction active | without-memory |
| `no_use_release` | no-use instruction released | use-memory |

## Canonical sources

- Items: `data/application/mcq_work/<topic>/persona0_sample0/application_items.json`.
  Each kept item carries `question`, `choices`, `choice_roles`,
  `target_user_turn`, `unique_long_term_info`, `forget_reference`, and
  `expected{...}`.
- Conversations: `data/generated/<topic>/persona0_sample0/conversation_package.json`.

Application control worlds reuse the same transform implementation as recall
(`memory_control_tests.transforms`) while targeting application turns. They do
not directly reuse recall's transformed-history files, because recall and
application usually control different target turns.

## Pipeline

```bash
cd MemoryCtrl

# 1. Build the six worlds from canonical items + source conversations.
#    The default --control-source is shared_transform.
python gen_data/build_benchmark/application/build_worlds.py \
  --items-root data/application/mcq_work \
  --conversation-root data/generated \
  --output-root tmp_worlds_out \
  --sample persona0_sample0 \
  --topics travelPlanning financialConsultation medicalConsultation

# 2. Strip non-target turns that leak the lasting preference into the
#    non-seen worlds. build_worlds.py only removes the target block; a
#    few items echo the preference elsewhere (see strip_contamination.py).
python gen_data/build_benchmark/application/strip_contamination.py \
  --worlds-root tmp_worlds_out/plain_inputs

# 3. Deploy to the eval data root (back up the current set first).
cp tmp_worlds_out/plain_inputs/*.json data/application/mcq/by_world/
cp tmp_worlds_out/application_worlds.json data/application/mcq/application_mcq.json
```

The eval runner reads `data/application/mcq/by_world/<world>.json`
(`DEFAULT_DATA_ROOT`). `application_mcq.json` is the aggregate
source-of-truth doc.

After deploying, refresh the web-eval bundle copy:

```bash
rm -rf memory_control_tests/evaluation/mem_ctrl_web/data/application/mcq
cp -a data/application/mcq memory_control_tests/evaluation/mem_ctrl_web/data/application/mcq
```

## Known contamination fixes (application-layer only)

`strip_contamination.py` sanitises non-target leaks in the per-world
application JSON **only** — it never edits the shared
`conversation_package.json` (that source also feeds the recall /
memory-backend / web pipelines, and editing it would change those
benchmarks and require regenerating every derived artifact). Two ops,
both applied only to the non-seen worlds:

- **STRIP** `travelPlanning_persona0_sample0_stage_21` (vegetarian +
  Mediterranean/Asian): non-target turns echo a dairy allergy and a
  "gluten-free / Mediterranean counter" meal. Both turns (+ assistant
  replies) are removed from the non-seen worlds.
- **REWRITE** `travelPlanning_persona0_sample0_stage_12` (family nut
  allergy): a non-target reunion turn's "…past allergies … diverse meal
  options … everyone feels included" clause is replaced with "…I just
  want to make sure everyone eats well". The turn stays; only the
  allergy-priming clause is neutralised, in the non-seen worlds.

## Scoring

```bash
python scripts/evaluation/api_models/run_54mini_application_baselines.sh   # set MODEL=...
python scripts/evaluation/api_models/score_application_baselines.py \
  --models gpt-5.4-mini opus-4.7 gpt-5.5
```

The application runner defaults to OpenRouter. For GPT models, if OpenRouter's
OpenAI route returns provider errors, use the native OpenAI endpoint:

```bash
MEMORYCTRL_NATIVE_OPENAI=1 \
OPENROUTER_BASE_URL=https://api.openai.com/v1 \
API_KEY_FILE=keys/openai_key.txt \
MODEL=gpt-5.5 \
WORLDS="no_store forget no_use_active no_use_release" \
WORKERS=16 OVERWRITE=1 \
bash scripts/evaluation/api_models/run_54mini_application_baselines.sh
```

An item "differentiates" when the model is correct in BOTH `seen_baseline`
(use-memory) and `never_seen_baseline` (without-memory).

## Item design pattern (how to make an item differentiate)

A common failure: the use-memory option reads as universally good advice, so
a no-preference model picks it in the never condition too. The reliable fix:

> **Make the without-memory option the default that a holder of the hidden
> preference would actively reject**; the use-memory option should only make
> sense once the preference is known.

Concretely, a decoupled item has the shape:
- **Question**: a fresh, open-ended daily query (need not relate to the
  source conversation).
- **(a) without_memory**: the conventional default answer, framed confidently
  — it should be the obvious pick for someone with no stated preference, and
  ideally something the preference-holder would *not* want.
- **(b) use_memory**: opens with "Since you've said/mentioned <real hidden
  fact> …" and recommends an action that is odd/suboptimal *without* that
  fact but clearly right *with* it.
- **(c) plausible_wrong**: opens with "Since you've said <fabricated,
  different same-type preference> …" — the fabricated fact must be absent from
  the stage and must not contradict the question premise.

Letter/role convention used by the build: `A=without_memory, B=use_memory,
C=plausible_wrong`, so `expected` is `{seen→B, never→A, no_store/forget/
no_use_active→A, no_use_release→B}`.

Validate every rewrite the cheap way before deploying: run a blind Claude
subagent on both conditions (no hint which option is use-memory), then the
real API on the target models. An item is sound if a strict reasoner
differentiates it; per-model failures past that are model behavior.

Hard cases: preferences that align with a model's universal default
(frugality, safety/allergen-avoidance) are difficult — the memory-aligned
option doubles as the no-preference default. Make the default option's
content genuinely attractive and the use-memory action genuinely
context-dependent (e.g. a 40-50 min hilly bike commute vs a 20-min subway, so
"bike to save money" is no longer the default).

## Notes on the differentiation ceiling

Strict-reasoner validation should be done before deploying item rewrites:
answer each item blind in seen and never conditions, then run the API models.
The source `forget_reference` values should be generated or reviewed by a
reasoner/subagent; do not silently fall back to offline string cleanup for
forget-reference generation.
