# Application Benchmark Data

Application memory-control artifacts live here. Unlike recall MCQs, application
MCQs test whether a prior user constraint or preference changes a later
practical answer.

Current layout:

```text
mcq_work/
  SUMMARY.md
  <topic>/<sample>/
    application_items.json

mcq/
  application_mcq.json
  by_world/
    never_seen_baseline.json
    seen_baseline.json
    no_store.json
    forget.json
    no_use_active.json
    no_use_release.json
```

Do not mix application-control questions into `data/recall/`; recall and
application tasks use different ask points, worlds, and scoring.

The main application setting is per-stage source context. It does not
concatenate all stages into a full-history prompt. Each application world record
embeds its own `messages` and `context_messages`, rendered from exactly one
`Conversation Stage XX` in
`data/generated/<topic>/<sample>/conversation_package.json`.

For control worlds, application uses the same transform logic as recall
(`memory_control_tests.transforms`) but with application-specific target turns:

- `no_store`: adds the no-store instruction to the application target user turn.
- `forget`: inserts a forget user/assistant pair using the item's
  `forget_reference`.
- `no_use_active`: inserts a no-use restriction after the application target.
- `no_use_release`: inserts the same restriction and then a release before the
  application MCQ. If recall's stricter stage-local release gap cannot fit
  because the target is near the end of the stage, application appends the
  release pair at stage end so the final MCQ is asked after release.

Application does **not** directly reuse recall's existing transformed-history
files, because recall and application often target different turns. It reuses
the transform logic and instruction style, not the precomputed recall targets.

Only exportable source items are rendered into `mcq/by_world/` and
the aggregate `mcq/application_mcq.json`.

Current deployed coverage:

- 36 application MCQs
- 6 control worlds
- 216 total world records

## Running API Application Eval

Plain API evaluation reads the pre-rendered world files under
`data/application/mcq/by_world/`.

```bash
cd MemoryCtrl

MODEL=gpt-5.4-mini \
WORLDS="seen_baseline never_seen_baseline no_store forget no_use_active no_use_release" \
WORKERS=24 OVERWRITE=1 \
bash scripts/evaluation/api_models/run_54mini_application_baselines.sh
```

Run the same command with `MODEL=opus-4.7` and `MODEL=gpt-5.5` for the other
baseline API models. Use `OVERWRITE=1` only when you intentionally want to
replace an existing result file.

If OpenRouter's OpenAI route is unavailable, run GPT models through the native
OpenAI endpoint:

```bash
MEMORYCTRL_NATIVE_OPENAI=1 \
OPENROUTER_BASE_URL=https://api.openai.com/v1 \
API_KEY_FILE=keys/openai_key.txt \
MODEL=gpt-5.4-mini \
WORLDS="no_store forget no_use_active no_use_release" \
WORKERS=16 OVERWRITE=1 \
bash scripts/evaluation/api_models/run_54mini_application_baselines.sh
```
