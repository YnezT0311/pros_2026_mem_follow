# MemoryCtrl Web Evaluation Bundle

This folder is the self-contained web-evaluation bundle for local ChatGPT and Claude testing.

The default web protocol is stage-local: each session feeds one `Conversation Stage NN`, asks that stage's MCQs, then clears memory and deletes the conversation before the next stage. The legacy full-topic concat protocol is still available through the `*_long_context_*` scripts.

Layout:

```text
mem_ctrl_web/
  data/
    benchmark_work_v2/
      baseline/<topic>/<sample>/conversation_package.json
    recall/mcq/
      <topic>/<sample>/whole_recall.json
      <topic>/<sample>/slot_recall.json
      <topic>/<sample>/stage_XX/whole_recall.json
      <topic>/<sample>/stage_XX/slot_recall.json
    application/mcq/
      application_mcq.json
      by_world/
        seen_baseline.json
        never_seen_baseline.json
        no_store.json
        forget.json
        no_use_active.json
        no_use_release.json
    generated/
      <topic>/<sample>/conversation_package.json
  chatgpt/
  claude/
```

Recall data is bundled under both `mem_ctrl_web/data/benchmark_work_v2` and
`mem_ctrl_web/data/recall/mcq`, depending on the runner path. The stage
subdirectories are inspection-friendly copies; the web runners can also read the
top-level `whole_recall.json` and `slot_recall.json` files directly. The
matching source conversations live in `mem_ctrl_web/data/generated`.

The application data source is `mem_ctrl_web/data/application/mcq`. Application
worlds are pre-rendered under `by_world/`; the web runners do not remove target
turns on the fly. `seen_baseline` and `never_seen_baseline` are separate stored
world files, and the same is true for `no_store`, `forget`, `no_use_active`, and
`no_use_release`. The control worlds are built with the same transform logic as
recall, but target the application MCQ turn rather than recall's key turn.

## Web Protocols

Recall web eval is stage-local by default. For each topic/persona/world, the
runner opens one browser session per stage with active recall MCQs, sends the
stage's user turns, asks that stage's recall questions, then deletes the chat
before the next stage. Memory is cleared before and after each completed
topic/persona/world session.

Application web eval uses the pre-rendered application world records directly.
Each application MCQ is one browser session: the runner sends the user turns
from that record's `context_messages`, then asks that record's application MCQ.
This keeps web input aligned with the API baseline inputs and avoids rebuilding
seen/unseen/forget/no-use contexts inside the web runner.

Application cleanup follows the same session cleanup discipline as recall:

- Before a fresh application session, the runner clears saved Memory and deletes
  existing chats/conversations.
- After a completed application session, the runner clears saved Memory again
  and deletes the chat/conversation before moving to the next MCQ.
- If a run is interrupted, rerunning the same command resumes from the
  session trace and does not manually clear Memory/delete the active
  conversation before resuming that incomplete session.

This means application does not rely on memory carrying across MCQs; every MCQ
is isolated unless it is an intentional resume of an incomplete session.

## Quick Test

The quick test runs only `travelPlanning` / `Conversation Stage 01`, asks only Stage 01 `whole_recall` MCQs, and covers `baseline`, `no_store`, and `forget`.

Before running either web test:

- Turn on Memory in ChatGPT / Claude.
- Keep the browser sidebar open. The cleanup code needs the sidebar/chat controls to be visible.
- The script will first open the browser for one manual login step. After login finishes, the second browser pass starts the test automatically.
- For a fresh session, the automatic test pass clears Memory and deletes the current conversation before running.
- If a previous run was interrupted, rerun the same command without manually clearing Memory or deleting the conversation. The runner resumes from `session_trace.jsonl`, skips completed history turns / MCQs, and continues in the existing chat.
- Memory and conversation cleanup happens only after a topic/persona/world session finishes successfully.

ChatGPT:

```bash
cd mem_ctrl_web/chatgpt

python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

bash run_test.sh
```

Claude:

```bash
cd mem_ctrl_web/claude

python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

bash run_test.sh
```

## Get The Data

If you already have this repository on the server, copy the whole bundle:

```bash
scp -r USER@SERVER: path/to/MemoryCtrl/memory_control_tests/evaluation/mem_ctrl_web .
```

If the bundle is committed to GitHub, download this folder:

```text
memory_control_tests/evaluation/mem_ctrl_web
```

After either route, your local folder should contain:

```text
mem_ctrl_web/data/recall/mcq
mem_ctrl_web/data/benchmark_work_v2
mem_ctrl_web/data/application/mcq
mem_ctrl_web/data/generated
mem_ctrl_web/chatgpt
mem_ctrl_web/claude
```

## Full ChatGPT Eval

```bash
cd mem_ctrl_web/chatgpt
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

TOPICS="travelPlanning financialConsultation medicalConsultation" WORLDS="baseline no_store forget" bash run_chatgpt_eval.sh
```

## Application ChatGPT Eval

```bash
cd mem_ctrl_web/chatgpt
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

TOPICS="travelPlanning financialConsultation medicalConsultation" \
  WORLDS="seen_baseline never_seen_baseline no_store forget no_use_active no_use_release" \
  bash run_chatgpt_application_eval.sh
```

Useful application knobs:

```bash
# One topic or one world
TOPIC=travelPlanning WORLDS="forget" bash run_chatgpt_application_eval.sh

# Debug only a few application MCQs
LIMIT=3 TOPIC=travelPlanning WORLDS="seen_baseline never_seen_baseline" \
  bash run_chatgpt_application_eval.sh

# Put outputs somewhere explicit
RESULTS=./application_results_20260531 bash run_chatgpt_application_eval.sh
```

## Full Claude Eval

```bash
cd mem_ctrl_web/claude
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

TOPICS="travelPlanning financialConsultation medicalConsultation" WORLDS="baseline no_store forget" bash run_claude_eval.sh
```

## Application Claude Eval

```bash
cd mem_ctrl_web/claude
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

TOPICS="travelPlanning financialConsultation medicalConsultation" \
  WORLDS="seen_baseline never_seen_baseline no_store forget no_use_active no_use_release" \
  bash run_claude_application_eval.sh
```

The same `TOPIC`, `TOPICS`, `WORLDS`, `LIMIT`, `RESULTS`, `DATA`, and
`SESSION_DIR` environment variables work for Claude.

Recall defaults:

```text
ChatGPT run script: ../data/benchmark_work_v2
Claude run script:  ../data/recall/mcq
```

For application, use the dedicated application runner:

```text
ChatGPT application script: bash run_chatgpt_application_eval.sh
Claude application script:  bash run_claude_application_eval.sh
```

The application runner default data directory is `../data/application/mcq`.
Override with `DATA=/path/to/recall/mcq` or `DATA=/path/to/application/mcq` if needed. Use `run_chatgpt_long_context_eval.sh`, `run_claude_long_context_eval.sh`, or pass `--long_context` for the old full-topic concat protocol.

Full eval defaults to all samples, because `LIMIT=0` means no sample limit. For debugging, add `LIMIT=1`. To run one topic only, use `TOPIC=financialConsultation` instead of `TOPICS="..."`.

Resume behavior:

- Completed sessions are skipped.
- Interrupted sessions resume from their existing `session_trace.jsonl`.
- Do not manually clear Memory or delete the active conversation before resuming an interrupted session.
- The current bundle has one sample/persona per topic. In stage-local mode, each topic/persona/world is split into one session per stage that has active MCQs.
