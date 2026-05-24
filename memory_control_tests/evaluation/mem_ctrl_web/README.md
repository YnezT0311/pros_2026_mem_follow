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
    generated/
      <topic>/<sample>/conversation_package.json
  chatgpt/
  claude/
```

The default data source is the bundled `mem_ctrl_web/data/recall/mcq`. The stage subdirectories are inspection-friendly copies; the web runners can also read the top-level `whole_recall.json` and `slot_recall.json` files directly. The matching source conversations live in `mem_ctrl_web/data/generated`.

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

Both default runners now use:

```text
../data/recall/mcq
```

Override with `DATA=/path/to/recall/mcq` if needed. Use `run_chatgpt_long_context_eval.sh`, `run_claude_long_context_eval.sh`, or pass `--long_context` for the old full-topic concat protocol.

Full eval defaults to all samples, because `LIMIT=0` means no sample limit. For debugging, add `LIMIT=1`. To run one topic only, use `TOPIC=financialConsultation` instead of `TOPICS="..."`.

Resume behavior:

- Completed sessions are skipped.
- Interrupted sessions resume from their existing `session_trace.jsonl`.
- Do not manually clear Memory or delete the active conversation before resuming an interrupted session.
- The current bundle has one sample/persona per topic. In stage-local mode, each topic/persona/world is split into one session per stage that has active MCQs.
