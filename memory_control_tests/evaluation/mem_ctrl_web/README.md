# MemoryCtrl Web Evaluation Bundle

This folder is the self-contained web-evaluation bundle for local ChatGPT and Claude testing.

Layout:

```text
mem_ctrl_web/
  data/
    benchmark_work_v2/
      baseline/<topic>/<sample>/conversation_package.json
      <topic>/<sample>/mcq_questions.json
      <topic>/<sample>/memory_targets.json
      <topic>/<sample>/<sample>.no_store.transformed_history.json
      <topic>/<sample>/<sample>.forget.transformed_history.json
  chatgpt/
  claude/
```

The bundled `data/` is the minimal web-eval subset. It uses the same benchmark work directory shape as the API and memory-system evaluation.

## Quick Test

The quick test runs only `travelPlanning` / `Conversation Stage 01`, asks only Stage 01 `whole_recall` MCQs, and covers only `baseline` plus `no_store`.

Before running either web test:

- Turn on Memory in ChatGPT / Claude.
- Keep the browser sidebar open. The cleanup code needs the sidebar/chat controls to be visible.
- The script will first open the browser for one manual login step. After login finishes, the second browser pass starts the test automatically.
- The automatic test pass clears Memory and deletes the current conversation before running, then cleans up after each test session.

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
mem_ctrl_web/data/benchmark_work_v2
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

Both runners default to:

```text
../data/benchmark_work_v2
```

Override with `DATA=/path/to/benchmark_work_v2` if needed.

Full eval defaults to all samples, because `LIMIT=0` means no sample limit. For debugging, add `LIMIT=1`. To run one topic only, use `TOPIC=financialConsultation` instead of `TOPICS="..."`.
