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

## Get The Data

If you already have this repository on the server, copy the whole bundle:

```bash
scp -r USER@SERVER:/mnt/yao_data/proj_2026_agent/MemoryCtrl/memory_control_tests/evaluation/mem_ctrl_web .
```

If the bundle is committed to GitHub, either download the repository zip from GitHub and keep this folder:

```text
memory_control_tests/evaluation/mem_ctrl_web
```

or use sparse checkout:

```bash
git clone --filter=blob:none --sparse <GITHUB_REPO_URL> memctrl-web-download
cd memctrl-web-download
git sparse-checkout set memory_control_tests/evaluation/mem_ctrl_web
cp -R memory_control_tests/evaluation/mem_ctrl_web ../mem_ctrl_web
```

After either route, your local folder should contain:

```text
mem_ctrl_web/data/benchmark_work_v2
mem_ctrl_web/chatgpt
mem_ctrl_web/claude
```

## Run ChatGPT

```bash
cd mem_ctrl_web/chatgpt
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

TOPIC=financialConsultation LIMIT=1 WORLDS="baseline no_store forget" bash run_chatgpt_eval.sh
```

Quick Stage 01 smoke test:

```bash
bash run_test.sh
```

This feeds only `travelPlanning` / `Conversation Stage 01` and asks only Stage 01 `whole_recall` MCQs for `baseline` and `no_store`.

## Run Claude

```bash
cd mem_ctrl_web/claude
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

TOPIC=financialConsultation LIMIT=1 WORLDS="baseline no_store forget" bash run_claude_eval.sh
```

Quick Stage 01 smoke test:

```bash
bash run_test.sh
```

This feeds only `travelPlanning` / `Conversation Stage 01` and asks only Stage 01 `whole_recall` MCQs for `baseline` and `no_store`.

Both runners default to:

```text
../data/benchmark_work_v2
```

Override with `DATA=/path/to/benchmark_work_v2` if needed.
