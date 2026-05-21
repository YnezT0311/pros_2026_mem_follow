# ChatGPT Web Evaluation

Run from this folder after downloading `mem_ctrl_web/`.

```bash
cd mem_ctrl_web/chatgpt
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m patchright install chrome

TOPIC=financialConsultation LIMIT=1 WORLDS="baseline no_store forget" bash run_chatgpt_eval.sh
```

The runner reads data from `../data/benchmark_work_v2` by default and writes outputs under `./results/`.

Quick Stage 01 smoke test:

```bash
bash run_test.sh
```

This runs only `travelPlanning` / `Conversation Stage 01`, asks only Stage 01 `whole_recall` MCQs, and covers only `baseline` plus `no_store`.

The pinned web automation dependencies are:

```text
greenlet==3.4.0
patchright==1.58.2
playwright==1.58.0
pyee==13.0.1
typing_extensions==4.15.0
```
