#!/usr/bin/env bash
# Overnight orchestration:
#   1. Run Option-2 LangMem baseline for persona 2 + 3
#   2. Compute mean whole_recall across all 4 baseline personas (persona0 = old smoke, persona1-3 = Option 2)
#   3. If mean > 0.70, also run no_store and forget batteries
#   4. Always regenerate the report at the end

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="logs/langmem_overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "logs in $LOG_DIR"

echo "=== [1/4] baseline persona2+3 ==="
conda run -n langmem311 --no-capture-output python scripts/rerun_langmem_battery.py \
    --workers 2 --worlds baseline --skip_baseline_smoked \
    > "$LOG_DIR/baseline.log" 2>&1
echo "baseline complete"

# Compute mean whole_recall across all 4 baseline personas (Late stage only — primary cell)
mean_whole=$(python -c "
import json
from pathlib import Path
from collections import Counter
root = Path('eval_results/travelPlanning/baseline/gpt-5.4-mini+LangMem')
totals = []
for p in range(4):
    # use Late (primary) result; filename has no .early/.intermediate suffix
    f = root / f'conversation_travelPlanning_persona{p}_sample0.baseline.langmem_retrieval_eval_gpt_5_4_mini.json'
    if not f.exists():
        # try the gpt-5.4-mini (hyphens) naming as fallback
        f = root / f'conversation_travelPlanning_persona{p}_sample0.baseline.langmem_retrieval_eval_gpt-5.4-mini.json'
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    w = d.get('whole_recall_results', [])
    if w:
        rate = sum(1 for it in w if it.get('predicted_answer_type')=='remember_correct') / len(w)
        totals.append(rate)
if totals:
    print(f'{sum(totals)/len(totals):.4f}')
else:
    print('0.0000')
")
echo "mean baseline whole_recall (Late stage) across 4 personas = $mean_whole"
echo "$mean_whole" > "$LOG_DIR/mean_whole.txt"

# Decision threshold
threshold=0.70
proceed=$(python -c "print(1 if float('$mean_whole') > float('$threshold') else 0)")

if [ "$proceed" = "1" ]; then
    echo "=== mean $mean_whole > $threshold — proceeding to no_store + forget ==="
    echo "=== [2/4] no_store all personas ==="
    conda run -n langmem311 --no-capture-output python scripts/rerun_langmem_battery.py \
        --workers 4 --worlds no_store \
        > "$LOG_DIR/no_store.log" 2>&1
    echo "no_store complete"

    echo "=== [3/4] forget all personas ==="
    conda run -n langmem311 --no-capture-output python scripts/rerun_langmem_battery.py \
        --workers 4 --worlds forget \
        > "$LOG_DIR/forget.log" 2>&1
    echo "forget complete"
else
    echo "=== mean $mean_whole <= $threshold — STOPPING ==="
    echo "Will not run no_store/forget. Investigate baseline LangMem results."
fi

echo "=== [4/4] regenerate report ==="
python -m memory_control_tests.analysis.build_report > "$LOG_DIR/report.log" 2>&1
echo "report at eval_results/travelPlanning/report.html"

echo "=== DONE ==="
echo "logs: $LOG_DIR"
