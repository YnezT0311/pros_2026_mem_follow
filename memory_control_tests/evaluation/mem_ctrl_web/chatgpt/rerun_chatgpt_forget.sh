#!/bin/bash
# =============================================================================
# rerun_chatgpt_forget.sh  —  foolproof re-run of ChatGPT web forget (recall)
# -----------------------------------------------------------------------------
# Background: the earlier ChatGPT forget run was INVALID because the per-stage
# forget directive was never delivered to the model. Root cause (same shape as
# the Claude no_store bug): the runner loads a *pre-existing* combined
# transformed-history file DIRECTLY and only regenerates it when it is missing
# (see evaluate_chatgpt_web.py:_load_conv). A STALE
#   travelPlanning/persona0_sample0/persona0_sample0.forget.transformed_history.json
# left on disk predated the forget-directive fix and carried the directive in
# ONLY stage_01; every other control stage ran directive-less, so the model had
# nothing to forget and "violation" was massively inflated.
#
# This script removes the stale forget transformed-history file(s), REBUILDS the
# combined file from the committed source conversation + sidecar (memory_targets
# .json) using the SAME code path the runner would use lazily
# (shared.apply_world_transform, stage_id=""), VERIFIES every forget insertion's
# directive line is actually present in its target stage, and only then re-runs.
#
# Just run it:
#     ./rerun_chatgpt_forget.sh
#
# Optional env:
#     SKIP_SYNC=1   skip git clean+pull (only delete + rebuild + verify + run)
#     RESULTS=...    results dir (forwarded to run_chatgpt_eval.sh)
#     TOPIC=...      topic (default travelPlanning)
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # .../mem_ctrl_web/chatgpt
WEB_DATA="$(cd "$SELF_DIR/../data" && pwd)"                         # .../mem_ctrl_web/data
MEMCTRL_ROOT="$(cd "$SELF_DIR/../../../.." && pwd)"                 # .../MemoryCtrl (for python import)
TOPIC="${TOPIC:-travelPlanning}"
DATA_DIR="$WEB_DATA/benchmark_work_v2"
PERSONA_DIR="$DATA_DIR/$TOPIC/persona0_sample0"
REPO_ROOT="$(git -C "$SELF_DIR" rev-parse --show-toplevel 2>/dev/null || true)"

echo "============================================"
echo " ChatGPT forget re-run ($TOPIC)"
echo " repo:      ${REPO_ROOT:-<not a git checkout>}"
echo " data dir:  $DATA_DIR"
echo " persona:   $PERSONA_DIR"
echo "============================================"

# ---------------------------------------------------------------------------
# 1) Pull corrected source conversation + sidecar from the shared branch so the
#    rebuild uses the fixed inputs. (The transformed-history file itself is
#    regenerated below, so it does not matter whether git restores a stale one.)
# ---------------------------------------------------------------------------
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: '$SELF_DIR' is not inside a git checkout, so auto pull is impossible."
    echo "       Sync the fix manually, then re-run with:  SKIP_SYNC=1 $0"
    exit 1
  fi
  echo ""
  echo "[1/4] git pull --ff-only ..."
  git -C "$REPO_ROOT" pull --ff-only
else
  echo ""
  echo "[1/4] SKIP_SYNC=1 -> skipping git pull"
fi

# ---------------------------------------------------------------------------
# 2) Remove the STALE forget transformed-history file(s) so they cannot be
#    reused. Both the combined file (what a normal run reads) and any per-stage
#    files are removed.
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Removing stale forget transformed-history file(s) ..."
shopt -s nullglob
removed=0
for f in "$PERSONA_DIR"/*.forget.transformed_history.json "$PERSONA_DIR"/*.forget.stage_*.transformed_history.json; do
  rm -f "$f"; echo "   removed: ${f#$PERSONA_DIR/}"; removed=$((removed+1))
done
shopt -u nullglob
[[ $removed -eq 0 ]] && echo "   (none present)"

# ---------------------------------------------------------------------------
# 3) REBUILD the combined forget transformed-history from source + sidecar using
#    the exact code path the runner uses lazily, then VERIFY every directive is
#    present in its target stage. Aborts loudly on any mismatch.
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Rebuilding + verifying forget transformed-history ..."
PYTHONPATH="$MEMCTRL_ROOT" python3 - "$DATA_DIR" "$TOPIC" "$PERSONA_DIR" <<'PY'
import json, sys
from pathlib import Path
from memory_control_tests.evaluation.shared import apply_world_transform

data_dir, topic, persona_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

def resolve_existing(path: str) -> Path:
    raw = Path(path)
    for cand in (raw, data_dir / raw):
        if cand.exists():
            return cand
    return raw

# mirror evaluate_chatgpt_web.py:_load_split_mcq_payloads (mcq_questions.json branch)
rendered = persona_dir / "mcq_questions.json"
if not rendered.exists():
    rendered = persona_dir / "whole_recall.json"
if not rendered.exists():
    sys.exit(f"FAIL: no mcq_questions.json / whole_recall.json under {persona_dir}")
r = json.loads(rendered.read_text(encoding="utf-8"))
if rendered.name == "whole_recall.json":
    src_conv_rel = r.get("source_conversation", ""); src_sidecar_rel = r.get("source_sidecar", "")
    items = r.get("items", [])
else:
    src_conv_rel = r.get("source_conversation", ""); src_sidecar_rel = r.get("source_sidecar", "")
    items = r.get("whole_recall_set", [])

src_path = resolve_existing(src_conv_rel)
sc_path  = resolve_existing(src_sidecar_rel)
if not src_path.exists(): sys.exit(f"FAIL: source conversation missing -> {src_conv_rel}")
if not sc_path.exists():  sys.exit(f"FAIL: sidecar missing -> {src_sidecar_rel}")
source_conv = json.loads(src_path.read_text(encoding="utf-8"))
sidecar     = json.loads(sc_path.read_text(encoding="utf-8"))

# mirror _target_references_from_items(items, "")
refs_by_ts = {}
for it in items:
    ts = str(it.get("timestamp", "")).strip()
    label = str(it.get("identifier_label", "")).strip()
    if label:
        refs_by_ts[ts] = f"the {label[0].lower() + label[1:]}"
    elif it.get("task_goal"):
        refs_by_ts[ts] = str(it.get("task_goal")).strip()
target_refs = [
    refs_by_ts.get(str(t.get("timestamp", "")).strip(), str(t.get("task_goal", "that earlier request")))
    for t in sidecar.get("key_turns", [])
]

transformed = apply_world_transform(source_conv, sidecar, "forget", target_refs, "all_stages", "", stage_id="")

out = persona_dir / f"persona0_sample0.forget.transformed_history.json"
out.write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"   rebuilt -> {out.name}")

# ---- verify: every forget insertion's directive line is present in its stage ----
md  = transformed.get("_memory_control_metadata", {})
ins = md.get("forget_insertions", [])
if not ins:
    sys.exit("FAIL: forget_insertions empty after rebuild (sidecar has no forget targets?)")

missing = []
for entry in ins:
    stage = entry.get("forget_stage", "")
    line  = (entry.get("forget_user_line") or "").strip()
    stage_text = " ".join(l for l in transformed.get(stage, []) if isinstance(l, str))
    if not line or line not in stage_text:
        missing.append(f"{stage}: {entry.get('key_timestamp','?')} -> directive line not found")

stages_with_directive = sorted({e.get("forget_stage", "") for e in ins})
if missing:
    print("FAIL: forget directive NOT correctly in place after rebuild:")
    for m in missing[:20]:
        print("   -", m)
    sys.exit(f"Aborting — {len(missing)}/{len(ins)} insertions missing. Check the sidecar/source before re-running.")
if len(stages_with_directive) <= 1:
    sys.exit(f"FAIL: directive present in only {len(stages_with_directive)} stage(s) "
             f"({stages_with_directive}); that is the stale-file bug signature. Aborting.")

print(f"   OK: {len(ins)} forget insertions verified across {len(stages_with_directive)} stages.")
PY

# ---------------------------------------------------------------------------
# 4) Re-run ChatGPT web eval, forget only, this topic only.
#    run_chatgpt_eval.sh handles browser login + timing.
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Launching ChatGPT web run (forget / $TOPIC) ..."
echo "      A browser login step may appear first — complete it, then press Enter."
echo ""
cd "$SELF_DIR"
TOPIC="$TOPIC" WORLDS=forget ./run_chatgpt_eval.sh

echo ""
echo "============================================"
echo " Done. Re-check the new forget sessions and confirm session_trace"
echo " user turns contain the directive in each control stage"
echo " (e.g. 'Please clear anything I shared about ...')."
echo "============================================"
