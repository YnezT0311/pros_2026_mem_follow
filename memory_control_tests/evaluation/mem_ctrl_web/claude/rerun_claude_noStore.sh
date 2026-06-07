#!/bin/bash
# =============================================================================
# rerun_claude_noStore.sh  —  foolproof re-run of Claude web no_store (recall)
# -----------------------------------------------------------------------------
# Background: the earlier Claude no_store run (2026-06-05) was INVALID because
# the web export was missing both the per-stage no_store transformed-history
# files and the sidecar (memory_targets.json). The runner then fell back to
# on-the-fly generation with an empty sidecar -> NO no_store directive was ever
# delivered to the model. The fix (correct files + sidecar) is now committed.
#
# Important: the runner loads an existing per-stage file DIRECTLY and never
# regenerates it. So any STALE no_store file left on this machine from the bad
# run would be reused and reproduce the bug. This script removes those stale
# files, pulls the corrected ones, VERIFIES the directive is present, and only
# then re-runs. Just run it:
#
#     ./rerun_claude_noStore.sh
#
# Optional env:
#     SKIP_SYNC=1   skip the git clean + pull (only verify + run)
#     RESULTS=...   results dir (forwarded to run_claude_eval.sh)
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../mem_ctrl_web/claude
WEB_DATA="$(cd "$SELF_DIR/../data" && pwd)"                # .../mem_ctrl_web/data
PERSONA_DIR="$WEB_DATA/recall/mcq/travelPlanning/persona0_sample0"
SIDECAR="$WEB_DATA/recall/mcq_work/travelPlanning/persona0_sample0/memory_targets.json"
REPO_ROOT="$(git -C "$SELF_DIR" rev-parse --show-toplevel)"

# Control stages that MUST carry a no_store directive (stage_02 / stage_04 have
# no control point by design and are intentionally directive-free).
CONTROL_STAGES=(01 03 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23)

echo "============================================"
echo " Claude no_store re-run (travelPlanning)"
echo " repo:      $REPO_ROOT"
echo " web data:  $WEB_DATA"
echo "============================================"

# ---------------------------------------------------------------------------
# 1) Remove STALE UNTRACKED no_store files + sidecar so they cannot be reused
#    and so they cannot block 'git pull'. Only untracked files are deleted;
#    files already tracked by git are left for pull to update cleanly.
# ---------------------------------------------------------------------------
remove_if_untracked () {
  local f="$1"
  [ -e "$f" ] || return 0
  if git -C "$REPO_ROOT" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    return 0   # tracked -> leave it, pull will update it
  fi
  rm -f "$f"
  echo "   removed stale untracked: ${f#$REPO_ROOT/}"
}

if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  echo ""
  echo "[1/4] Removing stale untracked no_store files + sidecar ..."
  shopt -s nullglob
  for f in "$PERSONA_DIR"/persona0_sample0.no_store.stage_*.transformed_history.json; do
    remove_if_untracked "$f"
  done
  shopt -u nullglob
  remove_if_untracked "$SIDECAR"
  echo "      done."

  # -------------------------------------------------------------------------
  # 2) Pull the corrected files + sidecar from the shared branch.
  # -------------------------------------------------------------------------
  echo ""
  echo "[2/4] git pull --ff-only ..."
  git -C "$REPO_ROOT" pull --ff-only
  # Make sure tracked copies match the committed version (undo any local edits).
  git -C "$REPO_ROOT" checkout -- \
    "$PERSONA_DIR"/persona0_sample0.no_store.stage_*.transformed_history.json \
    "$SIDECAR" 2>/dev/null || true
else
  echo ""
  echo "[1-2/4] SKIP_SYNC=1 -> skipping clean + pull"
fi

# ---------------------------------------------------------------------------
# 3) VERIFY the fix is actually in place before spending a real run.
#    Aborts loudly if any control stage is missing its directive.
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Verifying no_store directive is present ..."
python - "$PERSONA_DIR" "$SIDECAR" "${CONTROL_STAGES[@]}" <<'PY'
import json, sys
from pathlib import Path
persona = Path(sys.argv[1]); sidecar = Path(sys.argv[2]); stages = sys.argv[3:]

if not sidecar.exists():
    sys.exit(f"FAIL: sidecar missing -> {sidecar}\n"
             "      (did you pull the fix? without it the run will be directive-less)")

def norm(s): return ' '.join(s.replace('’',"'").replace('‘',"'").lower().split())
intent = ['remember','store','retain','keep them','drop the details','this round',
          'do not keep','saved after','to memory','not remembered','do not record',
          'do not add']

missing=[]
for st in stages:
    f = persona / f"persona0_sample0.no_store.stage_{st}.transformed_history.json"
    if not f.exists():
        missing.append(f"stage_{st}: file missing"); continue
    d = json.load(open(f))
    ins = d.get('_memory_control_metadata',{}).get('no_store_insertions',[])
    if not ins:
        missing.append(f"stage_{st}: no_store_insertions empty"); continue
    key = ins[0].get('key_stage','')
    txt = norm(' '.join(l for l in d.get(key,[]) if isinstance(l,str)))
    if not any(p in txt for p in intent):
        missing.append(f"stage_{st}: directive text not found in {key}")

if missing:
    print("FAIL: no_store directive NOT correctly in place:")
    for m in missing: print("   -", m)
    sys.exit("Aborting — fix/sync the web export before re-running (see script header).")
print(f"   OK: directive present in all {len(stages)} control stages + sidecar found.")
PY

# ---------------------------------------------------------------------------
# 4) Re-run Claude web eval, no_store only, travelPlanning only.
#    (run_claude_eval.sh handles browser login + timing.)
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Launching Claude web run (no_store / travelPlanning) ..."
echo "      A browser login step will appear first — complete it, then press Enter."
echo ""
cd "$SELF_DIR"
TOPIC=travelPlanning WORLDS=no_store ./run_claude_eval.sh

echo ""
echo "============================================"
echo " Done. Re-check the new no_store sessions and"
echo " confirm session_trace user turns contain the"
echo " directive (e.g. '... not something I want remembered.')."
echo "============================================"
