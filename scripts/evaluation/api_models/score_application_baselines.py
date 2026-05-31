"""Score application MCQ baselines for differentiation.

An application item "differentiates" when the model answers correctly in
BOTH conditions:
  - seen_baseline      -> the use-memory option (target turn present)
  - never_seen_baseline-> the without-memory option (target turn absent)

Reads the per-world result JSONs produced by
`memory_control_tests.evaluation.application` under
eval_results/application/<world>/<model_tag>/.

Usage:
    python scripts/evaluation/api_models/score_application_baselines.py \
        --models gpt-5.4-mini opus-4.7 gpt-5.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path("eval_results/application")
DEFAULT_DATA_ROOT = Path("data/application/mcq/by_world")


def load_world(world: str, model_tag: str) -> dict:
    p = RESULTS_ROOT / world / model_tag / f"{world}.plain_api_{model_tag}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {r["id"]: r for r in data.get("results", [])}


def load_expected_ids(data_root: Path) -> list[str]:
    p = data_root / "seen_baseline.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [str(item["id"]) for item in data.get("items", [])]


def score(model_tag: str, expected_ids: list[str] | None = None) -> dict | None:
    seen = load_world("seen_baseline", model_tag)
    never = load_world("never_seen_baseline", model_tag)
    if not seen or not never:
        return None
    ids = list(expected_ids or sorted(set(seen) | set(never)))
    expected_set = set(ids)
    seen_extra = sorted(set(seen) - expected_set)
    never_extra = sorted(set(never) - expected_set)
    seen_missing = sorted(expected_set - set(seen))
    never_missing = sorted(expected_set - set(never))
    seen_ok = sum(1 for i in ids if seen.get(i, {}).get("is_expected"))
    never_ok = sum(1 for i in ids if never.get(i, {}).get("is_expected"))
    diff_ids = [
        i for i in ids
        if seen.get(i, {}).get("is_expected") and never.get(i, {}).get("is_expected")
    ]
    return {
        "model": model_tag,
        "n": len(ids),
        "seen_ok": seen_ok,
        "never_ok": never_ok,
        "differentiating": len(diff_ids),
        "diff_ids": diff_ids,
        "seen_missing": seen_missing,
        "never_missing": never_missing,
        "seen_extra": seen_extra,
        "never_extra": never_extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model tags as they appear in the result path (slashes -> _)")
    ap.add_argument("--show-ids", action="store_true", help="Print the differentiating item ids")
    ap.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Current application by_world directory used to validate item ids",
    )
    args = ap.parse_args()
    expected_ids = load_expected_ids(Path(args.data_root))

    rows = []
    for m in args.models:
        tag = m.replace("/", "_")
        s = score(tag, expected_ids=expected_ids or None)
        if s is None:
            print(f"{m}: (results missing)")
            continue
        rows.append(s)

    if not rows:
        return
    print(f"{'model':<18} {'n':>3} {'seen':>6} {'never':>6} {'differ':>7}")
    for s in rows:
        print(f"{s['model']:<18} {s['n']:>3} {s['seen_ok']:>6} {s['never_ok']:>6} {s['differentiating']:>7}")
        problems = {
            "seen_missing": s["seen_missing"],
            "never_missing": s["never_missing"],
            "seen_extra": s["seen_extra"],
            "never_extra": s["never_extra"],
        }
        for label, values in problems.items():
            if values:
                print(f"  WARNING {s['model']} {label}: {len(values)}")
                for item_id in values:
                    print(f"    {item_id}")
    if args.show_ids:
        for s in rows:
            print(f"\n[{s['model']}] differentiating ({s['differentiating']}):")
            for i in s["diff_ids"]:
                print(f"  {i}")


if __name__ == "__main__":
    main()
