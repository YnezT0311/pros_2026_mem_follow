#!/usr/bin/env python3
"""Compare blind plain-solver outputs against application baseline expectations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def choice(result: dict[str, Any]) -> str:
    return str(result.get("choice") or "").strip().upper()


def compare(input_root: Path, output_path: Path) -> dict[str, Any]:
    expected = {item["id"]: item for item in load_json(input_root / "expected.json")["items"]}
    with_results = {item["id"]: item for item in load_json(input_root / "with_target_solver_results.json")["items"]}
    without_results = {item["id"]: item for item in load_json(input_root / "without_target_solver_results.json")["items"]}

    rows: list[dict[str, Any]] = []
    summary: dict[str, Counter] = defaultdict(Counter)
    for item_id, exp in expected.items():
        with_choice = choice(with_results[item_id])
        without_choice = choice(without_results[item_id])
        with_ok = with_choice == str(exp["with_target_baseline"]).upper()
        without_ok = without_choice == str(exp["without_target_baseline"]).upper()
        row = {
            "id": item_id,
            "topic": exp["topic"],
            "stage_id": exp["stage_id"],
            "with_choice": with_choice,
            "with_expected": exp["with_target_baseline"],
            "with_ok": with_ok,
            "with_rationale": with_results[item_id].get("rationale", ""),
            "without_choice": without_choice,
            "without_expected": exp["without_target_baseline"],
            "without_ok": without_ok,
            "without_rationale": without_results[item_id].get("rationale", ""),
            "pass": with_ok and without_ok,
        }
        rows.append(row)
        summary[exp["topic"]]["total"] += 1
        summary[exp["topic"]]["pass"] += int(row["pass"])
        summary[exp["topic"]]["with_ok"] += int(with_ok)
        summary[exp["topic"]]["without_ok"] += int(without_ok)

    out = {
        "total": len(rows),
        "passed": sum(int(row["pass"]) for row in rows),
        "failed": [row for row in rows if not row["pass"]],
        "summary": {topic: dict(counts) for topic, counts in summary.items()},
    }
    dump_json(output_path, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = compare(Path(args.input_root), Path(args.output))
    print(json.dumps({"total": out["total"], "passed": out["passed"], "failed": len(out["failed"]), "summary": out["summary"]}, indent=2))


if __name__ == "__main__":
    main()
