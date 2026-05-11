"""Re-parse `model_response` and (optionally) rebuild `summary` for eval JSONs.

Replaces both the old single-file `scores.py` and the old batch
`reparse_eval_outputs.py`. Operates on a single file or recursively on a
directory tree.

Usage:
    # single file, write to <stem>.scored.json
    python -m memory_control_tests.evaluation.scores --path eval_results/.../foo.json

    # in-place batch over a directory tree, only re-parse choices (no summary rebuild)
    python -m memory_control_tests.evaluation.scores --path eval_results/ --write_in_place

    # full rebuild including summary
    python -m memory_control_tests.evaluation.scores --path eval_results/ --write_in_place --rebuild_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ..common import build_recall_summary
from .shared import extract_choice


def _reparse_item(item: Dict[str, Any]) -> bool:
    choices = item.get("choices") or {}
    choice_to_answer_type = item.get("choice_to_answer_type") or {}
    if not choices or not choice_to_answer_type:
        return False
    labels = list(choices.keys())
    model_response = str(item.get("model_response", "") or "")
    predicted_choice = extract_choice(model_response, labels)
    predicted_answer_type = choice_to_answer_type.get(predicted_choice, "")
    changed = (
        item.get("predicted_choice") != predicted_choice
        or item.get("predicted_answer_type") != predicted_answer_type
    )
    item["predicted_choice"] = predicted_choice
    item["predicted_answer_type"] = predicted_answer_type
    return changed


def _process_file(path: Path, *, rebuild_summary: bool) -> tuple[bool, int]:
    """Returns (file_changed, num_items_changed)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, 0
    if not isinstance(data, dict):
        return False, 0

    items_changed = 0
    for key in ("whole_recall_results", "slot_recall_results"):
        for item in data.get(key, []):
            if _reparse_item(item):
                items_changed += 1

    summary_changed = False
    if rebuild_summary:
        new_summary = build_recall_summary(
            data.get("world", "baseline"),
            data.get("whole_recall_results", []),
            data.get("slot_recall_results", []),
        )
        if data.get("summary") != new_summary:
            data["summary"] = new_summary
            summary_changed = True

    file_changed = bool(items_changed) or summary_changed
    if file_changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_changed, items_changed


def _process_file_to_separate(path: Path, *, rebuild_summary: bool, output: Path | None) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("whole_recall_results", "slot_recall_results"):
        for item in data.get(key, []):
            _reparse_item(item)
    if rebuild_summary:
        data["summary"] = build_recall_summary(
            data.get("world", "baseline"),
            data.get("whole_recall_results", []),
            data.get("slot_recall_results", []),
        )
    out = output or path.with_name(path.name.replace(".json", ".scored.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _iter_target_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-parse evaluation outputs and optionally rebuild summary.")
    parser.add_argument("--path", required=True, help="A JSON file or directory to scan recursively.")
    parser.add_argument("--output", default="", help="Single-file output path (only valid when --path is a file and --write_in_place is not set).")
    parser.add_argument("--write_in_place", action="store_true", help="Overwrite each scanned file. Default for directory mode.")
    parser.add_argument("--rebuild_summary", action="store_true", help="Also rebuild the summary block via build_recall_summary.")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"path does not exist: {root}")

    if root.is_dir() and not args.write_in_place:
        # Default to in-place for directories — printing scored copies for every file is rarely what you want.
        args.write_in_place = True

    if root.is_file() and not args.write_in_place:
        out = _process_file_to_separate(
            root,
            rebuild_summary=args.rebuild_summary,
            output=Path(args.output) if args.output else None,
        )
        print(out)
        return

    paths = _iter_target_files(root)
    files_changed = 0
    items_changed_total = 0
    for p in paths:
        file_changed, items_changed = _process_file(p, rebuild_summary=args.rebuild_summary)
        if file_changed:
            files_changed += 1
            items_changed_total += items_changed
            print(p)
    print(f"files_changed={files_changed}")
    print(f"items_changed={items_changed_total}")


if __name__ == "__main__":
    main()
