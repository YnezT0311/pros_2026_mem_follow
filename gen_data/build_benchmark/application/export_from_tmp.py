#!/usr/bin/env python3
"""Export approved tmp application MCQ artifacts into data/application."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


TOPICS = ["travelPlanning", "financialConsultation", "medicalConsultation"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def item_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise TypeError("items.json must be a list or an object with an items list")


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_topic(run_root: Path, output_root: Path, topic: str, sample: str) -> int:
    src_dir = run_root / topic / sample
    dst_dir = output_root / "mcq_work" / topic / sample
    items = item_rows(load_json(src_dir / "items.json"))
    kept = [item for item in items if item.get("status") == "kept"]
    dropped = [item for item in items if item.get("status") != "kept"]

    dump_json(dst_dir / "application_items.json", {"topic": topic, "sample_id": sample, "items": kept})
    if dropped:
        dump_json(dst_dir / "dropped_items.json", {"topic": topic, "sample_id": sample, "items": dropped})

    review_dir = dst_dir / "review"
    for name in [
        "planner_candidates.json",
        "planner_candidates_normalized.json",
        "planner_review.md",
        "questioner_notes.json",
        "analyst_precheck.md",
        "analyst_precheck.json",
        "analyst_round2.md",
        "analyst_round2.json",
        "analyst_focused_stage04_stage23.md",
        "analyst_focused_stage04_stage23.json",
        "analyst_focused_round3_stage04_stage23.md",
        "analyst_focused_round3_stage04_stage23.json",
        "analyst_focused_round4_stage04_stage23.md",
        "analyst_focused_round4_stage04_stage23.json",
        "fixer_notes_round1.md",
        "fixer_notes_round1.json",
        "fixer_notes_round2.json",
    ]:
        copy_if_exists(src_dir / name, review_dir / name)
    return len(kept)


def export_worlds(run_root: Path, output_root: Path) -> None:
    worlds_src = run_root / "transformed_worlds"
    worlds_dst = output_root / "mcq_work" / "_worlds"
    if worlds_dst.exists():
        shutil.rmtree(worlds_dst)
    shutil.copytree(worlds_src, worlds_dst)

    application_worlds = load_json(worlds_src / "application_worlds.json")
    export_root = output_root / "mcq"
    export_root.mkdir(parents=True, exist_ok=True)
    dump_json(export_root / "application_mcq.json", application_worlds)
    by_world = export_root / "by_world"
    if by_world.exists():
        shutil.rmtree(by_world)
    shutil.copytree(worlds_src / "plain_inputs", by_world)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", default="data/application")
    parser.add_argument("--sample", default="persona0_sample0")
    parser.add_argument("--topics", nargs="+", default=TOPICS)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_root = Path(args.output_root)

    counts = {
        topic: export_topic(run_root, output_root, topic, args.sample)
        for topic in args.topics
    }
    export_worlds(run_root, output_root)
    copy_if_exists(run_root / "SUMMARY.md", output_root / "mcq_work" / "SUMMARY.md")
    print(json.dumps({"topic_counts": counts, "total": sum(counts.values())}, indent=2))


if __name__ == "__main__":
    main()
