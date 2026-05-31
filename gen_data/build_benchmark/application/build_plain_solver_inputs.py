#!/usr/bin/env python3
"""Build blind baseline-gate inputs for application MCQ items.

This renders the same message shape as the plain evaluation adapter and writes
with-target / without-target files for agent or API solver validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_worlds import (
    build_eval_prompt,
    context_for_world,
    load_json,
    parse_source_stage,
    persona_messages,
)


TOPICS = ["travelPlanning", "financialConsultation", "medicalConsultation"]


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


def build_solver_inputs(
    *,
    items_root: Path,
    conversation_root: Path,
    output_root: Path,
    topics: list[str],
    sample: str,
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {
        "with_target_baseline": [],
        "without_target_baseline": [],
    }
    expected: list[dict[str, Any]] = []

    for topic in topics:
        items = item_rows(load_json(items_root / topic / sample / "items.json"))
        conversation = load_json(conversation_root / topic / sample / "conversation_package.json")
        pmsgs = persona_messages(conversation)
        for item in items:
            if item.get("status") != "kept":
                continue
            mcq_prompt = build_eval_prompt(item["question"], item["choices"])
            expected.append(
                {
                    "id": item["id"],
                    "topic": topic,
                    "stage_id": item["stage_id"],
                    "with_target_baseline": item["expected"]["with_target_baseline"],
                    "without_target_baseline": item["expected"]["without_target_baseline"],
                }
            )
            source_messages = parse_source_stage(conversation, item["stage_id"])
            for condition in by_condition:
                world = "seen_baseline" if condition == "with_target_baseline" else "never_seen_baseline"
                context = context_for_world(item, source_messages, world)
                by_condition[condition].append(
                    {
                        "id": item["id"],
                        "topic": topic,
                        "stage_id": item["stage_id"],
                        "condition": condition,
                        "context_source": {
                            "conversation_root": str(conversation_root),
                            "stage_id": item["stage_id"],
                            "synthetic_context": False,
                        },
                        "messages": pmsgs + context + [{"role": "user", "content": mcq_prompt}],
                    }
                )

    for condition, rows in by_condition.items():
        dump_json(output_root / f"{condition}.json", {"condition": condition, "items": rows})
    dump_json(output_root / "expected.json", {"items": expected})
    summary = {
        "conditions": {condition: len(rows) for condition, rows in by_condition.items()},
        "expected": len(expected),
    }
    dump_json(output_root / "README.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-root", required=True)
    parser.add_argument("--conversation-root", default="data/generated")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample", default="persona0_sample0")
    parser.add_argument("--topics", nargs="+", default=TOPICS)
    args = parser.parse_args()

    summary = build_solver_inputs(
        items_root=Path(args.items_root),
        conversation_root=Path(args.conversation_root),
        output_root=Path(args.output_root),
        topics=args.topics,
        sample=args.sample,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
