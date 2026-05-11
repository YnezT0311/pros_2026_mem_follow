#!/usr/bin/env python3
"""Build Hugging Face friendly exports from PersonaMem source data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

STAGE_ORDER = ["Initial", "Early", "Intermediate", "Late"]
SAMPLE_RE = re.compile(r"(.+)_persona(\d+)_sample(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the PersonaMem project.",
    )
    parser.add_argument(
        "--topic",
        default="travelPlanning",
        help="Topic to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory for HF-ready files.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Export format.",
    )
    return parser.parse_args()


def sample_meta_from_path(path: Path) -> dict:
    match = SAMPLE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unrecognized sample filename: {path}")
    stem, persona_id, sample_index = match.groups()
    normalized_stem = stem
    if normalized_stem.startswith("conversation_"):
        normalized_stem = normalized_stem[len("conversation_") :]
    topic = normalized_stem.split("_persona", 1)[0]
    return {
        "sample_id": f"{normalized_stem}_persona{persona_id}_sample{sample_index}",
        "topic": topic,
        "persona_id": int(persona_id),
        "sample_index": int(sample_index),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_stage_values(obj: dict, prefix: str) -> Iterable[tuple[str, object]]:
    for stage in STAGE_ORDER:
        key = f"{prefix} {stage} Stage"
        if key in obj:
            yield stage.lower(), obj[key]


def normalize_items(values: object) -> list[tuple[int, object]]:
    if isinstance(values, list):
        return list(enumerate(values))
    if isinstance(values, dict):
        return list(enumerate(values.values()))
    return [(0, values)]


def parse_conversation_lines(lines: list[str]) -> list[dict]:
    messages = []
    for line in lines:
        if ": " in line:
            role, text = line.split(": ", 1)
        else:
            role, text = "unknown", line
        role = role.strip().lower().replace("_", " ")
        messages.append({"role": role, "text": text.strip()})
    return messages


def flatten_stage_string_lists(obj: dict, prefix: str) -> list[dict]:
    items = []
    for stage_name, values in iter_stage_values(obj, prefix):
        for idx, value in enumerate(values):
            items.append(
                {
                    "stage": stage_name,
                    "stage_item_index": idx,
                    "text": value,
                }
            )
    return items


def build_conversations(project_root: Path, topic: str) -> list[dict]:
    rows = []
    baseline_dir = project_root / "data" / "baseline" / topic
    for path in sorted(baseline_dir.glob("conversation_*.json")):
        obj = load_json(path)
        meta = sample_meta_from_path(path)

        conversation_rows = flatten_stage_string_lists(obj, "Conversation")
        interaction_history = []
        for stage_name, values in iter_stage_values(obj, "Interaction History"):
            for idx, value in normalize_items(values):
                interaction_history.append(
                    {
                        "stage": stage_name,
                        "stage_item_index": idx,
                        "item": value,
                    }
                )
        conversation_messages = []
        for item in conversation_rows:
            parsed = parse_conversation_lines([item["text"]])[0]
            conversation_messages.append(
                {
                    "stage": item["stage"],
                    "stage_turn_index": item["stage_item_index"],
                    "role": parsed["role"],
                    "text": parsed["text"],
                }
            )
        rows.append(
            {
                **meta,
                "source_file": str(path.relative_to(project_root)),
                "original_persona": obj["Original Persona"],
                "expanded_persona": obj["Expanded Persona"],
                "contains_synthetic_pii": bool(obj.get("Persona PII")),
                "persona_pii": obj.get("Persona PII", {}),
                "conversation": [item["text"] for item in conversation_rows],
                "interaction_history": interaction_history,
                "num_messages": len(conversation_messages),
                "num_interaction_history_items": len(interaction_history),
            }
        )
    return sorted(rows, key=lambda row: (row["topic"], row["persona_id"], row["sample_index"]))


def build_whole_recall(project_root: Path, topic: str) -> list[dict]:
    rows = []
    test_dir = project_root / "data" / "test" / topic / "whole_recall"
    for path in sorted(test_dir.glob("whole_recall_qa_*.json")):
        obj = load_json(path)
        meta = sample_meta_from_path(Path(path.name.replace("whole_recall_qa_", "conversation_")))
        for item_idx, item in enumerate(obj.get("items", [])):
            rendered = item["rendered"]
            rows.append(
                {
                    **meta,
                    "source_file": str(path.relative_to(project_root)),
                    "qa_family": "whole_recall",
                    "item_index": item_idx,
                    "timestamp": item["timestamp"],
                    "turn_role": item.get("turn_role"),
                    "identifier_label": item.get("identifier_label"),
                    "user_turn": item.get("user_turn"),
                    "task_goal": item.get("task_goal"),
                    "question": rendered["question"],
                    "choice_a": rendered["choices"].get("A"),
                    "choice_b": rendered["choices"].get("B"),
                    "choice_c": rendered["choices"].get("C"),
                    "choice_order": rendered.get("choice_order", []),
                    "correct_choice": rendered.get("remember_correct_choice"),
                    "distractor_choice": rendered.get("distractor_irrelevant_choice"),
                    "not_remember_choice": rendered.get("not_remember_choice"),
                    "answer_type_to_choice": rendered.get("answer_type_to_choice", {}),
                    "choice_to_answer_type": rendered.get("choice_to_answer_type", {}),
                    "is_identifier_unique_to_target": item.get(
                        "is_identifier_unique_to_target"
                    ),
                    "disambiguation": item.get("disambiguation", {}),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["topic"],
            row["persona_id"],
            row["sample_index"],
            row["timestamp"],
            row["item_index"],
        ),
    )


def build_slot_recall(project_root: Path, topic: str) -> list[dict]:
    rows = []
    test_dir = project_root / "data" / "test" / topic / "slot_recall"
    for path in sorted(test_dir.glob("slot_recall_qa_*.json")):
        obj = load_json(path)
        meta = sample_meta_from_path(Path(path.name.replace("slot_recall_qa_", "conversation_")))
        for parent_idx, parent_item in enumerate(obj.get("items", [])):
            rendered_items = parent_item.get("rendered", {}).get("items", [])
            for item_idx, item in enumerate(rendered_items):
                rows.append(
                    {
                        **meta,
                        "source_file": str(path.relative_to(project_root)),
                        "qa_family": "slot_recall",
                        "parent_item_index": parent_idx,
                        "item_index": item_idx,
                        "timestamp": parent_item["timestamp"],
                        "turn_role": parent_item.get("turn_role"),
                        "identifier_label": item.get("identifier_label"),
                        "user_turn": parent_item.get("user_turn"),
                        "task_goal": parent_item.get("task_goal"),
                        "sensitive_key": item.get("sensitive_key"),
                        "sensitive_value": item.get("sensitive_value"),
                        "question": item.get("question"),
                        "choice_a": item["choices"].get("A"),
                        "choice_b": item["choices"].get("B"),
                        "choice_c": item["choices"].get("C"),
                        "choice_order": item.get("choice_order", []),
                        "correct_choice": item.get("remember_correct_choice"),
                        "distractor_choice": item.get("distractor_irrelevant_choice"),
                        "not_remember_choice": item.get("not_remember_choice"),
                        "answer_type_to_choice": item.get("answer_type_to_choice", {}),
                        "choice_to_answer_type": item.get("choice_to_answer_type", {}),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["topic"],
            row["persona_id"],
            row["sample_index"],
            row["timestamp"],
            row["parent_item_index"],
            row["item_index"],
        ),
    )


def ensure_parquet_support():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Parquet export requires `pyarrow`. Install it with `pip install pyarrow`, "
            "or rerun with `--format jsonl`."
        ) from exc
    return pa, pq


def write_records(records: list[dict], path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        pa, pq = ensure_parquet_support()
        table = pa.Table.from_pylist(records)
        pq.write_table(table, path)
        return

    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()

    conversations = build_conversations(project_root, args.topic)
    whole_recall = build_whole_recall(project_root, args.topic)
    slot_recall = build_slot_recall(project_root, args.topic)

    suffix = "parquet" if args.format == "parquet" else "jsonl"
    write_records(
        conversations,
        output_dir / "conversations" / f"data.{suffix}",
        args.format,
    )
    write_records(
        whole_recall,
        output_dir / "whole_recall_mcq" / f"test.{suffix}",
        args.format,
    )
    write_records(
        slot_recall,
        output_dir / "slot_recall_mcq" / f"test.{suffix}",
        args.format,
    )

    summary = {
        "topic": args.topic,
        "format": args.format,
        "conversations_rows": int(len(conversations)),
        "whole_recall_rows": int(len(whole_recall)),
        "slot_recall_rows": int(len(slot_recall)),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
