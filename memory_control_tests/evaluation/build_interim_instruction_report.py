import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from memory_control_tests.evaluation.annotate_slot_types_llm import (
    ALLOWED_SLOT_TYPES,
    _load_client,
    _request_text,
)
from memory_control_tests.evaluation.summarize_instruction_control_results import (
    PERIOD_ORDER,
    _aggregate_forget_immediate,
    _aggregate_forget_persistence,
    _aggregate_forget_timing,
    _aggregate_no_store_stage,
    _aggregate_no_use_subset,
    _aggregate_world,
    _load_records,
)


def _fmt_rate(summary: Dict[str, Any]) -> str:
    return f"{summary.get('remember_correct_rate', 0.0):.3f} / {summary.get('not_remember_rate', 0.0):.3f}"


def _world_rows(group: List[Dict[str, Any]], world: str, persona_limit: int) -> Dict[str, Any]:
    chosen = [r for r in group if r["world"] == world and r["persona"] < persona_limit]
    return {
        "coverage_personas": sorted({r["persona"] for r in chosen}),
        "coverage_files": len(chosen),
        "metrics": _aggregate_world(group, world, persona_limit),
    }


def _instruction_strength(metrics: Dict[str, Any]) -> float:
    whole = metrics.get("whole_recall_key_turns", {})
    slot = metrics.get("slot_recall_key_turns", {})
    return (whole.get("not_remember_rate", 0.0) + slot.get("not_remember_rate", 0.0)) / 2.0


def _probe_cost(baseline: Dict[str, Any], other: Dict[str, Any]) -> float:
    bw = baseline.get("whole_recall_probe_turns", {}).get("remember_correct_rate", 0.0)
    bs = baseline.get("slot_recall_probe_turns", {}).get("remember_correct_rate", 0.0)
    ow = other.get("whole_recall_probe_turns", {}).get("remember_correct_rate", 0.0)
    os = other.get("slot_recall_probe_turns", {}).get("remember_correct_rate", 0.0)
    return ((bw - ow) + (bs - os)) / 2.0


def _build_slot_batch_prompt(batch: List[Tuple[int, Dict[str, str]]]) -> str:
    allowed = ", ".join(ALLOWED_SLOT_TYPES)
    payload = [
        {
            "id": idx,
            "identifier_label": item["identifier_label"],
            "question": item["question"],
            "sensitive_key": item["sensitive_key"],
            "sensitive_value": item["sensitive_value"],
        }
        for idx, item in batch
    ]
    return (
        "You are assigning coarse slot types for memory-analysis.\n"
        f"Choose exactly one label from: {allowed}.\n"
        "Return valid JSON only as a list of objects with keys `id`, `slot_type_llm`, and `reason`.\n"
        "The label should describe the kind of detail the slot is about, not whether it was remembered.\n\n"
        f"Items:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _classify_slot_items_batched(
    items: List[Dict[str, Any]],
    model: str,
    api_key_file: str,
    batch_size: int = 25,
) -> Dict[Tuple[str, str, str, str], str]:
    client = _load_client(api_key_file)
    keyed: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for item in items:
        key = (
            str(item.get("identifier_label", "")).strip(),
            str(item.get("question", "")).strip(),
            str(item.get("sensitive_key", "")).strip(),
            str(item.get("sensitive_value", "")).strip(),
        )
        keyed[key] = {
            "identifier_label": key[0],
            "question": key[1],
            "sensitive_key": key[2],
            "sensitive_value": key[3],
        }

    keys = list(keyed.keys())
    out: Dict[Tuple[str, str, str, str], str] = {}
    for start in range(0, len(keys), batch_size):
        chunk_keys = keys[start : start + batch_size]
        batch = [(idx, keyed[key]) for idx, key in enumerate(chunk_keys)]
        raw = _request_text(client, model, [{"role": "user", "content": _build_slot_batch_prompt(batch)}])
        parsed = json.loads(raw)
        by_id = {int(entry["id"]): str(entry.get("slot_type_llm", "")).strip() for entry in parsed}
        for idx, key in enumerate(chunk_keys):
            label = by_id.get(idx, "other_detail")
            if label not in ALLOWED_SLOT_TYPES:
                label = "other_detail"
            out[key] = label
    return out


def _slot_type_section(records: List[Dict[str, Any]], slot_type_model: str, slot_type_api_key_file: str) -> Dict[str, Any]:
    plain = [r for r in records if r["backend"] == "plain"]
    slot_items = []
    for rec in plain:
        slot_items.extend(rec.get("slot_recall_results", []))
    labels = _classify_slot_items_batched(slot_items, slot_type_model, slot_type_api_key_file)

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in plain:
        world = rec["world"]
        for item in rec.get("slot_recall_results", []):
            key = (
                str(item.get("identifier_label", "")).strip(),
                str(item.get("question", "")).strip(),
                str(item.get("sensitive_key", "")).strip(),
                str(item.get("sensitive_value", "")).strip(),
            )
            slot_type = labels.get(key, "other_detail")
            if item.get("turn_role") == "key":
                grouped[world][slot_type].append(item)

    def summarize(world: str, sort_answer_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        rows = []
        for slot_type, items in grouped[world].items():
            if len(items) < 10:
                continue
            remember = sum(1 for x in items if x.get("predicted_answer_type") == "remember_correct") / len(items)
            not_remember = sum(1 for x in items if x.get("predicted_answer_type") == "not_remember") / len(items)
            rows.append(
                (
                    slot_type,
                    {
                        "n": len(items),
                        "remember_correct_rate": remember,
                        "not_remember_rate": not_remember,
                    },
                )
            )
        key_name = "remember_correct_rate" if sort_answer_type == "remember_correct" else "not_remember_rate"
        return sorted(rows, key=lambda kv: (-kv[1][key_name], -kv[1]["n"], kv[0]))

    return {
        "baseline_most_remembered": summarize("baseline", "remember_correct")[:5],
        "no_store_most_forgotten": summarize("no_store", "not_remember")[:5],
        "forget_most_forgotten": summarize("forget", "not_remember")[:5],
        "no_use_most_forgotten": summarize("no_use", "not_remember")[:5],
        "num_unique_slot_items": len(labels),
    }


def _coverage_line(rows: Dict[str, Any]) -> str:
    return f"personas={rows['coverage_personas']}, files={rows['coverage_files']}"


def _render_world_table(section: Dict[str, Any]) -> List[str]:
    m = section["metrics"]
    return [
        "| World | Coverage | whole key R/NR | slot key R/NR | whole probe R/NR | slot probe R/NR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| baseline | {_coverage_line(section['baseline'])} | {_fmt_rate(section['baseline']['metrics']['whole_recall_key_turns'])} | {_fmt_rate(section['baseline']['metrics']['slot_recall_key_turns'])} | {_fmt_rate(section['baseline']['metrics']['whole_recall_probe_turns'])} | {_fmt_rate(section['baseline']['metrics']['slot_recall_probe_turns'])} |",
        f"| no_store | {_coverage_line(section['no_store'])} | {_fmt_rate(section['no_store']['metrics']['whole_recall_key_turns'])} | {_fmt_rate(section['no_store']['metrics']['slot_recall_key_turns'])} | {_fmt_rate(section['no_store']['metrics']['whole_recall_probe_turns'])} | {_fmt_rate(section['no_store']['metrics']['slot_recall_probe_turns'])} |",
        f"| forget | {_coverage_line(section['forget'])} | {_fmt_rate(section['forget']['metrics']['whole_recall_key_turns'])} | {_fmt_rate(section['forget']['metrics']['slot_recall_key_turns'])} | {_fmt_rate(section['forget']['metrics']['whole_recall_probe_turns'])} | {_fmt_rate(section['forget']['metrics']['slot_recall_probe_turns'])} |",
        f"| no_use | {_coverage_line(section['no_use'])} | {_fmt_rate(section['no_use']['metrics']['whole_recall_key_turns'])} | {_fmt_rate(section['no_use']['metrics']['slot_recall_key_turns'])} | {_fmt_rate(section['no_use']['metrics']['whole_recall_probe_turns'])} | {_fmt_rate(section['no_use']['metrics']['slot_recall_probe_turns'])} |",
    ]


def _build_backend_section(records: List[Dict[str, Any]], backend: str, model: str, other_personas: int, forget_personas: int) -> Dict[str, Any]:
    group = [r for r in records if r["backend"] == backend and r["model"] == model]
    baseline = _world_rows(group, "baseline", other_personas)
    no_store = _world_rows(group, "no_store", other_personas)
    forget = _world_rows(group, "forget", forget_personas)
    no_use = _world_rows(group, "no_use", other_personas)
    return {
        "baseline": baseline,
        "no_store": no_store,
        "forget": forget,
        "no_use": no_use,
        "difficulty": {
            "no_store_key_suppression": _instruction_strength(no_store["metrics"]),
            "forget_key_suppression": _instruction_strength(forget["metrics"]),
            "no_use_key_suppression": _instruction_strength(no_use["metrics"]),
        },
        "probe_cost": {
            "no_store_vs_baseline": _probe_cost(baseline["metrics"], no_store["metrics"]),
            "forget_vs_baseline": _probe_cost(baseline["metrics"], forget["metrics"]),
            "no_use_vs_baseline": _probe_cost(baseline["metrics"], no_use["metrics"]),
        },
        "no_store_stage_effect": _aggregate_no_store_stage(group, other_personas),
        "forget_questions": {
            "immediate_effect": _aggregate_forget_immediate(group, forget_personas),
            "persistence": _aggregate_forget_persistence(group, forget_personas),
            "timing_sensitivity": _aggregate_forget_timing(group, forget_personas),
        },
        "no_use_questions": {
            "no_use@E_test@E": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Early Stage"),
            "no_use@I_test@I": _aggregate_no_use_subset(group, other_personas, "Conversation Intermediate Stage", "Conversation Intermediate Stage"),
            "no_use@L_test@L": _aggregate_no_use_subset(group, other_personas, "Conversation Late Stage", "Conversation Late Stage"),
            "no_use@E_test@I": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Intermediate Stage"),
            "no_use@E_test@L": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Late Stage"),
            "no_use@E_release@E_test@E": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Early Stage", "Conversation Early Stage"),
            "no_use@E_release@E_test@I": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Intermediate Stage", "Conversation Early Stage"),
            "no_use@E_release@E_test@L": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Late Stage", "Conversation Early Stage"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_md", default="eval_results/travelPlanning/interim_instruction_report.md")
    parser.add_argument("--models", nargs="+", default=["gpt-5.4-mini", "gpt-4o"])
    parser.add_argument("--other_personas", type=int, default=4)
    parser.add_argument("--forget_personas", type=int, default=10)
    parser.add_argument("--slot_type_model", default="gpt-5-mini")
    parser.add_argument("--slot_type_api_key_file", default="openrouter_key.txt")
    args = parser.parse_args()

    root = Path("data/test/travelPlanning/specs")
    eval_root = Path("eval_results/travelPlanning")
    records = _load_records([root, eval_root], args.models)

    sections = {}
    for backend in ["plain", "mem0", "LangMem"]:
        for model in args.models:
            key = f"{backend}__{model}"
            sections[key] = _build_backend_section(records, backend, model, args.other_personas, args.forget_personas)

    slot_type_summary = _slot_type_section(records, args.slot_type_model, args.slot_type_api_key_file)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = [
        "# Interim Instruction-Control Report",
        "",
        f"Generated: {now}",
        "",
        "This is a partial report based on the results currently on disk. `plain` is complete for the intended scope. `mem0` and `LangMem` are still running, so their sections should be read as preliminary coverage-based summaries.",
        "",
        "## Coverage",
        "",
        "- `plain`: complete for both `gpt-5.4-mini` and `gpt-4o`.",
        "- `mem0`: currently enough for `baseline/no_store` on personas `0-1`, partial `no_use`, early `forget` is beginning to appear.",
        "- `LangMem`: currently enough for `baseline/no_store` on personas `0-1` for `gpt-5.4-mini`, and persona `0-1` is emerging for `gpt-4o`; `no_use` is still partial.",
        "",
        "## Average Performance and Instruction Difficulty",
        "",
    ]

    for key, section in sections.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.extend(_render_world_table(section))
        lines.append("")
        lines.append("Key-suppression strength (higher means the instruction pushes forbidden key facts toward `not_remember` more strongly):")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(section["difficulty"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("Probe-utility cost (positive means lower probe `remember_correct_rate` than baseline):")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(section["probe_cost"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## `no_store` Test-Stage Effect")
    lines.append("")
    for key, section in sections.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(section["no_store_stage_effect"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## `forget` Questions")
    lines.append("")
    lines.append("The README-defined questions are: immediate effect of forgetting, persistence of forgetting, and timing sensitivity.")
    lines.append("")
    for key, section in sections.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(section["forget_questions"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## `no_use` Questions")
    lines.append("")
    lines.append("The README-defined questions are: immediate suppression, persistence, and recovery after release.")
    lines.append("")
    for key, section in sections.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(section["no_use_questions"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Slot-Type Difficulty in `slot_recall` (plain only, LLM post-processing)")
    lines.append("")
    lines.append(f"Unique slot items classified: {slot_type_summary['num_unique_slot_items']}")
    lines.append("")
    for title, rows in [
        ("Baseline: most easily remembered slot types", slot_type_summary["baseline_most_remembered"]),
        ("`no_store`: slot types with the highest `not_remember` on key turns", slot_type_summary["no_store_most_forgotten"]),
        ("`forget`: slot types with the highest `not_remember` on key turns", slot_type_summary["forget_most_forgotten"]),
        ("`no_use`: slot types with the highest `not_remember` on key turns", slot_type_summary["no_use_most_forgotten"]),
    ]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| slot_type_llm | n | remember | not_remember |")
        lines.append("|---|---:|---:|---:|")
        for slot_type, stats in rows:
            lines.append(
                f"| {slot_type} | {stats['n']} | {stats['remember_correct_rate']:.3f} | {stats['not_remember_rate']:.3f} |"
            )
        lines.append("")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
