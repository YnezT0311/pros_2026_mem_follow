import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


PERIOD_ORDER = {
    "Conversation Early Stage": 0,
    "Conversation Intermediate Stage": 1,
    "Conversation Late Stage": 2,
}

SUMMARY_KEYS = [
    "whole_recall_key_turns",
    "whole_recall_probe_turns",
    "slot_recall_key_turns",
    "slot_recall_probe_turns",
]


def classify_slot_type_for_summary(sensitive_key: str, sensitive_value: str, question: str = "") -> str:
    key = str(sensitive_key or "").strip().lower()
    value = str(sensitive_value or "").strip().lower()
    q = str(question or "").strip().lower()
    text = " ".join(part for part in [key, value, q] if part)

    if "email" in text or "@" in value:
        return "email"
    if any(token in text for token in ["phone", "call", "text"]) or re.search(r"\+?\d[\d\-\s\(\)]{6,}", value):
        return "phone"
    if any(token in text for token in ["budget", "cost", "price", "spend", "fare"]) or "$" in value:
        return "budget"
    if any(token in text for token in ["date", "when", "arrive", "departure", "return", "leave", "check-in", "check out"]) or re.search(r"\d{4}-\d{2}-\d{2}", value):
        return "date_or_time"
    if any(token in text for token in ["schedule", "timing", "time of day", "afternoon", "morning", "evening", "pickup"]):
        return "schedule_or_timing"
    if any(token in text for token in ["hotel", "guesthouse", "airport", "station", "neighborhood", "location", "address", "where"]):
        return "location_or_contact_point"
    if any(token in text for token in ["insurance", "coverage", "pre-existing condition", "asthma", "medical", "knee", "allergy", "diet", "gluten-free"]):
        if any(token in text for token in ["diet", "gluten-free", "vegetarian", "vegan"]):
            return "dietary_requirement"
        return "medical_or_access_need"
    if any(token in text for token in ["passport", "record", "reference", "confirmation", "booking code", "id", "miles", "account"]):
        return "document_or_account_reference"
    if any(token in text for token in ["preference", "prefer", "want", "need", "looking for", "style"]):
        return "preference_or_requirement"
    return "other_detail"


def _backend_from_name(name: str) -> Optional[str]:
    if ".mem0_retrieval_eval_" in name:
        return "mem0"
    if ".a_mem_retrieval_eval_" in name:
        return "A-Mem"
    if ".langmem_retrieval_eval_" in name:
        return "LangMem"
    if ".recall_eval_" in name:
        return "plain"
    return None


def _persona_from_name(name: str) -> Optional[int]:
    match = re.search(r"persona(\d+)_sample", name)
    return int(match.group(1)) if match else None


def _avg_rate(items: List[Dict[str, Any]], answer_type: str) -> float:
    total = len(items)
    if total == 0:
        return 0.0
    return sum(1 for item in items if item.get("predicted_answer_type") == answer_type) / total


def _slice_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_questions": len(items),
        "remember_correct_rate": _avg_rate(items, "remember_correct"),
        "not_remember_rate": _avg_rate(items, "not_remember"),
        "distractor_irrelevant_rate": _avg_rate(items, "distractor_irrelevant"),
    }


def _weighted_average(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(item["num_questions"] for item in summaries)
    if total == 0:
        return {
            "num_questions": 0,
            "remember_correct_rate": 0.0,
            "not_remember_rate": 0.0,
            "distractor_irrelevant_rate": 0.0,
        }
    return {
        "num_questions": total,
        "remember_correct_rate": sum(item["num_questions"] * item["remember_correct_rate"] for item in summaries) / total,
        "not_remember_rate": sum(item["num_questions"] * item["not_remember_rate"] for item in summaries) / total,
        "distractor_irrelevant_rate": sum(item["num_questions"] * item["distractor_irrelevant_rate"] for item in summaries) / total,
    }


def _load_candidate_paths(input_roots: Iterable[Path]) -> List[Path]:
    seen: Set[Path] = set()
    out: List[Path] = []
    for root in input_roots:
        if root.is_file() and root.suffix == ".json":
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(root)
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


def _load_records(input_roots: Iterable[Path], models: Iterable[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    model_set = set(models)
    for path in _load_candidate_paths(input_roots):
        backend = _backend_from_name(path.name)
        if backend is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", "")
        if model not in model_set:
            continue
        persona = _persona_from_name(path.name)
        if persona is None:
            continue
        records.append(
            {
                "path": str(path),
                "name": path.name,
                "backend": backend,
                "model": model,
                "world": data.get("world", ""),
                "persona": persona,
                "ask_period": data.get("ask_period", "Conversation Late Stage"),
                "no_use_restrict_period": data.get("no_use_restrict_period", ""),
                "no_use_release_period": data.get("no_use_release_period", ""),
                "whole_recall_results": data.get("whole_recall_results", []),
                "slot_recall_results": data.get("slot_recall_results", []),
                "summary": data.get("summary", {}),
            }
        )
    return records


def _group_key(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(rec["backend"], rec["model"])].append(rec)
    return grouped


def _aggregate_world(records: List[Dict[str, Any]], world: str, persona_limit: int) -> Dict[str, Any]:
    chosen = [r for r in records if r["world"] == world and r["persona"] < persona_limit]
    return {
        key: _weighted_average([r["summary"][key] for r in chosen if key in r.get("summary", {})])
        for key in SUMMARY_KEYS
    }


def _aggregate_no_store_stage(records: List[Dict[str, Any]], persona_limit: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ask_period in PERIOD_ORDER:
        chosen = [
            r for r in records
            if r["world"] == "no_store" and r["persona"] < persona_limit and r["ask_period"] == ask_period
        ]
        out[ask_period] = {
            key: _weighted_average([r["summary"][key] for r in chosen if key in r.get("summary", {})])
            for key in SUMMARY_KEYS
        }
    return out


def _aggregate_forget_immediate(records: List[Dict[str, Any]], persona_limit: int) -> Dict[str, Any]:
    chosen = [r for r in records if r["world"] == "forget" and r["persona"] < persona_limit]
    whole = []
    slot = []
    for rec in chosen:
        ask = rec["ask_period"]
        whole.extend(
            item for item in rec["whole_recall_results"]
            if item.get("turn_role") == "key" and item.get("forget_stage") == ask
        )
        slot.extend(
            item for item in rec["slot_recall_results"]
            if item.get("turn_role") == "key" and item.get("forget_stage") == ask
        )
    return {
        "whole_recall_key_turns": _slice_summary(whole),
        "slot_recall_key_turns": _slice_summary(slot),
    }


def _aggregate_forget_persistence(records: List[Dict[str, Any]], persona_limit: int) -> Dict[str, Any]:
    chosen = [r for r in records if r["world"] == "forget" and r["persona"] < persona_limit]
    out: Dict[str, Any] = {}
    for ask_period in PERIOD_ORDER:
        whole = []
        slot = []
        for rec in chosen:
            if rec["ask_period"] != ask_period:
                continue
            whole.extend(
                item for item in rec["whole_recall_results"]
                if item.get("turn_role") == "key" and item.get("forget_stage") == "Conversation Early Stage"
            )
            slot.extend(
                item for item in rec["slot_recall_results"]
                if item.get("turn_role") == "key" and item.get("forget_stage") == "Conversation Early Stage"
            )
        out[ask_period] = {
            "whole_recall_key_turns": _slice_summary(whole),
            "slot_recall_key_turns": _slice_summary(slot),
        }
    return out


def _aggregate_forget_timing(records: List[Dict[str, Any]], persona_limit: int) -> Dict[str, Any]:
    chosen = [r for r in records if r["world"] == "forget" and r["persona"] < persona_limit]
    out: Dict[str, Any] = {}
    for forget_stage in PERIOD_ORDER:
        whole = []
        slot = []
        for rec in chosen:
            if rec["ask_period"] != forget_stage:
                continue
            whole.extend(
                item for item in rec["whole_recall_results"]
                if item.get("turn_role") == "key" and item.get("forget_stage") == forget_stage
            )
            slot.extend(
                item for item in rec["slot_recall_results"]
                if item.get("turn_role") == "key" and item.get("forget_stage") == forget_stage
            )
        out[forget_stage] = {
            "whole_recall_key_turns": _slice_summary(whole),
            "slot_recall_key_turns": _slice_summary(slot),
        }
    return out


def _aggregate_no_use_subset(
    records: List[Dict[str, Any]],
    persona_limit: int,
    restrict_period: str,
    ask_period: str,
    release_period: str = "",
) -> Dict[str, Any]:
    chosen = [
        r for r in records
        if r["world"] == "no_use"
        and r["persona"] < persona_limit
        and r["no_use_restrict_period"] == restrict_period
        and r["ask_period"] == ask_period
        and (r["no_use_release_period"] or "") == release_period
    ]
    return {
        key: _weighted_average([r["summary"][key] for r in chosen if key in r.get("summary", {})])
        for key in SUMMARY_KEYS
    }


def _aggregate_slot_type_effects(records: List[Dict[str, Any]], persona_limit: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for world in ["baseline", "no_store", "forget", "no_use"]:
        chosen = [r for r in records if r["world"] == world and r["persona"] < persona_limit]
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"key": [], "probe": []})
        for rec in chosen:
            for item in rec.get("slot_recall_results", []):
                slot_type = str(item.get("slot_type_llm", "")).strip() or classify_slot_type_for_summary(
                    item.get("sensitive_key", ""),
                    item.get("sensitive_value", ""),
                    item.get("question", ""),
                )
                role = item.get("turn_role", "")
                if role in {"key", "probe"}:
                    grouped[slot_type][role].append(item)
        out[world] = {
            slot_type: {
                "key_turns": _slice_summary(payload["key"]),
                "probe_turns": _slice_summary(payload["probe"]),
            }
            for slot_type, payload in sorted(grouped.items())
        }
    return out


def _build_summary(records: List[Dict[str, Any]], other_personas: int, forget_personas: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for (backend, model), group in sorted(_group_key(records).items()):
        summary_key = f"{backend}__{model}"
        summary[summary_key] = {
            "world_difficulty": {
                "baseline": _aggregate_world(group, "baseline", other_personas),
                "no_store": _aggregate_world(group, "no_store", other_personas),
                "forget": _aggregate_world(group, "forget", forget_personas),
                "no_use": _aggregate_world(group, "no_use", other_personas),
            },
            "probe_utility_impact": {
                "baseline": {
                    "whole_recall_probe_turns": _aggregate_world(group, "baseline", other_personas)["whole_recall_probe_turns"],
                    "slot_recall_probe_turns": _aggregate_world(group, "baseline", other_personas)["slot_recall_probe_turns"],
                },
                "no_store": {
                    "whole_recall_probe_turns": _aggregate_world(group, "no_store", other_personas)["whole_recall_probe_turns"],
                    "slot_recall_probe_turns": _aggregate_world(group, "no_store", other_personas)["slot_recall_probe_turns"],
                },
                "forget": {
                    "whole_recall_probe_turns": _aggregate_world(group, "forget", forget_personas)["whole_recall_probe_turns"],
                    "slot_recall_probe_turns": _aggregate_world(group, "forget", forget_personas)["slot_recall_probe_turns"],
                },
                "no_use": {
                    "whole_recall_probe_turns": _aggregate_world(group, "no_use", other_personas)["whole_recall_probe_turns"],
                    "slot_recall_probe_turns": _aggregate_world(group, "no_use", other_personas)["slot_recall_probe_turns"],
                },
            },
            "no_store_test_stage_effect": _aggregate_no_store_stage(group, other_personas),
            "forget_questions": {
                "immediate_effect": _aggregate_forget_immediate(group, forget_personas),
                "persistence": _aggregate_forget_persistence(group, forget_personas),
                "timing_sensitivity": _aggregate_forget_timing(group, forget_personas),
            },
            "no_use_questions": {
                "immediate_suppression": {
                    "no_use@E_test@E": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Early Stage"),
                    "no_use@I_test@I": _aggregate_no_use_subset(group, other_personas, "Conversation Intermediate Stage", "Conversation Intermediate Stage"),
                    "no_use@L_test@L": _aggregate_no_use_subset(group, other_personas, "Conversation Late Stage", "Conversation Late Stage"),
                },
                "persistence": {
                    "no_use@E_test@I": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Intermediate Stage"),
                    "no_use@E_test@L": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Late Stage"),
                },
                "recovery_after_release": {
                    "no_use@E_release@E_test@E": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Early Stage", "Conversation Early Stage"),
                    "no_use@E_release@E_test@I": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Intermediate Stage", "Conversation Early Stage"),
                    "no_use@E_release@E_test@L": _aggregate_no_use_subset(group, other_personas, "Conversation Early Stage", "Conversation Late Stage", "Conversation Early Stage"),
                },
            },
            "slot_type_effects": _aggregate_slot_type_effects(
                group,
                forget_personas if backend != "plain" and any(r["world"] == "forget" for r in group) else other_personas,
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize instruction-control evaluation results.")
    parser.add_argument(
        "--input_roots",
        nargs="+",
        default=[
            "data/test/travelPlanning/specs",
            "eval_results/travelPlanning",
        ],
    )
    parser.add_argument("--output_json", default="eval_results/travelPlanning/instruction_control_summary.json")
    parser.add_argument("--output_md", default="eval_results/travelPlanning/instruction_control_summary.md")
    parser.add_argument("--models", nargs="+", default=["gpt-5.4-mini", "gpt-4o"])
    parser.add_argument("--other_personas", type=int, default=4)
    parser.add_argument("--forget_personas", type=int, default=10)
    args = parser.parse_args()

    records = _load_records([Path(p) for p in args.input_roots], args.models)
    summary = _build_summary(records, args.other_personas, args.forget_personas)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Instruction Control Summary", ""]
    for key, section in summary.items():
        lines.append(f"## {key}")
        lines.append("")
        for block_name, block_value in section.items():
            lines.append(f"### {block_name}")
            lines.append("```json")
            lines.append(json.dumps(block_value, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(str(out_json))
    print(str(args.output_md))


if __name__ == "__main__":
    main()
