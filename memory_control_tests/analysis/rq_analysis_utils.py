from __future__ import annotations

from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json

from memory_control_tests.analysis.summarize_instruction_control_results import _load_records


ROOT = Path("/mnt/yao_data/proj_2026_agent/PersonaMem-main")
DATA_ROOT = ROOT / "data/test/travelPlanning/specs"
EVAL_ROOT = ROOT / "eval_results/travelPlanning"
MODELS = ["gpt-4o", "gpt-5.4-mini"]
EXTRA_MODELS = ["openai/gpt-5.3-chat"]
EXPECTED_COUNTS = {"baseline": 12, "no_store": 12, "forget": 30, "no_use": 32}
PERSONA_LIMITS = {"baseline": 4, "no_store": 4, "forget": 10, "no_use": 4}
STAGE_ORDER = [
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]
STAGE_LABELS = {
    "Conversation Early Stage": "Early",
    "Conversation Intermediate Stage": "Intermediate",
    "Conversation Late Stage": "Late",
}
SYSTEM_ORDER = [
    "GPT-4o",
    "GPT-5.4-mini",
    "GPT-4o + mem0",
    "GPT-5.4-mini + mem0",
    "GPT-4o + A-Mem",
    "GPT-5.4-mini + A-Mem",
    "GPT-4o + LangMem",
    "GPT-5.4-mini + LangMem",
]
SYSTEM_GROUPS = {
    "API Models": ["GPT-4o", "GPT-5.4-mini"],
    "Memory Systems": [
        "GPT-4o + mem0",
        "GPT-5.4-mini + mem0",
        "GPT-4o + A-Mem",
        "GPT-5.4-mini + A-Mem",
        "GPT-4o + LangMem",
        "GPT-5.4-mini + LangMem",
    ],
}
WORLD_COLORS = {"no_store": "#1f77b4", "forget": "#d62728", "no_use": "#2ca02c"}
API_COLORS = {
    "GPT-4o": "#1f77b4",
    "GPT-5.4-mini": "#d62728",
    "GPT-5.3": "#2ca02c",
    "ChatGPT (5.3)": "#2ca02c",
}
SHORT_LABELS = {
    "GPT-4o": "4o",
    "GPT-5.4-mini": "5.4-mini",
    "GPT-4o + mem0": "4o+mem0",
    "GPT-5.4-mini + mem0": "5.4+mem0",
    "GPT-4o + A-Mem": "4o+A-Mem",
    "GPT-5.4-mini + A-Mem": "5.4+A-Mem",
    "GPT-4o + LangMem": "4o+LangMem",
    "GPT-5.4-mini + LangMem": "5.4+LangMem",
    "GPT-5.3 + LangMem": "5.3+LangMem",
    "ChatGPT (5.3)": "ChatGPT",
}


def system_label(rec: Dict[str, Any]) -> str:
    model = "GPT-4o" if rec["model"] == "gpt-4o" else "GPT-5.4-mini"
    if rec["backend"] == "plain":
        return model
    return f"{model} + {rec['backend']}"


def _record_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        rec["backend"],
        rec["model"],
        rec["world"],
        rec["persona"],
        rec["ask_period"],
        rec["no_use_restrict_period"],
        rec["no_use_release_period"],
    )


def load_complete_records() -> List[Dict[str, Any]]:
    raw = _load_records([DATA_ROOT, EVAL_ROOT], MODELS)
    dedup: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for rec in raw:
        label = system_label(rec)
        if "Zep" in label:
            continue
        key = _record_key(rec)
        current = dedup.get(key)
        if current is None or Path(rec["path"]).stat().st_mtime >= Path(current["path"]).stat().st_mtime:
            dedup[key] = rec
    return [rec for rec in dedup.values() if system_label(rec) in SYSTEM_ORDER]


def _summary_rate(summary: Dict[str, Any], key: str, field: str) -> float:
    return float(summary.get(key, {}).get(field, 0.0))


def _record_probe_tpr(rec: Dict[str, Any]) -> float:
    s = rec["summary"]
    vals = [
        _summary_rate(s, "whole_recall_probe_turns", "remember_correct_rate"),
        _summary_rate(s, "slot_recall_probe_turns", "remember_correct_rate"),
    ]
    return mean(vals)


def _record_key_fpr(rec: Dict[str, Any]) -> Optional[float]:
    if rec["world"] == "baseline":
        return None
    s = rec["summary"]
    vals = [
        _summary_rate(s, "whole_recall_key_turns", "remember_correct_rate"),
        _summary_rate(s, "slot_recall_key_turns", "remember_correct_rate"),
    ]
    return mean(vals)


def _world_records(records: List[Dict[str, Any]], label: str, world: str) -> List[Dict[str, Any]]:
    limit = PERSONA_LIMITS[world]
    return [
        rec for rec in records
        if system_label(rec) == label and rec["world"] == world and rec["persona"] < limit
    ]


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else float("nan")


def _combined_answer_rate(items: List[Dict[str, Any]], answer_type: str) -> float:
    if not items:
        return float("nan")
    hits = sum(1 for item in items if item.get("predicted_answer_type") == answer_type)
    return hits / len(items)


def completion_rows(records: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = ["System", "baseline", "no_store", "forget", "no_use"]
    rows = []
    for label in SYSTEM_ORDER:
        row = [label]
        for world, expected in EXPECTED_COUNTS.items():
            got = sum(1 for rec in records if system_label(rec) == label and rec["world"] == world)
            row.append(f"{got}/{expected}")
        rows.append(row)
    return headers, rows


def q1_q2_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for label in SYSTEM_ORDER:
        baseline_tpr = _safe_mean([_record_probe_tpr(rec) for rec in _world_records(records, label, "baseline")])
        row: Dict[str, Any] = {
            "System": label,
            "Baseline_FPR": None,
            "Baseline_TPR": baseline_tpr,
        }
        for world in ["no_store", "forget", "no_use"]:
            world_records = _world_records(records, label, world)
            fpr = _safe_mean([x for x in (_record_key_fpr(rec) for rec in world_records) if x is not None])
            tpr = _safe_mean([_record_probe_tpr(rec) for rec in world_records])
            row[f"{world}_FPR"] = fpr
            row[f"{world}_DeltaTPR"] = tpr - baseline_tpr
        rows.append(row)
    return rows


def q1_q2_mean_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"System": "Mean", "Baseline_FPR": None}
    out["Baseline_TPR"] = _safe_mean([row["Baseline_TPR"] for row in rows])
    for world in ["no_store", "forget", "no_use"]:
        out[f"{world}_FPR"] = _safe_mean([row[f"{world}_FPR"] for row in rows])
        out[f"{world}_DeltaTPR"] = _safe_mean([row[f"{world}_DeltaTPR"] for row in rows])
    return out


def q1_q2_html_table(records: List[Dict[str, Any]]) -> str:
    rows = q1_q2_rows(records)
    rows.append(q1_q2_mean_row(rows))
    lines = [
        "<table>",
        "<thead>",
        "<tr>",
        "<th rowspan='2'>System</th>",
        "<th colspan='1'>Baseline</th>",
        "<th colspan='1'>No-store</th>",
        "<th colspan='1'>Forget</th>",
        "<th colspan='1'>No-use</th>",
        "</tr>",
        "<tr>",
        "<th>TPR</th>",
        "<th>FPR (ΔTPR)</th>",
        "<th>FPR (ΔTPR)</th>",
        "<th>FPR (ΔTPR)</th>",
        "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append("<tr>")
        lines.append(f"<td>{escape(row['System'])}</td>")
        lines.append(f"<td>{row['Baseline_TPR']:.2f}</td>")
        for world in ["no_store", "forget", "no_use"]:
            lines.append(f"<td>{row[f'{world}_FPR']:.2f} ({row[f'{world}_DeltaTPR']:+.2f})</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def q1_q2_markdown_table(records: List[Dict[str, Any]]) -> str:
    rows = q1_q2_rows(records)
    rows.append(q1_q2_mean_row(rows))
    gpt53_records = load_gpt53_langmem_records()

    display_rows: List[List[str]] = []
    display_rows.append(["API Models", "", "", "", ""])
    for label in ["GPT-4o", "GPT-5.4-mini"]:
        row = next(r for r in rows if r["System"] == label)
        display_rows.append([
            label,
            f"{row['Baseline_TPR']:.2f}",
            f"{row['no_store_FPR']:.2f} ({row['no_store_DeltaTPR']:+.2f})",
            f"{row['forget_FPR']:.2f} ({row['forget_DeltaTPR']:+.2f})",
            f"{row['no_use_FPR']:.2f} ({row['no_use_DeltaTPR']:+.2f})",
        ])

    display_rows.append(["Memory Agent", "", "", "", ""])
    for model in ["GPT-4o", "GPT-5.4-mini"]:
        display_rows.append([model, "", "", "", ""])
        for variant in ["mem0", "A-Mem", "LangMem"]:
            label = f"{model} + {variant}"
            row = next(r for r in rows if r["System"] == label)
            display_rows.append([
                f"  -{variant}",
                f"{row['Baseline_TPR']:.2f}",
                f"{row['no_store_FPR']:.2f} ({row['no_store_DeltaTPR']:+.2f})",
                f"{row['forget_FPR']:.2f} ({row['forget_DeltaTPR']:+.2f})",
                f"{row['no_use_FPR']:.2f} ({row['no_use_DeltaTPR']:+.2f})",
            ])

    g53_base = [rec for rec in gpt53_records if rec["backend"] == "LangMem" and rec["model"] == "openai/gpt-5.3-chat" and rec["world"] == "baseline"]
    g53_no_store = [rec for rec in gpt53_records if rec["backend"] == "LangMem" and rec["model"] == "openai/gpt-5.3-chat" and rec["world"] == "no_store"]
    g53_forget = [rec for rec in gpt53_records if rec["backend"] == "LangMem" and rec["model"] == "openai/gpt-5.3-chat" and rec["world"] == "forget"]
    g53_baseline_tpr = _safe_mean([_record_probe_tpr(rec) for rec in g53_base]) if g53_base else None
    g53_no_store_fpr = _safe_mean([_record_key_fpr(rec) for rec in g53_no_store if _record_key_fpr(rec) is not None]) if g53_no_store else None
    g53_no_store_tpr = _safe_mean([_record_probe_tpr(rec) for rec in g53_no_store]) if g53_no_store else None
    g53_forget_fpr = _safe_mean([_record_key_fpr(rec) for rec in g53_forget if _record_key_fpr(rec) is not None]) if g53_forget else None
    g53_forget_tpr = _safe_mean([_record_probe_tpr(rec) for rec in g53_forget]) if g53_forget else None
    display_rows.append(["GPT-5.3", "", "", "", ""])
    display_rows.append([
        "  -LangMem",
        _fmt_cell("baseline", g53_baseline_tpr, None, g53_baseline_tpr),
        _fmt_cell("no_store", g53_baseline_tpr, g53_no_store_fpr, g53_no_store_tpr),
        _fmt_cell("forget", g53_baseline_tpr, g53_forget_fpr, g53_forget_tpr),
        "",
    ])

    chat_base_fpr, chat_base_tpr = chatgpt_world_metrics("baseline")
    chat_ns_fpr, chat_ns_tpr = chatgpt_world_metrics("no_store")
    chat_fg_fpr, chat_fg_tpr = chatgpt_world_metrics("forget")
    display_rows.append(["ChatGPT Web", "", "", "", ""])
    display_rows.append([
        "ChatGPT (5.3)",
        _fmt_cell("baseline", chat_base_tpr, chat_base_fpr, chat_base_tpr, chatgpt_style=True),
        _fmt_cell("no_store", chat_base_tpr, chat_ns_fpr, chat_ns_tpr, chatgpt_style=True),
        _fmt_cell("forget", chat_base_tpr, chat_fg_fpr, chat_fg_tpr, chatgpt_style=True),
        "",
    ])

    mean_row = rows[-1]
    display_rows.append(["Mean", f"{mean_row['Baseline_TPR']:.2f}",
                         f"{mean_row['no_store_FPR']:.2f} ({mean_row['no_store_DeltaTPR']:+.2f})",
                         f"{mean_row['forget_FPR']:.2f} ({mean_row['forget_DeltaTPR']:+.2f})",
                         f"{mean_row['no_use_FPR']:.2f} ({mean_row['no_use_DeltaTPR']:+.2f})"])

    headers = ["System", "Baseline", "No-store", "Forget", "No-use"]
    widths = [len(h) for h in headers]
    for row in display_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for row in display_rows:
        if row[1:] == ["", "", "", ""]:
            lines.append(row[0])
        else:
            lines.append(fmt_row(row))
    return "```text\n" + "\n".join(lines) + "\n```"


def _aggregate_system_world(records: List[Dict[str, Any]], label: str, world: str) -> Tuple[Optional[float], Optional[float]]:
    chosen = _world_records(records, label, world)
    if not chosen:
        return None, None
    tpr = _safe_mean([_record_probe_tpr(rec) for rec in chosen])
    if world == "baseline":
        return None, tpr
    fpr = _safe_mean([x for x in (_record_key_fpr(rec) for rec in chosen) if x is not None])
    return fpr, tpr


def load_gpt53_langmem_records() -> List[Dict[str, Any]]:
    return _load_records([DATA_ROOT, EVAL_ROOT], EXTRA_MODELS)


def _chatgpt_results_root() -> Path:
    return ROOT / "results/chatgpt_web_results/travelPlanning"


def _iter_chatgpt_rows(world: str) -> List[Dict[str, Any]]:
    root = _chatgpt_results_root()
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for sample_dir in sorted(root.iterdir()):
        path = sample_dir / f"test_type_{world}" / "results.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is None:
                rows.append(row)
    return rows


def chatgpt_world_metrics(world: str) -> Tuple[Optional[float], Optional[float]]:
    rows = _iter_chatgpt_rows(world)
    if not rows:
        return None, None
    probe = [row for row in rows if row.get("turn_role") == "probe"]
    key = [row for row in rows if row.get("turn_role") == "key"]
    tpr = _combined_answer_rate(probe, "remember_correct") if probe else None
    if world == "baseline":
        return None, tpr
    fpr = _combined_answer_rate(key, "remember_correct") if key else None
    return fpr, tpr


def _fmt_cell(world: str, baseline_tpr: Optional[float], fpr: Optional[float], tpr: Optional[float], *, chatgpt_style: bool = False) -> str:
    if world == "baseline":
        return "" if baseline_tpr is None else f"{baseline_tpr:.2f}"
    if fpr is None:
        return ""
    if tpr is None:
        return f"{fpr:.2f}"
    if chatgpt_style:
        return f"{fpr:.2f} (TPR={tpr:.2f})"
    if baseline_tpr is None:
        return f"{fpr:.2f} ({tpr:.2f})"
    delta = tpr - baseline_tpr
    return f"{fpr:.2f} ({delta:+.2f})"


def table1_grouped_html(records: List[Dict[str, Any]]) -> str:
    gpt53_records = load_gpt53_langmem_records()

    def row_html(system: str, variant: str, baseline_label: str, no_store_label: str, forget_label: str) -> str:
        return (
            "<tr>"
            f"<td>{escape(system)}</td>"
            f"<td>{escape(variant)}</td>"
            f"<td>{baseline_label}</td>"
            f"<td>{no_store_label}</td>"
            f"<td>{forget_label}</td>"
            "</tr>"
        )

    def from_label(label: str) -> Tuple[str, str, str]:
        _, baseline_tpr = _aggregate_system_world(records, label, "baseline")
        no_store_fpr, no_store_tpr = _aggregate_system_world(records, label, "no_store")
        forget_fpr, forget_tpr = _aggregate_system_world(records, label, "forget")
        return (
            _fmt_cell("baseline", baseline_tpr, None, baseline_tpr),
            _fmt_cell("no_store", baseline_tpr, no_store_fpr, no_store_tpr),
            _fmt_cell("forget", baseline_tpr, forget_fpr, forget_tpr),
        )

    html = [
        "<table style='border-collapse:collapse; font-family:Arial, sans-serif; font-size:14px;'>",
        "<thead>",
        "<tr>",
        "<th style='border:1px solid #d9d9d9; padding:8px; text-align:left;'>System</th>",
        "<th style='border:1px solid #d9d9d9; padding:8px; text-align:left;'></th>",
        "<th style='border:1px solid #d9d9d9; padding:8px; text-align:center;'>Baseline</th>",
        "<th style='border:1px solid #d9d9d9; padding:8px; text-align:center;'>No-store</th>",
        "<th style='border:1px solid #d9d9d9; padding:8px; text-align:center;'>Forget</th>",
        "</tr>",
        "</thead>",
        "<tbody>",
        "<tr><td colspan='5' style='border:1px solid #d9d9d9; padding:8px; font-weight:700; background:#fafafa;'>API Models</td></tr>",
    ]

    for label in ["GPT-4o", "GPT-5.4-mini"]:
        b, ns, fg = from_label(label)
        html.append(row_html(label, "", b, ns, fg))

    html.append("<tr><td colspan='5' style='border:1px solid #d9d9d9; padding:8px; font-weight:700; background:#fafafa;'>Memory Agent</td></tr>")
    for model in ["GPT-4o", "GPT-5.4-mini"]:
        html.append(
            f"<tr><td style='border:1px solid #d9d9d9; padding:8px; font-weight:700;'>{escape(model)}</td>"
            "<td style='border:1px solid #d9d9d9; padding:8px;'></td>"
            "<td style='border:1px solid #d9d9d9; padding:8px;'></td>"
            "<td style='border:1px solid #d9d9d9; padding:8px;'></td>"
            "<td style='border:1px solid #d9d9d9; padding:8px;'></td></tr>"
        )
        for variant in ["mem0", "A-Mem", "LangMem"]:
            label = f"{model} + {variant}"
            b, ns, fg = from_label(label)
            html.append(row_html("", f"-{variant}", b, ns, fg))

    g53_label = "GPT-5.3"
    g53_base = [rec for rec in gpt53_records if rec["backend"] == "LangMem" and rec["model"] == "openai/gpt-5.3-chat" and rec["world"] == "baseline"]
    g53_no_store = [rec for rec in gpt53_records if rec["backend"] == "LangMem" and rec["model"] == "openai/gpt-5.3-chat" and rec["world"] == "no_store"]
    g53_forget = [rec for rec in gpt53_records if rec["backend"] == "LangMem" and rec["model"] == "openai/gpt-5.3-chat" and rec["world"] == "forget"]
    g53_baseline_tpr = _safe_mean([_record_probe_tpr(rec) for rec in g53_base]) if g53_base else None
    g53_no_store_fpr = _safe_mean([_record_key_fpr(rec) for rec in g53_no_store if _record_key_fpr(rec) is not None]) if g53_no_store else None
    g53_no_store_tpr = _safe_mean([_record_probe_tpr(rec) for rec in g53_no_store]) if g53_no_store else None
    g53_forget_fpr = _safe_mean([_record_key_fpr(rec) for rec in g53_forget if _record_key_fpr(rec) is not None]) if g53_forget else None
    g53_forget_tpr = _safe_mean([_record_probe_tpr(rec) for rec in g53_forget]) if g53_forget else None
    html.append(
        f"<tr><td style='border:1px solid #d9d9d9; padding:8px; font-weight:700;'>{g53_label}</td>"
        "<td style='border:1px solid #d9d9d9; padding:8px;'>-LangMem</td>"
        f"<td style='border:1px solid #d9d9d9; padding:8px;'>{_fmt_cell('baseline', g53_baseline_tpr, None, g53_baseline_tpr)}</td>"
        f"<td style='border:1px solid #d9d9d9; padding:8px;'>{_fmt_cell('no_store', g53_baseline_tpr, g53_no_store_fpr, g53_no_store_tpr)}</td>"
        f"<td style='border:1px solid #d9d9d9; padding:8px;'>{_fmt_cell('forget', g53_baseline_tpr, g53_forget_fpr, g53_forget_tpr)}</td></tr>"
    )

    chat_base_fpr, chat_base_tpr = chatgpt_world_metrics("baseline")
    chat_ns_fpr, chat_ns_tpr = chatgpt_world_metrics("no_store")
    chat_fg_fpr, chat_fg_tpr = chatgpt_world_metrics("forget")
    html.append("<tr><td colspan='5' style='border:1px solid #d9d9d9; padding:8px; font-weight:700; background:#fafafa;'>ChatGPT Web</td></tr>")
    html.append(
        f"<tr><td style='border:1px solid #d9d9d9; padding:8px; font-weight:700;'>ChatGPT (5.3)</td>"
        "<td style='border:1px solid #d9d9d9; padding:8px;'></td>"
        f"<td style='border:1px solid #d9d9d9; padding:8px;'>{_fmt_cell('baseline', chat_base_tpr, chat_base_fpr, chat_base_tpr, chatgpt_style=True)}</td>"
        f"<td style='border:1px solid #d9d9d9; padding:8px;'>{_fmt_cell('no_store', chat_base_tpr, chat_ns_fpr, chat_ns_tpr, chatgpt_style=True)}</td>"
        f"<td style='border:1px solid #d9d9d9; padding:8px;'>{_fmt_cell('forget', chat_base_tpr, chat_fg_fpr, chat_fg_tpr, chatgpt_style=True)}</td></tr>"
    )
    html.append("</tbody></table>")
    return "\n".join(html)


def tradeoff_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for label in SYSTEM_ORDER:
        baseline_tpr = _safe_mean([_record_probe_tpr(rec) for rec in _world_records(records, label, "baseline")])
        group = "API Models" if label in SYSTEM_GROUPS["API Models"] else "Memory Systems"
        for world in ["no_store", "forget", "no_use"]:
            world_records = _world_records(records, label, world)
            rows.append(
                {
                    "System": label,
                    "World": world,
                    "TPR": _safe_mean([_record_probe_tpr(rec) for rec in world_records]),
                    "Delta_TPR": _safe_mean([_record_probe_tpr(rec) for rec in world_records]) - baseline_tpr,
                    "FPR": _safe_mean([x for x in (_record_key_fpr(rec) for rec in world_records) if x is not None]),
                    "Group": group,
                    "API": "GPT-4o" if "GPT-4o" in label else "GPT-5.4-mini",
                }
            )
    return rows


def no_store_stage_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for label in SYSTEM_ORDER:
        group = "API Models" if label in SYSTEM_GROUPS["API Models"] else "Memory Systems"
        for stage in STAGE_ORDER:
            chosen = [rec for rec in _world_records(records, label, "no_store") if rec["ask_period"] == stage]
            key_items: List[Dict[str, Any]] = []
            for rec in chosen:
                key_items.extend(
                    item for item in rec["whole_recall_results"]
                    if item.get("turn_role") == "key"
                )
                key_items.extend(
                    item for item in rec["slot_recall_results"]
                    if item.get("turn_role") == "key"
                )
            rows.append(
                {
                    "System": label,
                    "Stage": STAGE_LABELS[stage],
                    "TPR": _safe_mean([_record_probe_tpr(rec) for rec in chosen]),
                    "FPR": _safe_mean([x for x in (_record_key_fpr(rec) for rec in chosen) if x is not None]),
                    "TNR": _combined_answer_rate(key_items, "not_remember"),
                    "Group": group,
                }
            )
    return rows


def forget_immediate_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for label in SYSTEM_ORDER:
        group = "API Models" if label in SYSTEM_GROUPS["API Models"] else "Memory Systems"
        for stage in STAGE_ORDER:
            chosen = [rec for rec in _world_records(records, label, "forget") if rec["ask_period"] == stage]
            items: List[Dict[str, Any]] = []
            for rec in chosen:
                items.extend(
                    item for item in rec["whole_recall_results"]
                    if item.get("turn_role") == "key" and item.get("forget_stage") == stage
                )
                items.extend(
                    item for item in rec["slot_recall_results"]
                    if item.get("turn_role") == "key" and item.get("forget_stage") == stage
                )
            rows.append(
                {
                    "System": label,
                    "Stage": STAGE_LABELS[stage],
                    "TPR": _combined_answer_rate(items, "remember_correct"),
                    "FPR": _combined_answer_rate(items, "remember_correct"),
                    "TNR": _combined_answer_rate(items, "not_remember"),
                    "Group": group,
                }
            )
    return rows


def forget_persistence_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for label in SYSTEM_ORDER:
        group = "API Models" if label in SYSTEM_GROUPS["API Models"] else "Memory Systems"
        for stage in STAGE_ORDER:
            chosen = [rec for rec in _world_records(records, label, "forget") if rec["ask_period"] == stage]
            items: List[Dict[str, Any]] = []
            for rec in chosen:
                items.extend(
                    item for item in rec["whole_recall_results"]
                    if item.get("turn_role") == "key" and item.get("forget_stage") == "Conversation Early Stage"
                )
                items.extend(
                    item for item in rec["slot_recall_results"]
                    if item.get("turn_role") == "key" and item.get("forget_stage") == "Conversation Early Stage"
                )
            rows.append(
                {
                    "System": label,
                    "Stage": STAGE_LABELS[stage],
                    "TPR": _combined_answer_rate(items, "remember_correct"),
                    "FPR": _combined_answer_rate(items, "remember_correct"),
                    "TNR": _combined_answer_rate(items, "not_remember"),
                    "Group": group,
                }
            )
    return rows


def slot_metric_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for rec in records:
        if system_label(rec) not in SYSTEM_ORDER:
            continue
        for item in rec["slot_recall_results"]:
            if item.get("turn_role") != "key":
                continue
            slot_type = str(item.get("slot_type_llm", "")).strip()
            if not slot_type:
                continue
            world = rec["world"]
            if world == "baseline":
                metric = "TPR"
                value = 1.0 if item.get("predicted_answer_type") == "remember_correct" else 0.0
            else:
                metric = "TNR"
                value = 1.0 if item.get("predicted_answer_type") == "not_remember" else 0.0
            rows.append(
                {
                    "System": system_label(rec),
                    "World": world,
                    "slot_type": slot_type,
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def slot_metric_rows_for_metric(records: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    metric = metric.upper()
    rows = []
    for rec in records:
        if system_label(rec) not in SYSTEM_ORDER:
            continue
        for item in rec["slot_recall_results"]:
            if item.get("turn_role") != "key":
                continue
            slot_type = str(item.get("slot_type_llm", "")).strip()
            if not slot_type:
                continue
            pred = item.get("predicted_answer_type")
            if metric == "TPR":
                value = 1.0 if pred == "remember_correct" else 0.0
            elif metric == "TNR":
                value = 1.0 if pred == "not_remember" else 0.0
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            rows.append(
                {
                    "System": system_label(rec),
                    "World": rec["world"],
                    "slot_type": slot_type,
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def ordered_slot_types_by_forget(rows: List[Dict[str, Any]]) -> List[str]:
    slot_types = sorted({row["slot_type"] for row in rows})
    baseline_scores = {
        slot_type: _safe_mean(
            [row["value"] for row in rows if row["slot_type"] == slot_type and row["World"] == "baseline"]
        )
        for slot_type in slot_types
    }
    slot_types.sort(
        key=lambda s: (
            baseline_scores[s] != baseline_scores[s],
            -(baseline_scores[s] if baseline_scores[s] == baseline_scores[s] else -1.0),
        ),
    )
    return slot_types


def slot_heatmap_matrix(
    rows: List[Dict[str, Any]],
    worlds: Optional[List[str]] = None,
    slot_types: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[List[float]]]:
    slot_types = slot_types or ordered_slot_types_by_forget(rows)
    worlds = worlds or ["baseline", "no_store", "forget", "no_use"]
    matrix: List[List[float]] = []
    for slot_type in slot_types:
        line = []
        for world in worlds:
            values = [row["value"] for row in rows if row["slot_type"] == slot_type and row["World"] == world]
            line.append(_safe_mean(values))
        matrix.append(line)
    return slot_types, worlds, matrix


def ranked_forget_slot_rows(rows: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    slot_types = sorted({row["slot_type"] for row in rows if row["World"] == "forget"})
    ranked = []
    for slot_type in slot_types:
        values = [row["value"] for row in rows if row["slot_type"] == slot_type and row["World"] == "forget"]
        ranked.append((slot_type, _safe_mean(values)))
    ranked.sort(key=lambda x: x[1])
    return ranked


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def rows_to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = columns
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _svg_wrap(width: int, height: int, inner: List[str]) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
        + "".join(inner)
        + "</svg>"
    )


def _point_shape(cx: float, cy: float, group: str, color: str) -> str:
    if group == "API Models":
        return f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5' fill='{color}' opacity='0.9'/>"
    arms = []
    outer = 7
    inner = 3
    import math
    pts = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        radius = outer if i % 2 == 0 else inner
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        pts.append(f"{x:.1f},{y:.1f}")
    return f"<polygon points='{' '.join(pts)}' fill='{color}' opacity='0.9'/>"


def _short_label(label: str) -> str:
    return SHORT_LABELS.get(label, label)


def scatter_svg(rows: List[Dict[str, Any]], width: int = 720, height: int = 576, world_filter: Optional[str] = None) -> str:
    if world_filter is not None:
        rows = [row for row in rows if row["World"] == world_filter]
    left, top, right, bottom = 62, 32, 165, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 0.0, max(row["FPR"] for row in rows) * 1.15
    y_min, y_max = min(row["TPR"] for row in rows) * 0.95, 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min + 1e-9) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min + 1e-9) * plot_h

    inner = [
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='white' stroke='#999'/>",
        f"<text x='{width/2}' y='18' text-anchor='middle' font-size='15'>Q3. Trade-off{'' if world_filter is None else f' ({escape(world_filter)})'}</text>",
        f"<text x='{width/2}' y='{height-10}' text-anchor='middle' font-size='11'>FPR</text>",
        f"<text x='16' y='{height/2}' transform='rotate(-90 16 {height/2})' text-anchor='middle' font-size='11'>TPR</text>",
    ]
    for row in rows:
        x, y = sx(row["FPR"]), sy(row["TPR"])
        color = API_COLORS[row["API"]]
        inner.append(_point_shape(x, y, row["Group"], color))
        label = escape(_short_label(row["System"]))
        inner.append(f"<text x='{x+7:.1f}' y='{y-5:.1f}' font-size='8'>{label}</text>")
    legend_y = 62
    inner.append(f"<text x='{width-145}' y='{legend_y-16}' font-size='10' font-weight='700'>Color</text>")
    for api, color in [("GPT-4o", API_COLORS["GPT-4o"]), ("GPT-5.4-mini", API_COLORS["GPT-5.4-mini"]), ("GPT-5.3", API_COLORS["GPT-5.3"])]:
        inner.append(f"<circle cx='{width-140}' cy='{legend_y}' r='5' fill='{color}'/>")
        inner.append(f"<text x='{width-128}' y='{legend_y+4}' font-size='10'>{escape(api)}</text>")
        legend_y += 22
    legend_y += 8
    inner.append(f"<text x='{width-145}' y='{legend_y-16}' font-size='10' font-weight='700'>Shape</text>")
    inner.append(f"<circle cx='{width-140}' cy='{legend_y}' r='5' fill='#666'/>")
    inner.append(f"<text x='{width-128}' y='{legend_y+4}' font-size='10'>API</text>")
    legend_y += 22
    inner.append(_point_shape(width - 140, legend_y, "Memory Systems", "#666"))
    inner.append(f"<text x='{width-128}' y='{legend_y+4}' font-size='10'>Agent</text>")
    return _svg_wrap(width, height, inner)


def two_panel_line_svg(rows: List[Dict[str, Any]], title: str, width: int = 720, height: int = 576) -> str:
    margins = dict(left=54, top=40, right=150, bottom=46)
    panel_gap = 50
    panel_w = (width - margins["left"] - margins["right"] - panel_gap) / 2
    panel_h = height - margins["top"] - margins["bottom"]
    stage_names = ["Early", "Intermediate", "Late"]
    stage_x = {name: idx for idx, name in enumerate(stage_names)}

    def draw_panel(x0: float, metric: str, subtitle: str) -> List[str]:
        vals = [row[metric] for row in rows]
        ymin = 0.0
        ymax = max(vals) * 1.1 if vals else 1.0
        if ymax < 0.1:
            ymax = 0.1

        def sx(stage: str) -> float:
            return x0 + (stage_x[stage] / 2) * panel_w

        def sy(value: float) -> float:
            return margins["top"] + panel_h - (value - ymin) / (ymax - ymin + 1e-9) * panel_h

        inner = [
            f"<rect x='{x0}' y='{margins['top']}' width='{panel_w}' height='{panel_h}' fill='white' stroke='#999'/>",
            f"<text x='{x0 + panel_w/2}' y='28' text-anchor='middle' font-size='12'>{escape(subtitle)}</text>",
        ]
        for idx, label in enumerate(SYSTEM_ORDER):
            sub = [row for row in rows if row["System"] == label]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: stage_x[r["Stage"]])
            points = " ".join(f"{sx(r['Stage']):.1f},{sy(r[metric]):.1f}" for r in sub)
            color = API_COLORS["GPT-4o"] if "GPT-4o" in label else API_COLORS.get("GPT-5.4-mini", "#d62728")
            inner.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/>")
            for r in sub:
                inner.append(f"<circle cx='{sx(r['Stage']):.1f}' cy='{sy(r[metric]):.1f}' r='3' fill='{color}'/>")
        for stage in stage_names:
            inner.append(f"<text x='{sx(stage):.1f}' y='{height-16}' text-anchor='middle' font-size='10'>{stage}</text>")
        return inner

    inner = [f"<text x='{width/2}' y='18' text-anchor='middle' font-size='15'>{escape(title)}</text>"]
    inner.extend(draw_panel(margins["left"], "TPR", "(a) TPR vs stage"))
    inner.extend(draw_panel(margins["left"] + panel_w + panel_gap, "FPR", "(b) FPR vs stage"))
    legend_y = 70
    for label in SYSTEM_ORDER:
        color = API_COLORS["GPT-4o"] if "GPT-4o" in label else API_COLORS["GPT-5.4-mini"]
        inner.append(f"<line x1='{width-175}' y1='{legend_y}' x2='{width-155}' y2='{legend_y}' stroke='{color}' stroke-width='2'/>")
        inner.append(f"<circle cx='{width-165}' cy='{legend_y}' r='3' fill='{color}'/>")
        inner.append(f"<text x='{width-150}' y='{legend_y+4}' font-size='9'>{escape(_short_label(label))}</text>")
        legend_y += 14
    return _svg_wrap(width, height, inner)


def forget_stage_svg(rows: List[Dict[str, Any]], title: str, width: int = 720, height: int = 576) -> str:
    margins = dict(left=54, top=40, right=150, bottom=46)
    panel_gap = 50
    panel_w = (width - margins["left"] - margins["right"] - panel_gap) / 2
    panel_h = height - margins["top"] - margins["bottom"]
    stage_names = ["Early", "Intermediate", "Late"]
    stage_x = {name: idx for idx, name in enumerate(stage_names)}

    def draw_panel(x0: float, metric: str, subtitle: str) -> List[str]:
        vals = [row[metric] for row in rows]
        ymin = 0.0
        ymax = max(vals) * 1.1 if vals else 1.0
        if ymax < 0.1:
            ymax = 0.1

        def sx(stage: str) -> float:
            return x0 + (stage_x[stage] / 2) * panel_w

        def sy(value: float) -> float:
            return margins["top"] + panel_h - (value - ymin) / (ymax - ymin + 1e-9) * panel_h

        inner = [
            f"<rect x='{x0}' y='{margins['top']}' width='{panel_w}' height='{panel_h}' fill='white' stroke='#999'/>",
            f"<text x='{x0 + panel_w/2}' y='28' text-anchor='middle' font-size='12'>{escape(subtitle)}</text>",
        ]
        for label in SYSTEM_ORDER:
            sub = [row for row in rows if row["System"] == label]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: stage_x[r["Stage"]])
            points = " ".join(f"{sx(r['Stage']):.1f},{sy(r[metric]):.1f}" for r in sub)
            color = API_COLORS["GPT-4o"] if "GPT-4o" in label else API_COLORS["GPT-5.4-mini"]
            inner.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/>")
            for r in sub:
                inner.append(f"<circle cx='{sx(r['Stage']):.1f}' cy='{sy(r[metric]):.1f}' r='3' fill='{color}'/>")
        for stage in stage_names:
            inner.append(f"<text x='{sx(stage):.1f}' y='{height-16}' text-anchor='middle' font-size='10'>{stage}</text>")
        return inner

    inner = [f"<text x='{width/2}' y='18' text-anchor='middle' font-size='15'>{escape(title)}</text>"]
    inner.extend(draw_panel(margins["left"], "TNR", "(a) TNR vs stage"))
    inner.extend(draw_panel(margins["left"] + panel_w + panel_gap, "FPR", "(b) FPR vs stage"))
    legend_y = 70
    for label in SYSTEM_ORDER:
        color = API_COLORS["GPT-4o"] if "GPT-4o" in label else API_COLORS["GPT-5.4-mini"]
        inner.append(f"<line x1='{width-175}' y1='{legend_y}' x2='{width-155}' y2='{legend_y}' stroke='{color}' stroke-width='2'/>")
        inner.append(f"<circle cx='{width-165}' cy='{legend_y}' r='3' fill='{color}'/>")
        inner.append(f"<text x='{width-150}' y='{legend_y+4}' font-size='9'>{escape(_short_label(label))}</text>")
        legend_y += 14
    return _svg_wrap(width, height, inner)


def heatmap_svg(row_labels: List[str], col_labels: List[str], matrix: List[List[float]], title: str, width: int = 720, height: int = 576) -> str:
    left, top, right, bottom = 220, 50, 20, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / max(1, len(col_labels))
    cell_h = plot_h / max(1, len(row_labels))

    def color(val: float) -> str:
        if val != val:
            return "#eeeeee"
        blue = int(255 * (1 - val))
        green = int(220 * val)
        red = int(80 + 150 * val)
        return f"rgb({red},{green},{blue})"

    inner = [f"<text x='{width/2}' y='20' text-anchor='middle' font-size='16'>{escape(title)}</text>"]
    for i, label in enumerate(row_labels):
        y = top + i * cell_h
        inner.append(f"<text x='{left-8}' y='{y + cell_h*0.65:.1f}' text-anchor='end' font-size='11'>{escape(label)}</text>")
    for j, label in enumerate(col_labels):
        x = left + j * cell_w
        inner.append(f"<text x='{x + cell_w/2:.1f}' y='{top-8}' text-anchor='middle' font-size='11'>{escape(label)}</text>")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            x = left + j * cell_w
            y = top + i * cell_h
            fill = color(val)
            inner.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_w:.1f}' height='{cell_h:.1f}' fill='{fill}' stroke='white'/>")
            if val == val:
                inner.append(f"<text x='{x + cell_w/2:.1f}' y='{y + cell_h/2 + 4:.1f}' text-anchor='middle' font-size='10' fill='white'>{val:.2f}</text>")
    return _svg_wrap(width, height, inner)


def ranked_bar_svg(rows: List[Tuple[str, float]], title: str, width: int = 720, height: int = 576) -> str:
    left, top, right, bottom = 260, 45, 30, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_h = plot_h / max(1, len(rows))
    xmax = max((val for _, val in rows), default=1.0) * 1.1
    inner = [f"<text x='{width/2}' y='20' text-anchor='middle' font-size='16'>{escape(title)}</text>"]
    for i, (label, val) in enumerate(rows):
        y = top + i * bar_h
        w = plot_w * (val / (xmax + 1e-9))
        inner.append(f"<text x='{left-8}' y='{y + bar_h*0.65:.1f}' text-anchor='end' font-size='11'>{escape(label)}</text>")
        inner.append(f"<rect x='{left}' y='{y + 3:.1f}' width='{w:.1f}' height='{max(6, bar_h-6):.1f}' fill='#d62728'/>")
        inner.append(f"<text x='{left + w + 6:.1f}' y='{y + bar_h*0.65:.1f}' font-size='10'>{val:.2f}</text>")
    inner.append(f"<text x='{width/2}' y='{height-10}' text-anchor='middle' font-size='12'>FPR under forget</text>")
    return _svg_wrap(width, height, inner)


def setup_pretty_matplotlib() -> Tuple[Any, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8")
    plt.rcParams["axes.facecolor"] = "#f5f5f5"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#bbbbbb"
    plt.rcParams["grid.color"] = "#d9d9d9"
    plt.rcParams["grid.alpha"] = 0.6
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "#cccccc"
    return plt, np


def plot_tradeoff_matplotlib(rows: List[Dict[str, Any]], world_filter: str) -> Any:
    plt, _ = setup_pretty_matplotlib()
    rows = [row for row in rows if row["World"] == world_filter]
    fig, ax = plt.subplots(figsize=(5, 4))
    label_fs = 8
    title_fs = 8

    for row in rows:
        color = API_COLORS[row["API"]]
        marker = "o" if row["Group"] == "API Models" else "*"
        size = 60 if marker == "o" else 110
        ax.scatter(row["FPR"], row["TPR"], color=color, marker=marker, s=size, alpha=0.9)
        ax.text(
            row["FPR"] + 0.006,
            row["TPR"] + 0.004,
            _short_label(row["System"]),
            fontsize=label_fs,
        )

    ax.set_xlabel("FPR", fontsize=11, fontweight="bold")
    ax.set_ylabel("TPR", fontsize=11, fontweight="bold")
    ax.set_title(f"Q3. {world_filter} trade-off", fontsize=11, fontweight="bold")
    ax.grid(True)

    from matplotlib.lines import Line2D

    shape_handles = [
        Line2D([0], [0], marker="o", color="#666", linestyle="None", markersize=8, label="API model"),
        Line2D([0], [0], marker="*", color="#666", linestyle="None", markersize=10, label="Memory agent"),
    ]
    shape_legend = ax.legend(
        handles=shape_handles,
        fontsize=label_fs,
        title="Shape",
        title_fontsize=title_fs,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        alignment="left",
    )
    ax.add_artist(shape_legend)

    color_labels = [
        ("GPT-4o family", API_COLORS["GPT-4o"]),
        ("GPT-5.4-mini family", API_COLORS["GPT-5.4-mini"]),
        ("GPT-5.3 family", API_COLORS["GPT-5.3"]),
    ]
    x0 = 0.98
    y0 = 0.04
    line_gap = 0.05
    ax.text(x0, y0 + 3 * line_gap, "Color", transform=ax.transAxes, fontsize=title_fs, fontweight="bold", ha="right")
    for i, (label, color) in enumerate(color_labels):
        ax.text(
            x0,
            y0 + (2 - i) * line_gap,
            label,
            color=color,
            transform=ax.transAxes,
            fontsize=label_fs,
            ha="right",
        )

    fig.tight_layout()
    return fig


def plot_stage_lines_matplotlib(rows: List[Dict[str, Any]], title: str, left_metric: str = "TPR", right_metric: str = "FPR") -> Any:
    plt, _ = setup_pretty_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    stage_names = ["Early", "Intermediate", "Late"]
    x = list(range(len(stage_names)))

    for label in SYSTEM_ORDER:
        sub = [row for row in rows if row["System"] == label]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: stage_names.index(r["Stage"]))
        color = API_COLORS["GPT-4o"] if "GPT-4o" in label else API_COLORS["GPT-5.4-mini"]
        axes[0].plot(x, [row[left_metric] for row in sub], marker="o", linewidth=2, markersize=5, color=color, label=_short_label(label))
        axes[1].plot(x, [row[right_metric] for row in sub], marker="o", linewidth=2, markersize=5, color=color, label=_short_label(label))

    axes[0].set_title(f"(a) {left_metric} vs stage", fontsize=11, fontweight="bold")
    axes[1].set_title(f"(b) {right_metric} vs stage", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(left_metric, fontsize=11, fontweight="bold")
    axes[1].set_ylabel(right_metric, fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(stage_names)
        ax.set_xlabel("Stage", fontsize=11, fontweight="bold")
        ax.grid(True)

    handles, labels = axes[1].get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    fig.legend([h for h, _ in uniq], [l for _, l in uniq], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="System")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def _system_api_family(label: str) -> str:
    if "GPT-4o" in label:
        return "GPT-4o family"
    if "GPT-5.4-mini" in label:
        return "GPT-5.4-mini family"
    if "GPT-5.3" in label or "ChatGPT" in label:
        return "GPT-5.3 family"
    return "Other"


def plot_stage_mean_bar_matplotlib(
    rows: List[Dict[str, Any]],
    title: str,
    left_metric: str = "TPR",
    right_metric: str = "FPR",
) -> Any:
    plt, np = setup_pretty_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    stage_names = ["Early", "Intermediate", "Late"]
    categories = ["API Models", "Memory Systems"]
    category_colors = {"API Models": "#4C78A8", "Memory Systems": "#E45756"}
    x = np.arange(len(stage_names))
    width = 0.34

    for ax, metric, panel_title in [
        (axes[0], left_metric, f"(a) mean {left_metric} by stage"),
        (axes[1], right_metric, f"(b) mean {right_metric} by stage"),
    ]:
        for idx, category in enumerate(categories):
            vals = []
            for stage in stage_names:
                stage_rows = [
                    row[metric]
                    for row in rows
                    if row["Stage"] == stage and row.get("Group") == category
                ]
                clean = [v for v in stage_rows if v == v]
                vals.append(_safe_mean(clean))
            offset = (-0.5 + idx) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=category_colors[category],
                label=category,
                alpha=0.9,
            )

        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_names)
        ax.set_xlabel("Stage", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        ax.set_ylim(0, 1)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="Mean over systems")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_slot_heatmap_matplotlib(row_labels: List[str], col_labels: List[str], matrix: List[List[float]], title: str) -> Any:
    plt, np = setup_pretty_matplotlib()
    fig, ax = plt.subplots(figsize=(5, 4))
    arr = np.array(matrix, dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if val == val:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_dual_slot_heatmap_matplotlib(
    row_labels_left: List[str],
    col_labels_left: List[str],
    matrix_left: List[List[float]],
    row_labels_right: List[str],
    col_labels_right: List[str],
    matrix_right: List[List[float]],
    title: str,
) -> Any:
    plt, np = setup_pretty_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.8),
        gridspec_kw={"width_ratios": [max(1, len(col_labels_left)), max(1, len(col_labels_right))]},
    )

    arr_left = np.array(matrix_left, dtype=float)
    arr_right = np.array(matrix_right, dtype=float)
    im_left = axes[0].imshow(arr_left, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    im_right = axes[1].imshow(arr_right, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)

    for ax, arr, row_labels, col_labels, subtitle in [
        (axes[0], arr_left, row_labels_left, col_labels_left, "(a) Baseline TPR"),
        (axes[1], arr_right, row_labels_right, col_labels_right, "(b) Controlled-world TNR"),
    ]:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if val == val:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")

    axes[0].set_yticklabels([])
    axes[0].tick_params(axis="y", length=0)
    axes[1].set_yticklabels(row_labels_right)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    cax = fig.add_axes([0.02, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(im_right, cax=cax)
    cbar.set_label("Rate", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0.06, 0.02, 1, 0.95])
    return fig


def plot_metric_pair_heatmaps_shared_y_matplotlib(
    row_labels: List[str],
    left_col_labels: List[str],
    left_matrix: List[List[float]],
    right_col_labels: List[str],
    right_matrix: List[List[float]],
    title: str,
    left_title: str,
    right_title: str,
) -> Any:
    plt, np = setup_pretty_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": [max(1, len(left_col_labels)), max(1, len(right_col_labels))]},
    )

    arr_left = np.array(left_matrix, dtype=float)
    arr_right = np.array(right_matrix, dtype=float)
    im_left = axes[0].imshow(arr_left, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    im_right = axes[1].imshow(arr_right, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)

    panels = [
        (axes[0], arr_left, left_col_labels, left_title),
        (axes[1], arr_right, right_col_labels, right_title),
    ]
    for ax, arr, col_labels, subtitle in panels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if val == val:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")

    axes[0].set_yticklabels(row_labels)
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(im_right, cax=cax)
    cbar.set_label("Rate", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.02, 0.90, 0.95])
    return fig


def plot_ranked_bar_matplotlib(rows: List[Tuple[str, float]], title: str) -> Any:
    plt, _ = setup_pretty_matplotlib()
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [label for label, _ in rows]
    vals = [val for _, val in rows]
    ax.barh(labels, vals, color="#d62728")
    ax.invert_yaxis()
    ax.set_xlabel("TNR under forget", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="x")
    fig.tight_layout()
    return fig
