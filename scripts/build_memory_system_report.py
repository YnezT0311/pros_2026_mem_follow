#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval_results" / "travelPlanning"
OUTPUT_PATH = EVAL_ROOT / "memory_system_report.html"
SUMMARY_PATH = EVAL_ROOT / "instruction_control_summary.json"

SYSTEM_DIRS = {
    "mem0": "gpt-4o+mem0",
    "A-Mem": "gpt-4o+A-Mem",
    "LangMem": "gpt-4o+LangMem",
}

COMMON_ANSWER_PROMPT = """Question: {question}

Find the most appropriate model response and give your final answer (a), (b), (c) after the special token <final_answer>.

[(a) choice A, (b) choice B, (c) choice C]"""

AMEM_KEYWORD_PROMPT = """Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

LANGMEM_ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from one user's conversation history. These memories may contain information that is relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze the provided memories.
2. If the question asks about a specific fact, look for direct evidence in the memories.
3. If the memories contain contradictory information, prioritize the most recent or most specific evidence.
4. If there is not enough evidence to support a confident remembered answer, choose the option that appropriately says the information is not remembered.
5. Focus only on the content of the memories and the question.
6. Answer the multiple-choice question by selecting exactly one option label.

Memories:

{memories}

Question:
{question}

Options:
{options}

Return only the final answer label after the special token <final_answer>, for example <final_answer>(b)</final_answer>."""

MEM0_UPDATE_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

When deciding memory actions, follow these rules strictly:
- Only use existing IDs from the provided current memory block for UPDATE, DELETE, and NONE.
- Never invent or hallucinate an existing ID for UPDATE, DELETE, or NONE.
- If a fact seems related to an old memory but you are not fully sure which existing ID matches it, use ADD instead of UPDATE.
- If you want to delete something but there is no clearly matching existing ID in the provided current memory block, do not delete it.
- For ADD actions, generate a fresh new ID that does not overlap with the provided current-memory IDs.
- Return only valid JSON in the requested schema."""


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9@$.+-]+", "", text)
    return text


def _contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return _norm(needle) in _norm(haystack)


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truncate(text: str, limit: int = 700) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _get_store_texts(data: Dict[str, Any], system: str) -> List[str]:
    if system == "mem0":
        snap = data.get("mem0_debug", {}).get("preload", {}).get("post_add_snapshot", {})
        items: List[Dict[str, Any]] = []
        if isinstance(snap, dict):
            items = snap.get("normalized_items", []) or snap.get("results", [])
            raw = snap.get("raw")
            if not items and isinstance(raw, dict):
                items = raw.get("results", [])
        return [
            str(item.get("memory", "") or item.get("text", "") or item.get("value", ""))
            for item in items
            if isinstance(item, dict)
        ]
    if system == "A-Mem":
        notes = data.get("a_mem_debug", {}).get("preload", {}).get("written_notes", [])
        return [str(note.get("content", "")) for note in notes if isinstance(note, dict)]
    if system == "LangMem":
        snap = data.get("langmem_debug", {}).get("preload", {}).get("store_snapshot", [])
        out = []
        for item in snap:
            if not isinstance(item, dict):
                continue
            value = item.get("value", {})
            out.append(str(value.get("content", "")) if isinstance(value, dict) else str(value))
        return out
    return []


def _get_retrieved_texts(rec: Dict[str, Any], system: str) -> List[str]:
    rm = rec.get("retrieved_memories")
    out: List[str] = []
    if system == "mem0":
        if isinstance(rm, dict):
            for item in rm.get("results", []):
                if isinstance(item, dict):
                    out.append(str(item.get("memory", "")))
    elif system == "A-Mem":
        if isinstance(rm, dict):
            out.append(str(rm.get("raw_context", "")))
    elif system == "LangMem":
        if isinstance(rm, list):
            for item in rm:
                if not isinstance(item, dict):
                    continue
                value = item.get("value", {})
                out.append(str(value.get("content", "")) if isinstance(value, dict) else str(value))
        out.append(str(rec.get("retrieved_memories_text", "")))
    return [text for text in out if text]


def _sample_mem0_fact_prompt() -> Dict[str, str]:
    debug_path = ROOT / "tmp" / "mem0_gpt4o_baseline_persona0_debug.json"
    data = _json_load(debug_path)
    dbg = data["mem0_debug"]["preload"]["fact_extraction_debug"]
    return {
        "system": dbg.get("system_prompt_preview", ""),
        "user": dbg.get("user_prompt_preview", ""),
    }


def _overview() -> Dict[str, Any]:
    summary = _json_load(SUMMARY_PATH)
    out: Dict[str, Any] = {}
    for system in ["mem0", "A-Mem", "LangMem", "plain"]:
        key = f"{system}__gpt-4o"
        world_data = summary[key]["world_difficulty"]
        out[system] = {
            world: {
                "whole_key": world_data[world]["whole_recall_key_turns"],
                "slot_key": world_data[world]["slot_recall_key_turns"],
            }
            for world in ["baseline", "no_store", "forget"]
        }
    return out


def _error_bucket(system: str, world: str, in_store: bool | None, in_retrieved: bool) -> str:
    if world == "baseline":
        if in_store is False:
            return "not_stored_or_overcompressed"
        if in_store is True and not in_retrieved:
            return "stored_but_not_retrieved"
        if in_store is True and in_retrieved:
            return "retrieved_but_answered_wrong"
        return "store_snapshot_missing"

    if in_store is True and in_retrieved:
        return "retained_and_retrieved"
    if in_store is True and not in_retrieved:
        return "retained_not_retrieved_but_answered_remember"
    if in_store is False:
        return "not_retained_exact_value_but_answered_remember"
    return "store_snapshot_missing"


def _build_error_dataset() -> Dict[str, Any]:
    counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    records: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for system, dirname in SYSTEM_DIRS.items():
        for world in ["baseline", "no_store", "forget"]:
            for path in sorted((EVAL_ROOT / world / dirname).glob("*.json")):
                data = _json_load(path)
                store_texts = _get_store_texts(data, system)
                store_blob = "\n".join(store_texts)

                for rec in data.get("slot_recall_results", []):
                    if rec.get("turn_role") != "key":
                        continue
                    predicted = rec.get("predicted_answer_type", "")
                    if world == "baseline":
                        if predicted == "remember_correct":
                            continue
                    else:
                        if predicted != "remember_correct":
                            continue

                    gold_value = str(rec.get("sensitive_value", "")).strip()
                    retrieved_texts = _get_retrieved_texts(rec, system)
                    retrieved_blob = "\n".join(retrieved_texts)
                    in_store = _contains(store_blob, gold_value) if store_texts else None
                    in_retrieved = _contains(retrieved_blob, gold_value)
                    error_type = _error_bucket(system, world, in_store, in_retrieved)

                    counts[system][world][error_type] += 1
                    records[system][world][error_type].append(
                        {
                            "file": str(path.relative_to(ROOT)),
                            "ask_period": data.get("ask_period", ""),
                            "identifier_label": rec.get("identifier_label", ""),
                            "question": rec.get("question", ""),
                            "gold_value": gold_value,
                            "predicted_answer_type": predicted,
                            "sensitive_key": rec.get("sensitive_key", ""),
                            "model_response": rec.get("model_response", ""),
                            "store_excerpt": _truncate("\n\n".join(store_texts[:8]), 1200),
                            "retrieved_excerpt": _truncate("\n\n".join(retrieved_texts[:5]), 1200),
                            "store_snapshot_available": bool(store_texts),
                        }
                    )
    return {
        "counts": {
            system: {world: dict(counter) for world, counter in worlds.items()}
            for system, worlds in counts.items()
        },
        "records": records,
    }


def _pipeline_data() -> Dict[str, Any]:
    mem0_prompts = _sample_mem0_fact_prompt()
    return {
        "mem0": {
            "steps": [
                {
                    "title": "1. Preload Context",
                    "body": "Conversation messages up to ask_period are added into mem0 under one user/run session.",
                    "prompt_label": "Input Shape",
                    "prompt": "User/assistant turns are passed as chat messages into mem0.add(...).",
                },
                {
                    "title": "2. Fact Extraction",
                    "body": "mem0 converts conversation into compact user facts before writing vector memories.",
                    "prompt_label": "System Prompt Preview",
                    "prompt": mem0_prompts["system"],
                    "secondary_prompt_label": "User Prompt Preview",
                    "secondary_prompt": mem0_prompts["user"],
                },
                {
                    "title": "3. Memory Update",
                    "body": "New facts are merged against current memory with add/update/delete decisions.",
                    "prompt_label": "Update Prompt",
                    "prompt": MEM0_UPDATE_PROMPT,
                },
                {
                    "title": "4. Vector Search",
                    "body": "For each MCQ, mem0 runs memory.search(question, limit=k).",
                    "prompt_label": "Retrieved Context Format",
                    "prompt": "Retrieved memories are formatted as numbered memory strings with similarity scores.",
                },
                {
                    "title": "5. Final Answer",
                    "body": "The answer model sees retrieved memories plus the shared MCQ prompt.",
                    "prompt_label": "Answer Prompt Template",
                    "prompt": "Retrieved memories:\\n{memories}\\n\\n" + COMMON_ANSWER_PROMPT,
                },
            ]
        },
        "A-Mem": {
            "steps": [
                {
                    "title": "1. Incremental Note Writing",
                    "body": "A-Mem writes notes for conversation turns and evolves its memory graph incrementally.",
                    "prompt_label": "Visibility Note",
                    "prompt": "The evaluator exposes written notes and raw retrieval context, but the internal official note-writing prompt is not fully surfaced here.",
                },
                {
                    "title": "2. Keyword Generation",
                    "body": "Before retrieval, A-Mem asks an LLM to turn the question into comma-separated keywords.",
                    "prompt_label": "Keyword Prompt",
                    "prompt": AMEM_KEYWORD_PROMPT,
                },
                {
                    "title": "3. Raw Memory Retrieval",
                    "body": "A-Mem retrieves related memories via find_related_memories_raw(keyword_text, k).",
                    "prompt_label": "Retrieved Context Shape",
                    "prompt": "The raw context concatenates memory content, memory context, keywords, and tags.",
                },
                {
                    "title": "4. Final Answer",
                    "body": "The answer model receives the raw retrieved memory block plus the shared MCQ prompt.",
                    "prompt_label": "Answer Prompt Template",
                    "prompt": "Retrieved memories:\\n{raw_context}\\n\\n" + COMMON_ANSWER_PROMPT,
                },
            ]
        },
        "LangMem": {
            "steps": [
                {
                    "title": "1. Manager Writes",
                    "body": "LangMem batches user turns as 'Turn XXX | User: ...' and feeds them into a manage-memory agent.",
                    "prompt_label": "Manager Prompt",
                    "prompt": "You are a helpful assistant.\\n\\n## Memories\\n<memories>\\n{store.search(namespace, query)}\\n</memories>",
                },
                {
                    "title": "2. Store Snapshot",
                    "body": "After preload, the InMemoryStore snapshot is materialized and logged.",
                    "prompt_label": "Snapshot Note",
                    "prompt": "This step records written memory objects and store state; full hidden reasoning is not exposed.",
                },
                {
                    "title": "3. Retrieval Agent / Fallback Search",
                    "body": "At answer time, LangMem invokes the retrieval agent and also keeps top store.search hits.",
                    "prompt_label": "Retrieval Prompt",
                    "prompt": "The latest user question is injected into the retrieval agent; fallback retrieval uses store.search(question, limit=k).",
                },
                {
                    "title": "4. Final Answer",
                    "body": "The answer model is asked to answer only from memory evidence using a dedicated official prompt.",
                    "prompt_label": "Official Answer Prompt",
                    "prompt": LANGMEM_ANSWER_PROMPT,
                },
            ]
        },
    }


def _build_payload() -> Dict[str, Any]:
    return {
        "overview": _overview(),
        "errors": _build_error_dataset(),
        "pipelines": _pipeline_data(),
        "meta": {
            "scope": "gpt-4o only; overview includes baseline / no_store / forget. Error-type view focuses slot key questions because exact target values are alignable to stored/retrieved memory traces.",
            "generated_from": str(EVAL_ROOT.relative_to(ROOT)),
        },
    }


def _render_html(payload: Dict[str, Any]) -> str:
    data_json = html.escape(json.dumps(payload, ensure_ascii=False))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Memory Systems Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 252, 246, 0.92);
      --ink: #172033;
      --muted: #5f6a7d;
      --accent: #b85c38;
      --accent-deep: #7b2f1c;
      --line: rgba(23, 32, 51, 0.12);
      --shadow: 0 18px 45px rgba(34, 28, 20, 0.12);
      --mem0: #b85c38;
      --amem: #1f6f78;
      --langmem: #3f5c2a;
      --plain: #6f5c9a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184,92,56,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(31,111,120,0.13), transparent 24%),
        linear-gradient(180deg, #f9f4ec 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px 20px 64px;
    }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(249,241,230,0.95));
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -10% -45% auto;
      width: 360px;
      height: 360px;
      background: radial-gradient(circle, rgba(184,92,56,0.22), transparent 70%);
      pointer-events: none;
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 700; }}
    h1 {{
      font-size: clamp(2rem, 4vw, 3.6rem);
      letter-spacing: -0.05em;
      max-width: 12ch;
    }}
    .lede {{
      margin-top: 14px;
      max-width: 78ch;
      line-height: 1.6;
      color: var(--muted);
      font-size: 1rem;
    }}
    .chiprow {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 14px;
      background: rgba(255,255,255,0.78);
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .section {{
      margin-top: 28px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .section-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .section-sub {{
      color: var(--muted);
      max-width: 78ch;
      line-height: 1.5;
      font-size: 0.96rem;
    }}
    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      background: rgba(255,255,255,0.78);
    }}
    .sys-tag {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      margin-bottom: 12px;
    }}
    .dot {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
    }}
    .metric-block {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }}
    .metric-title {{
      display: flex;
      justify-content: space-between;
      font-size: 0.9rem;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .meter {{
      width: 100%;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(23,32,51,0.08);
    }}
    .meter > span {{
      display: block;
      height: 100%;
      border-radius: 999px;
    }}
    .triple {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .toolbar label {{
      display: grid;
      gap: 6px;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }}
    .type-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .type-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.9);
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease;
    }}
    .type-card:hover {{ transform: translateY(-2px); }}
    .type-card.active {{
      border-color: var(--accent);
      box-shadow: 0 10px 24px rgba(184,92,56,0.14);
    }}
    .type-name {{
      font-size: 0.9rem;
      font-weight: 700;
      line-height: 1.35;
    }}
    .type-count {{
      margin-top: 8px;
      font-size: 1.65rem;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .records {{
      display: grid;
      gap: 14px;
    }}
    .record {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: rgba(255,255,255,0.92);
    }}
    .record-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .record-title {{
      font-weight: 700;
      line-height: 1.45;
    }}
    .badge {{
      white-space: nowrap;
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(23,32,51,0.08);
      color: var(--muted);
      font-size: 0.8rem;
    }}
    .meta-grid {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .snippet {{
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(23,32,51,0.05);
      border: 1px solid rgba(23,32,51,0.08);
      white-space: pre-wrap;
      line-height: 1.5;
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.84rem;
      max-height: 240px;
      overflow: auto;
    }}
    .pipeline-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .pipeline-card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      background: rgba(255,255,255,0.88);
    }}
    .steps {{
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }}
    .step {{
      position: relative;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,244,237,0.94));
    }}
    .step:hover .prompt-pop {{
      opacity: 1;
      transform: translateY(0);
      pointer-events: auto;
    }}
    .step small {{
      display: inline-block;
      margin-top: 10px;
      color: var(--accent-deep);
      font-weight: 700;
      cursor: help;
    }}
    .prompt-pop {{
      position: absolute;
      z-index: 20;
      left: 10px;
      right: 10px;
      top: calc(100% + 8px);
      padding: 12px;
      border-radius: 14px;
      background: #162033;
      color: #f6f1e8;
      box-shadow: 0 18px 38px rgba(13, 18, 29, 0.24);
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.8rem;
      line-height: 1.45;
      white-space: pre-wrap;
      max-height: 280px;
      overflow: auto;
      opacity: 0;
      transform: translateY(8px);
      pointer-events: none;
      transition: opacity 140ms ease, transform 140ms ease;
    }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }}
    @media (max-width: 920px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 640px) {{
      .page {{ padding: 18px 14px 42px; }}
      .hero, .section {{ padding: 18px; border-radius: 18px; }}
      .toolbar {{ grid-template-columns: 1fr; }}
      .triple {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Memory Systems Report</h1>
      <div class="lede">
        This report summarizes corrected <code>gpt-4o</code> results for <code>mem0</code>, <code>A-Mem</code>, and <code>LangMem</code>.
        The overview shows high-level performance. The error section focuses on slot-key failures and violations because those can be aligned against stored and retrieved memory traces.
      </div>
      <div class="chiprow">
        <div class="chip">Scope: baseline / no_store / forget</div>
        <div class="chip">Memory traces: store snapshot + retrieved memories</div>
        <div class="chip">Source: {html.escape(payload["meta"]["generated_from"])}</div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Overview</h2>
          <div class="section-sub">Whole-recall and slot-recall key-turn performance after reparsing and file-level summary recomputation.</div>
        </div>
      </div>
      <div id="overview" class="overview-grid"></div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Error Type Analysis</h2>
          <div class="section-sub">Filter by memory system, world, and error type. The list below shows every matched error record in the current dataset.</div>
        </div>
      </div>
      <div class="toolbar">
        <label>System<select id="system-select"></select></label>
        <label>World<select id="world-select"></select></label>
        <label>Error Type<select id="type-select"></select></label>
        <label>Sort<select id="sort-select">
          <option value="identifier">Identifier Label</option>
          <option value="period">Ask Period</option>
          <option value="file">File</option>
        </select></label>
      </div>
      <div id="type-summary" class="type-summary"></div>
      <div id="records" class="records"></div>
      <div id="records-note" class="footer-note"></div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Pipeline Visualization</h2>
          <div class="section-sub">Hover the prompt labels to inspect prompt text. Prompts are hidden by default to keep the pipeline readable.</div>
        </div>
      </div>
      <div id="pipelines" class="pipeline-grid"></div>
      <div class="footer-note">
        Note: for A-Mem, the evaluator exposes written notes and retrieval prompts, but the internal official note-writing prompt is not fully surfaced in the current log format.
      </div>
    </section>
  </div>

  <script id="report-data" type="application/json">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('report-data').textContent);
    const COLORS = {{
      mem0: 'var(--mem0)',
      'A-Mem': 'var(--amem)',
      LangMem: 'var(--langmem)',
      plain: 'var(--plain)'
    }};

    function el(tag, cls, text) {{
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text !== undefined) node.textContent = text;
      return node;
    }}

    function pct(v) {{
      return `${{(v * 100).toFixed(1)}}%`;
    }}

    function renderOverview() {{
      const root = document.getElementById('overview');
      root.innerHTML = '';
      ['mem0', 'A-Mem', 'LangMem', 'plain'].forEach((system) => {{
        const card = el('div', 'card');
        const tag = el('div', 'sys-tag');
        const dot = el('span', 'dot');
        dot.style.background = COLORS[system];
        tag.append(dot, document.createTextNode(system));
        card.appendChild(tag);

        ['baseline', 'no_store', 'forget'].forEach((world) => {{
          const block = el('div', 'metric-block');
          const title = el('div', 'metric-title');
          title.append(el('span', null, world), el('span', null, `whole / slot`));
          block.appendChild(title);

          const whole = DATA.overview[system][world].whole_key;
          const slot = DATA.overview[system][world].slot_key;

          const wholeMeter = el('div', 'meter');
          const wholeBar = el('span');
          wholeBar.style.width = `${{whole.remember_correct_rate * 100}}%`;
          wholeBar.style.background = COLORS[system];
          wholeMeter.appendChild(wholeBar);

          const slotMeter = el('div', 'meter');
          const slotBar = el('span');
          slotBar.style.width = `${{slot.remember_correct_rate * 100}}%`;
          slotBar.style.background = COLORS[system];
          slotMeter.appendChild(slotBar);

          block.appendChild(wholeMeter);
          block.appendChild(el('div', 'triple', ''));
          block.lastChild.innerHTML = `
            <div>Whole remember: <strong>${{pct(whole.remember_correct_rate)}}</strong></div>
            <div>Whole not_rem: <strong>${{pct(whole.not_remember_rate)}}</strong></div>
            <div>Whole distractor: <strong>${{pct(whole.distractor_irrelevant_rate)}}</strong></div>
          `;
          block.appendChild(slotMeter);
          block.appendChild(el('div', 'triple', ''));
          block.lastChild.innerHTML = `
            <div>Slot remember: <strong>${{pct(slot.remember_correct_rate)}}</strong></div>
            <div>Slot not_rem: <strong>${{pct(slot.not_remember_rate)}}</strong></div>
            <div>Slot distractor: <strong>${{pct(slot.distractor_irrelevant_rate)}}</strong></div>
          `;
          card.appendChild(block);
        }});

        root.appendChild(card);
      }});
    }}

    const systemSelect = document.getElementById('system-select');
    const worldSelect = document.getElementById('world-select');
    const typeSelect = document.getElementById('type-select');
    const sortSelect = document.getElementById('sort-select');

    function fillSelect(select, values) {{
      select.innerHTML = '';
      values.forEach((value) => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }});
    }}

    function updateTypeOptions(preferred) {{
      const system = systemSelect.value;
      const world = worldSelect.value;
      const counts = DATA.errors.counts[system][world] || {{}};
      const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
      fillSelect(typeSelect, ['all', ...keys]);
      if (preferred && ['all', ...keys].includes(preferred)) {{
        typeSelect.value = preferred;
      }}
    }}

    function sortedRecords(records) {{
      const mode = sortSelect.value;
      return [...records].sort((a, b) => {{
        if (mode === 'period') return a.ask_period.localeCompare(b.ask_period);
        if (mode === 'file') return a.file.localeCompare(b.file);
        return a.identifier_label.localeCompare(b.identifier_label);
      }});
    }}

    function renderTypeSummary() {{
      const wrap = document.getElementById('type-summary');
      wrap.innerHTML = '';
      const system = systemSelect.value;
      const world = worldSelect.value;
      const counts = DATA.errors.counts[system][world] || {{}};
      const active = typeSelect.value;
      Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .forEach(([name, count]) => {{
          const card = el('div', 'type-card' + (active === name ? ' active' : ''));
          card.onclick = () => {{
            typeSelect.value = name;
            renderErrors();
          }};
          card.appendChild(el('div', 'type-name', name));
          card.appendChild(el('div', 'type-count', String(count)));
          wrap.appendChild(card);
        }});
    }}

    function renderErrors() {{
      renderTypeSummary();
      const root = document.getElementById('records');
      const note = document.getElementById('records-note');
      root.innerHTML = '';
      const system = systemSelect.value;
      const world = worldSelect.value;
      const type = typeSelect.value;
      const all = DATA.errors.records[system][world] || {{}};
      let records = [];
      if (type === 'all') {{
        records = Object.values(all).flat();
      }} else {{
        records = all[type] || [];
      }}
      records = sortedRecords(records);

      note.textContent = `${{records.length}} error records shown. Error view is slot-key only; exact-value matching is conservative for paraphrased dates, ranges, and normalized values.`;

      records.forEach((record) => {{
        const card = el('article', 'record');
        const head = el('div', 'record-head');
        const title = el('div', 'record-title', record.question);
        const badge = el('div', 'badge', record.predicted_answer_type);
        head.append(title, badge);
        card.appendChild(head);

        const meta = el('div', 'meta-grid');
        meta.innerHTML = `
          <div><strong>Gold Value:</strong> ${{record.gold_value || '(empty)'}}<\/div>
          <div><strong>Label:</strong> ${{record.identifier_label || '(none)'}}<\/div>
          <div><strong>Ask Period:</strong> ${{record.ask_period || '(none)'}}<\/div>
          <div><strong>Field:</strong> ${{record.sensitive_key || '(none)'}}<\/div>
          <div><strong>Store Snapshot:</strong> ${{record.store_snapshot_available ? 'available' : 'missing'}}<\/div>
          <div><strong>File:</strong> ${{record.file}}<\/div>
        `;
        card.appendChild(meta);

        card.appendChild(el('div', 'snippet', `Model response\\n${{record.model_response || '(empty)'}}`));
        card.appendChild(el('div', 'snippet', `Retrieved memory excerpt\\n${{record.retrieved_excerpt || '(empty)'}}`));
        card.appendChild(el('div', 'snippet', `Stored memory excerpt\\n${{record.store_excerpt || '(empty)'}}`));
        root.appendChild(card);
      }});
    }}

    function renderPipelines() {{
      const root = document.getElementById('pipelines');
      root.innerHTML = '';
      Object.entries(DATA.pipelines).forEach(([system, cfg]) => {{
        const card = el('div', 'pipeline-card');
        const tag = el('div', 'sys-tag');
        const dot = el('span', 'dot');
        dot.style.background = COLORS[system];
        tag.append(dot, document.createTextNode(system));
        card.appendChild(tag);

        const steps = el('div', 'steps');
        cfg.steps.forEach((step) => {{
          const stepNode = el('div', 'step');
          stepNode.appendChild(el('h3', null, step.title));
          const body = el('div', 'section-sub', step.body);
          body.style.marginTop = '8px';
          stepNode.appendChild(body);
          const promptLabel = el('small', null, `Hover to inspect: ${{step.prompt_label}}`);
          stepNode.appendChild(promptLabel);
          const pop = el('div', 'prompt-pop', step.prompt || '');
          stepNode.appendChild(pop);
          if (step.secondary_prompt) {{
            const promptLabel2 = el('small', null, `Hover to inspect: ${{step.secondary_prompt_label}}`);
            stepNode.appendChild(promptLabel2);
            const pop2 = el('div', 'prompt-pop', step.secondary_prompt);
            stepNode.appendChild(pop2);
          }}
          steps.appendChild(stepNode);
        }});
        card.appendChild(steps);
        root.appendChild(card);
      }});
    }}

    function initFilters() {{
      fillSelect(systemSelect, Object.keys(DATA.errors.counts));
      fillSelect(worldSelect, ['baseline', 'no_store', 'forget']);
      systemSelect.value = 'mem0';
      worldSelect.value = 'baseline';
      updateTypeOptions('all');
      [systemSelect, worldSelect].forEach((select) => {{
        select.addEventListener('change', () => {{
          updateTypeOptions('all');
          renderErrors();
        }});
      }});
      typeSelect.addEventListener('change', renderErrors);
      sortSelect.addEventListener('change', renderErrors);
    }}

    renderOverview();
    initFilters();
    renderErrors();
    renderPipelines();
  </script>
</body>
</html>"""


def main() -> None:
    payload = _build_payload()
    OUTPUT_PATH.write_text(_render_html(payload), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
