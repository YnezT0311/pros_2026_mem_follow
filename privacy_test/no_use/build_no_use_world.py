import argparse
import copy
import json
import os
import random
import re
import shutil
from typing import Dict, List, Optional, Tuple
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

PERIODS = [
    "Conversation Initial Stage",
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]

PERIOD_ALIASES = {
    "Conversation Initial Stage": ["Conversation Initial Stage", "Init Conversation"],
    "Conversation Early Stage": ["Conversation Early Stage", "Conversation Next Week"],
    "Conversation Intermediate Stage": ["Conversation Intermediate Stage", "Conversation Next Month"],
    "Conversation Late Stage": ["Conversation Late Stage", "Conversation Next Year"],
}

PERIOD_SHORT = {
    "Conversation Initial Stage": "initial",
    "Conversation Early Stage": "early",
    "Conversation Intermediate Stage": "intermediate",
    "Conversation Late Stage": "late",
}

CONTEXTUAL_HISTORY_ALIASES = {
    "Contextual Personal History Initial Stage": ["Contextual Personal History Initial Stage", "Init Contextual Personal History"],
    "Contextual Personal History Early Stage": ["Contextual Personal History Early Stage", "Contextual Personal History Next Week"],
    "Contextual Personal History Intermediate Stage": ["Contextual Personal History Intermediate Stage", "Contextual Personal History Next Month"],
    "Contextual Personal History Late Stage": ["Contextual Personal History Late Stage", "Contextual Personal History Next Year"],
}


def get_first_present(data: Dict, keys: List[str], default=None):
    for key in keys:
        if key in data:
            return data[key]
    return default


def get_period_lines(data: Dict, period: str):
    return get_first_present(data, PERIOD_ALIASES[period], [])

NO_USE_ON_TEMPLATES = [
    "For now, do not use what I told you before.",
    "For now, please answer without using my earlier details.",
    "Temporarily, do not rely on my past messages.",
    "For this phase, avoid using prior conversation memory.",
]

NO_USE_OFF_TEMPLATES = [
    "You can use everything now.",
    "Now you may use my earlier details again.",
    "The temporary restriction is over. You can use prior context.",
    "You can rely on previous conversation details from now on.",
]

ACK_ON_TEMPLATES = [
    "Understood. For now, I will answer without using earlier details.",
    "Got it. I will avoid relying on prior messages for now.",
    "Okay. I will respond without using previous details for now.",
    "Understood. For this phase, I will not rely on earlier context.",
]

ACK_OFF_TEMPLATES = [
    "Understood. I can use earlier details again now.",
    "Got it. I will use prior context again when relevant.",
    "Okay. I can use your earlier details again from now on.",
    "Understood. The temporary limit is lifted and I can use prior context.",
]

LOW_SIGNAL_PREFIXES = (
    "hi",
    "hello",
    "i'd like",
    "i would like",
    "i want to",
    "i'm going to",
    "i am going to",
    "i'll walk you through",
    "i will walk you through",
)

LOW_SIGNAL_PHRASES = (
    "go through my",
    "go over my",
    "review my",
    "walk you through",
    "help me plan",
    "get practical suggestions",
    "one by one",
    "next steps",
    "practical help prioritizing",
)

STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "and",
    "or", "but", "to", "of", "in", "on", "at", "for", "with", "through",
    "that", "this", "is", "are", "was", "were", "be", "been", "it", "as",
    "from", "by", "about", "into", "do", "not", "please",
}

GENERIC_CONFLICT_WORDS = {
    "clinic", "session", "client", "clients", "legal", "data", "checklist",
    "consent", "workshop", "audit", "project", "history", "family", "private",
    "report", "advisor", "program", "community", "support", "planning", "change",
    "changed", "issue", "issues", "participant", "participants", "template",
    "review", "research", "agreement", "pilot", "records", "details", "detail",
}

BROAD_CONFLICT_WORDS = {
    "short", "long", "small", "large", "using", "use", "used", "different",
    "would", "could", "should", "have", "has", "had", "how", "see", "ran",
    "run", "script", "return", "returns",
}


def iter_conversation_files(source_dir: str) -> List[str]:
    files = []
    for root, _, fnames in os.walk(source_dir):
        for fname in fnames:
            if fname.startswith("conversation_") and fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def missing_conversation_periods(data: Dict) -> List[str]:
    missing = []
    for period in PERIODS:
        value = get_period_lines(data, period)
        if not isinstance(value, list) or not value:
            missing.append(period)
    return missing


def extract_topic_and_ids(file_name: str) -> Tuple[str, str, str]:
    m = re.match(r"conversation_(.+)_persona(\d+)_sample(\d+)\.json", file_name)
    if not m:
        return "unknown", "unknown", "unknown"
    return m.group(1), m.group(2), m.group(3)


def split_user_prefix(line: str) -> str:
    if isinstance(line, str) and line.startswith("User:"):
        return line[len("User:"):].strip()
    return str(line).strip()


def is_low_signal_utility_text(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not t:
        return True
    if any(t.startswith(prefix) for prefix in LOW_SIGNAL_PREFIXES):
        return True
    if any(phrase in t for phrase in LOW_SIGNAL_PHRASES):
        return True
    if t.endswith("?"):
        return True
    return False


def weighted_fact_terms(text: str) -> Dict[str, float]:
    terms: Dict[str, float] = {}
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text or "")
    for raw in raw_tokens:
        token = raw.lower().strip(".,!?;:'\"()[]{}")
        if (
            not token
            or token in STOPWORDS
            or token in GENERIC_CONFLICT_WORDS
        ):
            continue
        if token.isdigit() or re.search(r"\d", token):
            weight = 3.0
        elif len(token) >= 11:
            weight = 3.0
        elif len(token) >= 8:
            weight = 2.0
        elif len(token) >= 5:
            weight = 1.5
        else:
            weight = 1.0
        if token in BROAD_CONFLICT_WORDS:
            weight = min(weight, 0.25)
        if "-" in token:
            weight += 0.5
        terms[token] = max(weight, terms.get(token, 0.0))
    return terms


def is_conflict_match(text: str, fact: str, threshold: float = 4.5) -> Tuple[bool, float, List[str]]:
    fact_terms = weighted_fact_terms(fact)
    if not fact_terms:
        return False, 0.0, []
    text_tokens = set(
        t.lower().strip(".,!?;:'\"()[]{}")
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text or "")
    )
    hits = sorted([t for t in fact_terms if t in text_tokens], key=lambda x: (-fact_terms[x], x))
    score = sum(fact_terms[h] for h in hits)
    exact_fact = (fact or "").strip().lower()
    phrase_match = bool(exact_fact and len(exact_fact) >= 24 and exact_fact in (text or "").strip().lower())
    strong_hits = [h for h in hits if fact_terms.get(h, 0.0) >= 2.0]
    matched = phrase_match or score >= threshold or (score >= 3.0 and len(strong_hits) >= 2)
    return matched, score, hits


def detect_scope_conflicts(data: Dict, rows: List[Dict]) -> List[Dict]:
    conflicts: List[Dict] = []
    for row in rows:
        fact = row["reveal_fact_text"]
        on_period = row["no_use_on_period"]
        on_ack = row["no_use_on_ack_index"]
        off_period = row.get("no_use_off_period")
        off_turn = row.get("no_use_off_turn_index")
        active_start = PERIODS.index(on_period)
        active_end = PERIODS.index(off_period) if off_period in PERIODS else len(PERIODS) - 1
        for p_idx in range(active_start, active_end + 1):
            period = PERIODS[p_idx]
            lines = get_period_lines(data, period)
            if not isinstance(lines, list):
                continue
            if period == on_period:
                start_i = (on_ack if isinstance(on_ack, int) else row["no_use_on_turn_index"]) + 1
            else:
                start_i = 0
            end_i = len(lines)
            if off_period in PERIODS and period == off_period and isinstance(off_turn, int):
                end_i = off_turn
            for i in range(start_i, min(end_i, len(lines))):
                line = lines[i]
                if not isinstance(line, str) or not line.startswith("Assistant:"):
                    continue
                matched, score, hits = is_conflict_match(line, fact, threshold=4.5)
                if matched:
                    conflicts.append({
                        "sample_id": row["sample_id"],
                        "period": period,
                        "line_index": i,
                        "reveal_fact": fact,
                        "hit_keywords": hits,
                        "match_score": score,
                        "line_text_before": line,
                    })
    return conflicts


def local_repair_scope_conflicts(data: Dict, conflicts: List[Dict]) -> List[Dict]:
    logs: List[Dict] = []
    seen = set()
    for cf in conflicts:
        key = (cf["period"], cf["line_index"])
        if key in seen:
            continue
        seen.add(key)
        period = cf["period"]
        idx = cf["line_index"]
        line = get_period_lines(data, period)[idx]
        body = line[len("Assistant:"):].strip() if line.startswith("Assistant:") else str(line)
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
        kept = []
        for part in parts:
            matched, _, _ = is_conflict_match(part, cf["reveal_fact"], threshold=4.5)
            if not matched:
                kept.append(part)
        if kept:
            new_line = "Assistant: " + " ".join(kept).strip()
            if new_line[-1] not in ".!?":
                new_line += "."
        else:
            new_line = "Assistant: I can still help with the current question without relying on earlier restricted details."
        if new_line != line:
            get_period_lines(data, period)[idx] = new_line
            logs.append({
                "sample_id": cf["sample_id"],
                "period": period,
                "line_index": idx,
                "operation": "assistant_scope_repair",
                "before": line,
                "after": new_line,
            })
    return logs


def format_history_block(data: Dict, period: str) -> str:
    parts = []
    parts.append(f"Expanded Persona:\n{data.get('Expanded Persona', '')}")
    history_key = {
        "Conversation Initial Stage": "Contextual Personal History Initial Stage",
        "Conversation Early Stage": "Contextual Personal History Early Stage",
        "Conversation Intermediate Stage": "Contextual Personal History Intermediate Stage",
        "Conversation Late Stage": "Contextual Personal History Late Stage",
    }[period]
    value = get_first_present(data, CONTEXTUAL_HISTORY_ALIASES[history_key], {})
    if value:
        parts.append(f"{history_key}:\n{json.dumps(value, ensure_ascii=False, indent=2)}")
    return "\n\n".join(parts).strip()


def build_allowed_scope_context(data: Dict, row: Dict, target_period: str, target_line_index: int) -> str:
    on_period = row["no_use_on_period"]
    on_ack = row["no_use_on_ack_index"]
    messages = []
    for period in PERIODS[PERIODS.index(on_period): PERIODS.index(target_period) + 1]:
        lines = get_period_lines(data, period)
        if not isinstance(lines, list):
            continue
        start_i = 0
        if period == on_period:
            start_i = (on_ack if isinstance(on_ack, int) else row["no_use_on_turn_index"]) + 1
        end_i = len(lines)
        if period == target_period:
            end_i = min(end_i, target_line_index + 1)
        for i in range(start_i, end_i):
            line = lines[i]
            if isinstance(line, str) and not line.startswith("Side_Note:"):
                messages.append(line)
    return "\n".join(messages).strip()


def rewrite_scope_assistant_with_model(
    data: Dict,
    row: Dict,
    period: str,
    line_index: int,
    client: Optional[OpenAI],
    model: str,
    max_retries: int,
    timeout: float,
) -> Optional[str]:
    if client is None:
        return None
    history_block = format_history_block(data, period)
    allowed_context = build_allowed_scope_context(data, row, period, line_index)
    period_lines = get_period_lines(data, period)
    original_reply = period_lines[line_index] if isinstance(period_lines, list) and 0 <= line_index < len(period_lines) else ""
    sys_msg = {
        "role": "system",
        "content": (
            "Rewrite one assistant reply in an ongoing synthetic conversation benchmark. "
            "Use only the allowed in-scope context shown below. "
            "Respond directly to the latest user point, stay natural and specific, and do not introduce new events. "
            "Do not use or mention earlier restricted memory from before the no-use instruction."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Current period: {period}\n\n"
            f"Current period history:\n{history_block}\n\n"
            f"Allowed dialogue context after the no-use instruction:\n{allowed_context}\n\n"
            f"Original assistant reply:\n{original_reply}\n\n"
            f"Restricted earlier memory:\n{row['reveal_fact_text']}\n\n"
            "Return only the rewritten assistant reply text. Do not add 'Assistant:'. Do not output explanation."
        ),
    }
    try:
        text = query_openai_chat(client, model, [sys_msg, user_msg], max_retries=max_retries, timeout=timeout)
    except Exception:
        return None
    text = re.sub(r"^\s*Assistant:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if text[-1] not in ".!?":
        text += "."
    return "Assistant: " + text


def apply_model_scope_repairs(
    data: Dict,
    file_rows: List[Dict],
    conflicts: List[Dict],
    client: Optional[OpenAI],
    model: str,
    max_retries: int,
    timeout: float,
) -> List[Dict]:
    if client is None:
        return []
    repair_logs = []
    seen = set()
    for cf in conflicts:
        key = (cf["sample_id"], cf["period"], cf["line_index"])
        if key in seen:
            continue
        seen.add(key)
        row = next((x for x in file_rows if x["sample_id"] == cf["sample_id"]), None)
        if row is None:
            continue
        rewritten = rewrite_scope_assistant_with_model(
            data=data,
            row=row,
            period=cf["period"],
            line_index=cf["line_index"],
            client=client,
            model=model,
            max_retries=max_retries,
            timeout=timeout,
        )
        if rewritten and rewritten != data[cf["period"]][cf["line_index"]]:
            repair_logs.append({
                "sample_id": cf["sample_id"],
                "period": cf["period"],
                "line_index": cf["line_index"],
                "operation": "assistant_scope_rewrite_model",
                "before": data[cf["period"]][cf["line_index"]],
                "after": rewritten,
            })
            data[cf["period"]][cf["line_index"]] = rewritten
    return repair_logs


def pick_user_turn(lines: List[str], min_words: int = 8, avoid_idx: Optional[int] = None) -> Optional[int]:
    cand = []
    for i, line in enumerate(lines):
        if avoid_idx is not None and i == avoid_idx:
            continue
        if not isinstance(line, str) or not line.startswith("User:"):
            continue
        content = split_user_prefix(line)
        low = content.lower()
        if any(k in low for k in ["do not use", "for now", "you can use everything now", "restriction is over"]):
            continue
        if "prior conversation memory" in low or "previous conversation details" in low:
            continue
        if len(content.split()) < min_words:
            continue
        score = 0
        if i > 0 and isinstance(lines[i - 1], str) and lines[i - 1].startswith("Side_Note"):
            score += 2
        if len(content.split()) >= 16:
            score += 1
        cand.append((score, i))
    if not cand:
        return None
    cand.sort(key=lambda x: (-x[0], x[1]))
    return cand[0][1]


def extract_reveal_fact(user_content: str, max_words: int = 32) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", user_content) if s.strip()]
    fact = sents[0] if sents else user_content.strip()
    words = fact.split()
    if len(words) > max_words:
        fact = " ".join(words[:max_words]).rstrip(",;:")
    return fact.rstrip(".!?")


def extract_utility_facts(data: Dict, blocked_user_texts: set, max_items: int = 8) -> List[str]:
    facts: List[str] = []
    seen = set()
    for p in PERIODS:
        lines = get_period_lines(data, p)
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, str) or not line.startswith("User:"):
                continue
            content = split_user_prefix(line)
            if not content or content in blocked_user_texts:
                continue
            if is_low_signal_utility_text(content):
                continue
            if len(content.split()) < 8:
                continue
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
            cands = [s for s in sents if len(s.split()) >= 8 and not is_low_signal_utility_text(s)]
            if not cands and not is_low_signal_utility_text(content):
                cands = [content]
            for cand in cands:
                k = cand.lower().strip()
                if k in seen:
                    continue
                seen.add(k)
                facts.append(cand.rstrip(".!?"))
                if len(facts) >= max_items:
                    return facts
    return facts


def pick_template(pool: List[str]) -> Tuple[int, str]:
    idx = random.randrange(len(pool))
    return idx, pool[idx]


def inject_instruction(lines: List[str], idx: int, user_text: str, assistant_text: str) -> Tuple[int, int]:
    lines[idx:idx] = [f"User: {user_text}", f"Assistant: {assistant_text}"]
    return idx, idx + 1


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_existing_summary(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def no_use_world_complete(
    tgt_path: str,
    existing_meta_by_world: Dict[str, List[Dict]],
    existing_ops_sample_ids: set,
    required_modes: List[str],
) -> bool:
    if not os.path.exists(tgt_path):
        return False
    rows = existing_meta_by_world.get(tgt_path, [])
    if not rows:
        return False
    present_modes = {r.get("mode") for r in rows}
    if not set(required_modes).issubset(present_modes):
        return False
    relevant_rows = [r for r in rows if r.get("mode") in required_modes]
    return bool(relevant_rows) and all(r.get("sample_id") in existing_ops_sample_ids for r in relevant_rows)


def count_complete_processed_worlds(source_files: List[str], source_dir: str, target_dir: str) -> int:
    count = 0
    for src_path in source_files:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if missing_conversation_periods(data):
            continue
        rel_path = os.path.relpath(src_path, source_dir)
        if os.path.exists(os.path.join(target_dir, rel_path)):
            count += 1
    return count


def summarize_no_use_outputs(
    source_files: List[str],
    source_dir: str,
    target_dir: str,
    meta: List[Dict],
    ops: List[Dict],
    mode: str,
    ack_source: str,
    ack_on_pool: List[str],
    ack_off_pool: List[str],
    cache_path: str,
    this_run: Dict,
) -> Dict:
    repair_ops = [op for op in ops if op.get("operation") not in {"inject_no_use_on", "inject_no_use_off"}]
    mode_counts = {"scope": 0, "temporal_scope": 0}
    for row in meta:
        row_mode = row.get("mode")
        if row_mode in mode_counts:
            mode_counts[row_mode] += 1
    summary = {
        "source_dir": source_dir,
        "target_dir": target_dir,
        "mode": mode,
        "completion_template_source": ack_source,
        "completion_on_template_count": len(ack_on_pool),
        "completion_off_template_count": len(ack_off_pool),
        "completion_template_cache_path": cache_path,
        "num_files_total": len(source_files),
        "num_files_processed": count_complete_processed_worlds(source_files, source_dir, target_dir),
        "num_files_with_no_use": len({row.get("world_file_no_use") for row in meta if row.get("world_file_no_use")}),
        "num_rows": len(meta),
        "num_on": len(meta),
        "num_off": sum(1 for row in meta if row.get("no_use_off_turn_index") is not None),
        "num_conflicts_before_repair": sum(int(row.get("conflicts_before_repair", 0) or 0) for row in meta),
        "num_repairs": len(repair_ops),
        "num_conflicts_after_repair": sum(int(row.get("conflicts_after_repair", 0) or 0) for row in meta),
        "mode_counts": mode_counts,
    }
    summary.update(this_run)
    return summary


def resolve_api_key(token_path: str, api_key_file: str) -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    p = os.path.join(token_path, api_key_file)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    if os.path.exists(api_key_file):
        with open(api_key_file, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    return ""


def query_openai_chat(client: OpenAI, model: str, messages: List[Dict[str, str]], max_retries: int = 3, timeout: float = 60.0) -> str:
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, timeout=timeout)
            txt = (resp.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception as e:
            last_err = e
    raise RuntimeError(f"OpenAI generation failed: {last_err}")


def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:json|text|plaintext)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _parse_template_candidates(raw: str, max_items: int) -> List[str]:
    txt = _strip_code_fence(raw)
    candidates: List[str] = []
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            for x in obj:
                if isinstance(x, str):
                    candidates.append(x.strip())
    except Exception:
        for line in txt.splitlines():
            line = re.sub(r"^\s*[-*]\s*", "", line.strip())
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line.strip())
            if line:
                candidates.append(line)

    out: List[str] = []
    seen = set()
    for c in candidates:
        c = re.sub(r"^\s*Assistant:\s*", "", c, flags=re.IGNORECASE).strip()
        c = re.sub(r"\s+", " ", c)
        if not c:
            continue
        if c[-1] not in ".!?":
            c += "."
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= max_items:
            break
    return out


def build_completion_template_pool(
    client: OpenAI,
    model: str,
    template_count: int,
    template_type: str,
    max_retries: int,
    timeout: float,
) -> List[str]:
    template_count = max(1, int(template_count))
    if template_type == "on":
        intent = (
            "Acknowledge that, for now, the assistant should not use earlier conversation details. "
            "Keep it short and natural."
        )
    else:
        intent = (
            "Acknowledge that the temporary no-use restriction is lifted, and prior details can be used again. "
            "Keep it short and natural."
        )
    sys_msg = {
        "role": "system",
        "content": "Generate concise assistant acknowledgements in chat style.",
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Return exactly {template_count} one-sentence replies as a JSON array of strings. "
            f"{intent} No numbering or markdown."
        ),
    }
    raw = query_openai_chat(client, model, [sys_msg, user_msg], max_retries=max_retries, timeout=timeout)
    return _parse_template_candidates(raw, max_items=template_count)


def load_completion_template_cache(path: str) -> Optional[Tuple[List[str], List[str]]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        on_pool = obj.get("ack_on_templates", [])
        off_pool = obj.get("ack_off_templates", [])
        if not isinstance(on_pool, list) or not isinstance(off_pool, list):
            return None
        on_pool = [str(x).strip() for x in on_pool if str(x).strip()]
        off_pool = [str(x).strip() for x in off_pool if str(x).strip()]
        if not on_pool or not off_pool:
            return None
        return on_pool, off_pool
    except Exception:
        return None


def save_completion_template_cache(path: str, on_pool: List[str], off_pool: List[str], model: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model,
        "ack_on_templates": on_pool,
        "ack_off_templates": off_pool,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_no_use_world(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    src_files = iter_conversation_files(args.source_dir)
    if not src_files:
        raise FileNotFoundError(f"No conversation files found under {args.source_dir}")

    if args.rebuild_target and os.path.exists(args.target_dir):
        shutil.rmtree(args.target_dir)
    os.makedirs(args.target_dir, exist_ok=True)

    modes = [args.mode] if args.mode in {"scope", "temporal_scope"} else ["scope", "temporal_scope"]

    api_key = resolve_api_key(args.token_path, args.api_key_file)
    client = None
    if api_key:
        kwargs = {"api_key": api_key}
        if args.api_base_url:
            kwargs["base_url"] = args.api_base_url
        if OpenAI is None:
            raise RuntimeError("openai package is required for no-use completion/repair but is not installed.")
        client = OpenAI(**kwargs)

    ack_on_pool = list(ACK_ON_TEMPLATES)
    ack_off_pool = list(ACK_OFF_TEMPLATES)
    ack_source = "builtin"
    cache_path = args.completion_template_cache_path
    if args.completion_use_templates:
        cached = None
        if not args.completion_templates_from_model:
            cached = load_completion_template_cache(cache_path)
        if cached is not None:
            ack_on_pool, ack_off_pool = cached
            ack_source = "cache"
        else:
            if client is None:
                ack_source = "builtin_fallback"
                save_completion_template_cache(cache_path, ack_on_pool, ack_off_pool, args.model)
            else:
                try:
                    gen_on = build_completion_template_pool(
                        client=client,
                        model=args.model,
                        template_count=args.completion_template_count,
                        template_type="on",
                        max_retries=args.max_retries,
                        timeout=args.request_timeout,
                    )
                    gen_off = build_completion_template_pool(
                        client=client,
                        model=args.model,
                        template_count=args.completion_template_count,
                        template_type="off",
                        max_retries=args.max_retries,
                        timeout=args.request_timeout,
                    )
                    if gen_on:
                        ack_on_pool = gen_on
                    if gen_off:
                        ack_off_pool = gen_off
                    save_completion_template_cache(cache_path, ack_on_pool, ack_off_pool, args.model)
                    ack_source = "model"
                except Exception:
                    ack_source = "builtin_fallback"
                    save_completion_template_cache(cache_path, ack_on_pool, ack_off_pool, args.model)

    print(
        f"[start] files={len(src_files)} mode={args.mode} "
        f"completion_use_templates={args.completion_use_templates} "
        f"completion_templates_from_model={args.completion_templates_from_model} "
        f"source={ack_source} "
        f"on_templates={len(ack_on_pool)} off_templates={len(ack_off_pool)} "
        f"cache_path={cache_path}",
        flush=True,
    )

    existing_meta = [] if args.rebuild_target else read_jsonl(args.meta_path)
    existing_ops = [] if args.rebuild_target else read_jsonl(args.ops_path)
    existing_meta_by_world: Dict[str, List[Dict]] = {}
    existing_ops_sample_ids = set()
    for row in existing_meta:
        existing_meta_by_world.setdefault(row.get("world_file_no_use", ""), []).append(row)
    for op in existing_ops:
        if "sample_id" in op:
            existing_ops_sample_ids.add(op["sample_id"])

    new_meta: List[Dict] = []
    new_ops: List[Dict] = []
    rebuilt_worlds = set()
    summary = {
        "num_on": 0,
        "num_off": 0,
        "this_run_num_files_processed": 0,
        "this_run_num_files_with_no_use": 0,
        "this_run_num_rows": 0,
        "this_run_num_on": 0,
        "this_run_num_off": 0,
        "this_run_num_conflicts_before_repair": 0,
        "this_run_num_repairs": 0,
        "this_run_num_conflicts_after_repair": 0,
    }

    for src_path in src_files:
        rel = os.path.relpath(src_path, args.source_dir)
        tgt = os.path.join(args.target_dir, rel)
        os.makedirs(os.path.dirname(tgt), exist_ok=True)

        with open(src_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        missing_periods = missing_conversation_periods(baseline)
        if missing_periods:
            print(f"[skip incomplete baseline] {src_path} missing={missing_periods}", flush=True)
            continue
        if args.skip_existing and no_use_world_complete(tgt, existing_meta_by_world, existing_ops_sample_ids, modes):
            print(f"[skip existing world] {src_path}", flush=True)
            continue
        updated = copy.deepcopy(baseline)
        rebuilt_worlds.add(tgt)

        topic, persona_id, sample_id = extract_topic_and_ids(os.path.basename(src_path))
        blocked_user_texts = set()
        file_has_row = False
        file_rows: List[Dict] = []

        for reveal_period in PERIODS:
            lines = updated.get(reveal_period, [])
            if not isinstance(lines, list) or not lines:
                continue
            reveal_idx = pick_user_turn(lines, min_words=10)
            if reveal_idx is None:
                continue

            reveal_content = split_user_prefix(lines[reveal_idx])
            reveal_fact = extract_reveal_fact(reveal_content)
            blocked_user_texts.add(reveal_content)
            rp = PERIODS.index(reveal_period)

            for mode in modes:
                on_candidates = []
                for p in PERIODS[rp:]:
                    plines = updated.get(p, [])
                    if not isinstance(plines, list) or not plines:
                        continue
                    avoid_idx = reveal_idx if p == reveal_period else None
                    on_idx = pick_user_turn(plines, min_words=8, avoid_idx=avoid_idx)
                    if on_idx is not None:
                        on_candidates.append((p, on_idx))
                if not on_candidates:
                    continue

                on_period, on_anchor_idx = random.choice(on_candidates)
                on_lines = updated[on_period]
                on_anchor_user = split_user_prefix(on_lines[on_anchor_idx])

                # For temporal_scope, make sure an OFF slot exists before mutating conversation.
                off_period = None
                off_anchor_idx = None
                if mode == "temporal_scope":
                    off_candidates = []
                    for p in PERIODS[PERIODS.index(on_period) + 1:]:
                        plines = updated.get(p, [])
                        if not isinstance(plines, list) or not plines:
                            continue
                        off_idx = pick_user_turn(plines, min_words=6)
                        if off_idx is not None:
                            off_candidates.append((p, off_idx))
                    if not off_candidates:
                        continue
                    off_period, off_anchor_idx = random.choice(off_candidates)

                on_tidx, on_text = pick_template(NO_USE_ON_TEMPLATES)
                _, on_ack = pick_template(ack_on_pool)
                on_turn_index, on_ack_index = inject_instruction(on_lines, on_anchor_idx, on_text, on_ack)
                summary["num_on"] += 1
                summary["this_run_num_on"] += 1
                blocked_user_texts.add(on_anchor_user)

                off_turn_index = None
                off_ack_index = None
                off_tidx = None
                off_text = None

                if mode == "temporal_scope":
                    if off_period is None or off_anchor_idx is None:
                        continue
                    off_lines = updated[off_period]
                    off_anchor_user = split_user_prefix(off_lines[off_anchor_idx])
                    off_tidx, off_text = pick_template(NO_USE_OFF_TEMPLATES)
                    _, off_ack = pick_template(ack_off_pool)
                    off_turn_index, off_ack_index = inject_instruction(off_lines, off_anchor_idx, off_text, off_ack)
                    summary["num_off"] += 1
                    summary["this_run_num_off"] += 1
                    blocked_user_texts.add(off_anchor_user)

                sid = (
                    f"p{persona_id}_{topic}_s{sample_id}_{PERIOD_SHORT[reveal_period]}_r{reveal_idx}_"
                    f"{PERIOD_SHORT[on_period]}_on{on_turn_index}_{mode}"
                )

                row = {
                    "sample_id": sid,
                    "persona_id": int(persona_id) if str(persona_id).isdigit() else persona_id,
                    "topic": topic,
                    "file_name": os.path.basename(src_path),
                    "world_file_baseline": src_path,
                    "world_file_no_use": tgt,
                    "mode": mode,
                    "policy_type": mode,
                    "reveal_period": reveal_period,
                    "reveal_turn_index": reveal_idx,
                    "reveal_fact_text": reveal_fact,
                    "no_use_on_period": on_period,
                    "no_use_on_turn_index": on_turn_index,
                    "no_use_on_ack_index": on_ack_index,
                    "no_use_on_text": on_text,
                    "no_use_on_variant": on_tidx,
                    "no_use_off_period": off_period,
                    "no_use_off_turn_index": off_turn_index,
                    "no_use_off_ack_index": off_ack_index,
                    "no_use_off_text": off_text,
                    "no_use_off_variant": off_tidx,
                    "conflict_flag": False,
                    "conflicts_before_repair": 0,
                    "conflicts_after_repair": 0,
                }
                new_meta.append(row)
                file_rows.append(row)
                new_ops.append({
                    "sample_id": sid,
                    "operation": "inject_no_use_on",
                    "period": on_period,
                    "turn_index": on_turn_index,
                    "ack_index": on_ack_index,
                    "line_after": f"User: {on_text}",
                })
                if mode == "temporal_scope":
                    new_ops.append({
                        "sample_id": sid,
                        "operation": "inject_no_use_off",
                        "period": off_period,
                        "turn_index": off_turn_index,
                        "ack_index": off_ack_index,
                        "line_after": f"User: {off_text}",
                })

                file_has_row = True
                summary["this_run_num_rows"] += 1

        if file_has_row:
            utility_pool = extract_utility_facts(baseline, blocked_user_texts, max_items=8)
            for r in file_rows:
                if "utility_facts" not in r:
                    r["utility_facts"] = utility_pool
            conflicts_before = detect_scope_conflicts(updated, file_rows)
            all_repair_logs = []
            curr_conflicts = conflicts_before
            for _ in range(max(1, int(args.max_repair_rounds))):
                if not curr_conflicts:
                    break
                round_logs = apply_model_scope_repairs(
                    data=updated,
                    file_rows=file_rows,
                    conflicts=curr_conflicts,
                    client=client,
                    model=args.model,
                    max_retries=args.max_retries,
                    timeout=args.request_timeout,
                )
                curr_conflicts = detect_scope_conflicts(updated, file_rows)
                if curr_conflicts:
                    local_logs = local_repair_scope_conflicts(updated, curr_conflicts)
                    round_logs.extend(local_logs)
                    curr_conflicts = detect_scope_conflicts(updated, file_rows)
                if not round_logs:
                    break
                all_repair_logs.extend(round_logs)
            conflicts_after = curr_conflicts
            new_ops.extend(all_repair_logs)
            summary["this_run_num_conflicts_before_repair"] += len(conflicts_before)
            summary["this_run_num_repairs"] += len(all_repair_logs)
            summary["this_run_num_conflicts_after_repair"] += len(conflicts_after)
            by_sid_before = {}
            by_sid_after = {}
            for cf in conflicts_before:
                by_sid_before[cf["sample_id"]] = by_sid_before.get(cf["sample_id"], 0) + 1
            for cf in conflicts_after:
                by_sid_after[cf["sample_id"]] = by_sid_after.get(cf["sample_id"], 0) + 1
            for r in file_rows:
                r["conflicts_before_repair"] = by_sid_before.get(r["sample_id"], 0)
                r["conflicts_after_repair"] = by_sid_after.get(r["sample_id"], 0)
                r["conflict_flag"] = r["conflicts_after_repair"] > 0
            summary["this_run_num_files_with_no_use"] += 1

        with open(tgt, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=4, ensure_ascii=False)
        summary["this_run_num_files_processed"] += 1

    replaced_sample_ids = {
        row.get("sample_id")
        for row in existing_meta
        if row.get("world_file_no_use") in rebuilt_worlds
    }
    final_meta = [row for row in existing_meta if row.get("world_file_no_use") not in rebuilt_worlds] + new_meta
    final_ops = [op for op in existing_ops if op.get("sample_id") not in replaced_sample_ids] + new_ops
    final_meta.sort(key=lambda x: x.get("sample_id", ""))
    final_ops.sort(key=lambda x: (x.get("sample_id", ""), x.get("operation", ""), x.get("period", ""), x.get("line_index", -1)))
    final_summary = summarize_no_use_outputs(
        source_files=src_files,
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        meta=final_meta,
        ops=final_ops,
        mode=args.mode,
        ack_source=ack_source,
        ack_on_pool=ack_on_pool,
        ack_off_pool=ack_off_pool,
        cache_path=cache_path,
        this_run=summary,
    )

    write_jsonl(args.meta_path, final_meta)
    write_jsonl(args.ops_path, final_ops)
    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print(json.dumps(final_summary, indent=2))
    print(f"Wrote meta: {args.meta_path}")
    print(f"Wrote ops: {args.ops_path}")
    print(f"Wrote summary: {args.summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build no-use world (scope / temporal_scope).")
    p.add_argument("--source_dir", type=str, default="data/output")
    p.add_argument("--target_dir", type=str, default="data/no_use/world")
    p.add_argument("--meta_path", type=str, default="data/no_use/no_use_meta.jsonl")
    p.add_argument("--ops_path", type=str, default="data/no_use/no_use_ops.jsonl")
    p.add_argument("--summary_path", type=str, default="data/no_use/no_use_summary.json")
    p.add_argument("--mode", type=str, default="both", choices=["scope", "temporal_scope", "both"])
    p.add_argument("--model", type=str, default="gpt-5-mini")
    p.add_argument("--token_path", type=str, default=".")
    p.add_argument("--api_key_file", type=str, default="openai_key.txt")
    p.add_argument("--api_base_url", type=str, default="")
    p.add_argument("--completion_use_templates", action="store_true", help="Use shared completion template pools for ON/OFF acks.")
    p.add_argument("--completion_templates_from_model", action="store_true", help="Regenerate ON/OFF completion template pools via model and overwrite cache.")
    p.add_argument("--completion_template_count", type=int, default=10)
    p.add_argument("--completion_template_cache_path", type=str, default="data/no_use/completion_templates.json")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--max_repair_rounds", type=int, default=3)
    p.add_argument("--request_timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild_target", action="store_true")
    p.add_argument("--skip_existing", action="store_true", help="Skip source files whose world/meta/ops are already complete.")
    return p.parse_args()


if __name__ == "__main__":
    build_no_use_world(parse_args())
