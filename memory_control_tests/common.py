import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PERIODS = [
    "Conversation Initial Stage",
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]

INITIAL_INTERACTION_HISTORY = "Interaction History Initial Stage"
INITIAL_CONVERSATION_HISTORY = "Conversation History Initial Stage"

SIDE_NOTE_RE = re.compile(r"^Side_Note:\s*\[(.*)\]\s+(\d{2}/\d{2}/\d{4}(?:-I\d{2})?)\s*$")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "into", "is", "it", "me", "my", "of", "on", "or", "our",
    "that", "the", "their", "them", "they", "this", "to", "use", "using",
    "was", "we", "with", "you", "your",
}


def load_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str, data: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def normalize_sensitive_value(value: str) -> str:
    return normalize_text(value)


def period_index(period: str) -> int:
    return PERIODS.index(period)


def parse_side_note(line: str) -> Optional[Tuple[str, str]]:
    match = SIDE_NOTE_RE.match((line or "").strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def split_conversation_blocks(lines: List[str]) -> Tuple[List[str], List[Dict]]:
    intro: List[str] = []
    blocks: List[Dict] = []
    current: Optional[Dict] = None

    for idx, line in enumerate(lines or []):
        parsed = parse_side_note(line)
        if parsed:
            if current is not None:
                blocks.append(current)
            event_text, timestamp = parsed
            current = {
                "timestamp": timestamp,
                "event_text": event_text,
                "start_index": idx,
                "lines": [line],
            }
            continue

        if current is None:
            intro.append(line)
        else:
            current["lines"].append(line)

    if current is not None:
        blocks.append(current)
    return intro, blocks


def flatten_sensitive_values(record: Dict) -> List[str]:
    values: List[str] = []
    sensitive = record.get("[Sensitive Info]") or {}
    for arr in sensitive.values():
        if isinstance(arr, list):
            values.extend(str(v) for v in arr if str(v).strip())
    return values


def extract_interaction_candidates(data: Dict) -> List[Dict]:
    conv_history = data.get(INITIAL_CONVERSATION_HISTORY, [])
    convo_lines = data.get("Conversation Initial Stage", [])
    _, blocks = split_conversation_blocks(convo_lines if isinstance(convo_lines, list) else [])
    block_by_timestamp = {block["timestamp"]: block for block in blocks}

    candidates: List[Dict] = []
    for idx, item in enumerate(conv_history):
        if not isinstance(item, dict) or item.get("kind") != "interaction":
            continue
        timestamp = item.get("timestamp")
        if not timestamp:
            continue
        block = block_by_timestamp.get(timestamp, {})
        sensitive_values = flatten_sensitive_values(item)
        candidates.append(
            {
                "rank": len(candidates),
                "timestamp": timestamp,
                "history_index": idx,
                "event_id": item.get("event_id"),
                "source_event_id": item.get("source_event_id"),
                "source_event_date": item.get("source_event_date"),
                "task_goal": item.get("[Task Goal]", ""),
                "prev_event": item.get("[Prev Event]", ""),
                "context_can_add": copy.deepcopy(item.get("[Context Can Add]") or {}),
                "sensitive_info": copy.deepcopy(item.get("[Sensitive Info]") or {}),
                "sensitive_values": sensitive_values,
                "conversation_block": copy.deepcopy(block),
            }
        )
    return candidates


def weighted_terms(text: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for raw in TOKEN_RE.findall(text or ""):
        token = raw.lower().strip(".,!?;:'\"()[]{}")
        if not token or token in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in token):
            weight = 3.0
        elif len(token) >= 10:
            weight = 2.5
        elif len(token) >= 7:
            weight = 2.0
        elif len(token) >= 5:
            weight = 1.5
        else:
            weight = 1.0
        weights[token] = max(weight, weights.get(token, 0.0))
    return weights


def candidate_fact_string(candidate: Dict) -> str:
    parts = list((candidate.get("context_can_add") or {}).keys())
    parts.extend(candidate.get("sensitive_values", []))
    return " ".join(p for p in parts if p)


def score_conflict(text: str, candidate: Dict) -> Tuple[float, List[str]]:
    text_norm = normalize_text(text)
    hits: List[str] = []
    score = 0.0

    for value in candidate.get("sensitive_values", []):
        value_norm = normalize_sensitive_value(value)
        if value_norm and value_norm in text_norm:
            hits.append(value)
            score += 6.0

    fact_terms = weighted_terms(candidate_fact_string(candidate))
    tokens = {
        token.lower().strip(".,!?;:'\"()[]{}")
        for token in TOKEN_RE.findall(text or "")
    }
    for term, weight in fact_terms.items():
        if term in tokens:
            hits.append(term)
            score += weight

    return score, sorted(set(hits))


def detect_future_mentions(data: Dict, candidate: Dict, threshold: float = 8.0) -> List[Dict]:
    mentions: List[Dict] = []
    start = period_index("Conversation Early Stage")
    for period in PERIODS[start:]:
        lines = data.get(period, [])
        if not isinstance(lines, list):
            continue
        for idx, line in enumerate(lines):
            if not isinstance(line, str) or line.startswith("Side_Note"):
                continue
            score, hits = score_conflict(line, candidate)
            if score >= threshold:
                mentions.append(
                    {
                        "period": period,
                        "line_index": idx,
                        "line": line,
                        "score": round(score, 3),
                        "hits": hits,
                    }
                )
    return mentions


def annotate_duplicate_sensitive_values(candidates: List[Dict]) -> None:
    occurrences: Dict[str, List[int]] = {}
    for idx, candidate in enumerate(candidates):
        for value in candidate.get("sensitive_values", []):
            norm = normalize_sensitive_value(value)
            if not norm:
                continue
            occurrences.setdefault(norm, []).append(idx)

    for idx, candidate in enumerate(candidates):
        earlier_duplicate_values = []
        later_duplicate_values = []
        for value in candidate.get("sensitive_values", []):
            norm = normalize_sensitive_value(value)
            refs = occurrences.get(norm, [])
            if len(refs) <= 1:
                continue
            if idx != refs[-1]:
                earlier_duplicate_values.append(value)
            if idx != refs[0]:
                later_duplicate_values.append(value)
        candidate["earlier_duplicate_sensitive_values"] = sorted(set(earlier_duplicate_values))
        candidate["later_duplicate_sensitive_values"] = sorted(set(later_duplicate_values))
        candidate["prefer_as_key"] = not candidate["earlier_duplicate_sensitive_values"]


def choose_key_and_probe_turns(candidates: List[Dict]) -> Dict[str, List[Dict]]:
    annotate_duplicate_sensitive_values(candidates)
    for candidate in candidates:
        candidate["future_mentions"] = candidate.get("future_mentions", [])
        candidate["future_conflict_count"] = len(candidate["future_mentions"])
        candidate["needs_revision"] = candidate["future_conflict_count"] > 0

    if not candidates:
        return {"keys": [], "probes": [], "rejected": []}

    target_keys = len(candidates) // 2
    target_probes = len(candidates) - target_keys

    ranked = sorted(
        candidates,
        key=lambda c: (
            c["future_conflict_count"],
            0 if c["prefer_as_key"] else 1,
            -c["history_index"],
        ),
    )

    keys = ranked[:target_keys]
    remaining = [c for c in ranked if c not in keys]
    probes = sorted(
        remaining,
        key=lambda c: (
            c["future_conflict_count"],
            c["history_index"],
        ),
    )[:target_probes]
    rejected = [c for c in candidates if c not in keys and c not in probes]

    return {"keys": keys, "probes": probes, "rejected": rejected}


def build_baseline_spec(data: Dict, source_path: str) -> Dict:
    candidates = extract_interaction_candidates(data)
    for candidate in candidates:
        candidate["future_mentions"] = detect_future_mentions(data, candidate)

    selection = choose_key_and_probe_turns(candidates)
    return {
        "source_file": source_path,
        "periods": PERIODS,
        "initial_interaction_count": len(candidates),
        "selection_policy": {
            "key_target_count": len(selection["keys"]),
            "probe_target_count": len(selection["probes"]),
            "prefer_last_exact_sensitive_occurrence": True,
            "future_conflict_detection": "later-stage line match using exact sensitive values plus weighted lexical overlap from task/context",
        },
        "key_turns": selection["keys"],
        "protected_probe_turns": selection["probes"],
        "rejected_turns": selection["rejected"],
    }
