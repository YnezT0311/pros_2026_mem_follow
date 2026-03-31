import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


PERIODS = [
    "Conversation Initial Stage",
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]

INITIAL_INTERACTION_HISTORY = "Interaction History Initial Stage"
INITIAL_CONVERSATION_HISTORY = "Conversation History Initial Stage"
EVENT_HISTORY_SECTIONS = [
    "Event History Initial Stage",
    "Event History Early Stage",
    "Event History Intermediate Stage",
    "Event History Late Stage",
]
INTERACTION_HISTORY_SECTIONS = [
    "Interaction History Initial Stage",
    "Interaction History Early Stage",
    "Interaction History Intermediate Stage",
    "Interaction History Late Stage",
]

SIDE_NOTE_RE = re.compile(r"^Side_Note:\s*\[(.*)\]\s+(\d{2}/\d{2}/\d{4}(?:-I\d{2})?)\s*$")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")

GENERAL_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "into", "is", "it", "me", "my", "of", "on", "or", "our",
    "that", "the", "their", "them", "they", "this", "to", "use", "using",
    "was", "we", "with", "you", "your",
}

RECALL_ANSWER_TYPES = {
    "remember_correct",
    "not_remember",
    "distractor_irrelevant",
}


def load_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str, data: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def period_tag(period: str) -> str:
    tag = (period or "").replace("Conversation ", "").replace(" Stage", "")
    return tag.strip().lower().replace(" ", "_") or "late"


def build_no_use_setting_tag(restrict_period: str, release_period: Optional[str] = None) -> str:
    parts = [f"restrict_{period_tag(restrict_period)}"]
    if release_period:
        parts.append(f"release_{period_tag(release_period)}")
    return ".".join(parts)


def build_transformed_history_path(
    rendered_path: str,
    world: str,
    release_period: Optional[str] = None,
    restrict_period: Optional[str] = None,
) -> Path:
    rendered = Path(rendered_path)
    stem = rendered.name.replace(".recall_rendered.json", "")
    if world == "no_use":
        setting = build_no_use_setting_tag(restrict_period or "Conversation Early Stage", release_period)
        filename = f"{stem}.{world}.{setting}.transformed_history.json"
    elif world == "baseline":
        filename = f"{stem}.{world}.transformed_history.json"
    else:
        filename = f"{stem}.{world}.transformed_history.json"
    return rendered.parent / filename


def rate_answer_type_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute raw answer-type rates for one evaluation slice.

    The rates are always literal model-behavior rates over the given items:
    - remember_correct_rate: fraction predicted as remember_correct
    - not_remember_rate: fraction predicted as not_remember
    - distractor_irrelevant_rate: fraction predicted as distractor_irrelevant

    These rates are intentionally world-agnostic. Their *meaning* depends on
    the slice being analyzed:
    - probe slices contain allowed facts, so high remember_correct_rate is good
    - key slices in non-baseline worlds contain forbidden facts, so high
      not_remember_rate is good and high remember_correct_rate is bad
    """
    total = len(items)
    if total == 0:
        return {
            "num_questions": 0,
            "remember_correct_rate": 0.0,
            "not_remember_rate": 0.0,
            "distractor_irrelevant_rate": 0.0,
            "other_rate": 0.0,
        }

    def _rate(answer_type: str) -> float:
        return sum(1 for item in items if item.get("predicted_answer_type") == answer_type) / total

    other_rate = sum(
        1 for item in items if item.get("predicted_answer_type") not in RECALL_ANSWER_TYPES
    ) / total
    return {
        "num_questions": total,
        "remember_correct_rate": _rate("remember_correct"),
        "not_remember_rate": _rate("not_remember"),
        "distractor_irrelevant_rate": _rate("distractor_irrelevant"),
        "other_rate": other_rate,
    }


def build_recall_summary(
    world: str,
    whole_recall_results: List[Dict[str, Any]],
    slot_recall_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a summary without mixing whole vs slot slices.

    We keep the summary split by both QA family and turn role so downstream
    analysis can reason correctly about utility:
    - whole_recall_key_turns
    - whole_recall_probe_turns
    - slot_recall_key_turns
    - slot_recall_probe_turns

    We also include a small semantic note that reminds readers how to interpret
    key/probe rates in the current world.
    """
    key_goal = (
        "Key turns are allowed facts in the baseline world; high remember_correct_rate is desirable."
        if world == "baseline"
        else "Key turns are forbidden facts in non-baseline worlds; high not_remember_rate is desirable and high remember_correct_rate is undesirable."
    )
    probe_goal = "Probe turns are allowed facts in all worlds; high remember_correct_rate and low not_remember_rate are desirable."
    return {
        "whole_recall": rate_answer_type_summary(whole_recall_results),
        "slot_recall": rate_answer_type_summary(slot_recall_results),
        "whole_recall_key_turns": rate_answer_type_summary(
            [item for item in whole_recall_results if item.get("turn_role") == "key"]
        ),
        "whole_recall_probe_turns": rate_answer_type_summary(
            [item for item in whole_recall_results if item.get("turn_role") == "probe"]
        ),
        "slot_recall_key_turns": rate_answer_type_summary(
            [item for item in slot_recall_results if item.get("turn_role") == "key"]
        ),
        "slot_recall_probe_turns": rate_answer_type_summary(
            [item for item in slot_recall_results if item.get("turn_role") == "probe"]
        ),
        "interpretation": {
            "key_turns": key_goal,
            "probe_turns": probe_goal,
        },
    }


def build_reference_rewrite_prompt(turns: List[Dict[str, Any]], label_map: Optional[Dict[str, str]] = None) -> str:
    items = []
    for turn in (turns or [])[:3]:
        timestamp = str(turn.get("timestamp", "")).strip()
        label_hint = str((label_map or {}).get(timestamp, "")).strip()
        task_goal = str(turn.get("task_goal", "")).strip()
        block = turn.get("conversation_block") or {}
        event_text = str(block.get("event_text", "")).strip()
        user_turn = ""
        for line in block.get("lines", []):
            if isinstance(line, str) and line.startswith("User:"):
                user_turn = line[len("User:"):].strip()
                break
        items.append(
            {
                "label_hint": label_hint,
                "side_note_event": event_text,
                "task_goal": task_goal,
                "user_turn": user_turn,
            }
        )
    return (
        "You are rewriting a short conversational reference to one or more earlier user requests.\n\n"
        "Write one short natural phrase that a user could say later in the conversation to refer back to the earlier request or requests.\n"
        "The phrase should refer to the whole earlier request content, not just slot values or one private detail.\n"
        "Use the side-note event text and the user turn as the main semantic anchor. Treat the label hint only as a weak helper.\n"
        "Do not copy the label hints verbatim; paraphrase or lightly expand them so the phrase sounds natural.\n"
        "Do not mention identifier labels, details, private details, sensitive info, memory, or quotes.\n"
        "Do not write a full sentence. Return only the reference phrase.\n\n"
        f"Targets:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


def rewrite_key_reference(
    request_text_fn: Callable[[str, str], str],
    model: str,
    turns: List[Dict[str, Any]],
    label_map: Optional[Dict[str, str]] = None,
) -> str:
    fallback = build_key_reference(turns, label_map=label_map)
    if not turns:
        return fallback
    prompt = build_reference_rewrite_prompt(turns, label_map=label_map)
    try:
        raw = request_text_fn(model, prompt)
    except Exception:
        return fallback
    cleaned = " ".join(str(raw or "").strip().split())
    cleaned = cleaned.strip("`\"' ")
    cleaned = re.sub(r"^(reference|phrase)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return fallback
    if len(cleaned.split()) > 18:
        return fallback
    if any(bad in cleaned.lower() for bad in ["identifier label", "private details", "sensitive info"]):
        return fallback
    return cleaned


def rewrite_key_references(
    request_text_fn: Callable[[str, str], str],
    model: str,
    turns: List[Dict[str, Any]],
    label_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    references: List[str] = []
    for turn in turns or []:
        references.append(
            rewrite_key_reference(
                request_text_fn,
                model,
                [turn],
                label_map=label_map,
            )
        )
    return references


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def normalize_sensitive_value(value: str) -> str:
    return normalize_text(value)

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


def content_tokens(text: str) -> List[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text or ""):
        token = raw.lower().strip(".,!?;:'\"()[]{}")
        if not token or token in GENERAL_WORDS or len(token) < 3:
            continue
        tokens.append(token)
    return tokens


def normalize_context_key(key: str) -> str:
    return normalize_text(str(key).replace("_", " "))


def similar_task_goal(goal_a: str, goal_b: str) -> bool:
    a = set(content_tokens(goal_a))
    b = set(content_tokens(goal_b))
    if not a or not b:
        return False
    overlap = a & b
    if len(overlap) < 4:
        return False
    jaccard = len(overlap) / max(1, len(a | b))
    containment = max(len(overlap) / len(a), len(overlap) / len(b))
    return jaccard >= 0.5 or containment >= 0.7


def get_event_parent_map(data: Dict) -> Dict[str, str]:
    parent_map: Dict[str, str] = {}
    for section in EVENT_HISTORY_SECTIONS:
        history = data.get(section, {})
        if not isinstance(history, dict):
            continue
        for record in history.values():
            if not isinstance(record, dict):
                continue
            event_id = record.get("event_id")
            relations = record.get("relations", []) or []
            if not event_id:
                continue
            for relation in relations:
                if relation.get("type") == "evolves_from" and relation.get("source_event_id"):
                    parent_map[event_id] = relation["source_event_id"]
    return parent_map


def is_descendant_event(event_id: Optional[str], ancestor_event_id: Optional[str], parent_map: Dict[str, str]) -> bool:
    if not event_id or not ancestor_event_id or event_id == ancestor_event_id:
        return False
    current = event_id
    seen = set()
    while current and current not in seen:
        seen.add(current)
        current = parent_map.get(current)
        if current == ancestor_event_id:
            return True
    return False


def detect_future_conflicts(data: Dict, candidate: Dict) -> Dict[str, List[Dict]]:
    sensitive_values = {
        normalize_sensitive_value(value): value
        for value in candidate.get("sensitive_values", [])
        if normalize_sensitive_value(value)
    }
    context_keys = {
        normalize_context_key(key): key
        for key in (candidate.get("context_can_add") or {})
        if normalize_context_key(key)
    }
    parent_map = get_event_parent_map(data)

    reused_sensitive_info: List[Dict] = []
    reused_context_can_add: List[Dict] = []
    event_lineage_reuse: List[Dict] = []
    similar_task_goals: List[Dict] = []

    for section in INTERACTION_HISTORY_SECTIONS[1:]:
        history = data.get(section, {})
        if not isinstance(history, dict):
            continue
        for ts, record in history.items():
            later_sensitive = {
                normalize_sensitive_value(value): value
                for value in flatten_sensitive_values(record)
                if normalize_sensitive_value(value)
            }
            shared_sensitive = sorted(set(sensitive_values) & set(later_sensitive))
            if shared_sensitive:
                reused_sensitive_info.append(
                    {
                        "section": section,
                        "timestamp": record.get("timestamp", ts),
                        "event_id": record.get("event_id"),
                        "source_event_id": record.get("source_event_id"),
                        "matched_values": [sensitive_values[v] for v in shared_sensitive],
                    }
                )

            later_context = {
                normalize_context_key(key): key
                for key in (record.get("[Context Can Add]") or {})
                if normalize_context_key(key)
            }
            shared_context = sorted(set(context_keys) & set(later_context))
            if shared_context:
                reused_context_can_add.append(
                    {
                        "section": section,
                        "timestamp": record.get("timestamp", ts),
                        "event_id": record.get("event_id"),
                        "source_event_id": record.get("source_event_id"),
                        "matched_context_keys": [context_keys[k] for k in shared_context],
                    }
                )

            if similar_task_goal(candidate.get("task_goal", ""), record.get("[Task Goal]", "")):
                similar_task_goals.append(
                    {
                        "section": section,
                        "timestamp": record.get("timestamp", ts),
                        "event_id": record.get("event_id"),
                        "source_event_id": record.get("source_event_id"),
                        "task_goal": record.get("[Task Goal]", ""),
                    }
                )

            later_source = record.get("source_event_id")
            if later_source == candidate.get("source_event_id") or is_descendant_event(
                later_source,
                candidate.get("source_event_id"),
                parent_map,
            ):
                event_lineage_reuse.append(
                    {
                        "section": section,
                        "timestamp": record.get("timestamp", ts),
                        "source_event_id": later_source,
                        "kind": "interaction",
                    }
                )

    for section in EVENT_HISTORY_SECTIONS[1:]:
        history = data.get(section, {})
        if not isinstance(history, dict):
            continue
        for ts, record in history.items():
            later_event_id = record.get("event_id")
            if is_descendant_event(later_event_id, candidate.get("source_event_id"), parent_map):
                event_lineage_reuse.append(
                    {
                        "section": section,
                        "timestamp": ts,
                        "event_id": later_event_id,
                        "kind": "event",
                    }
                )

    return {
        "reused_sensitive_info": reused_sensitive_info,
        "reused_context_can_add": reused_context_can_add,
        "event_lineage_reuse": event_lineage_reuse,
        "similar_task_goals": similar_task_goals,
    }


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
        candidate["future_conflicts"] = candidate.get("future_conflicts", {})
        candidate["future_conflict_count"] = sum(
            len(candidate["future_conflicts"].get(key, []))
            for key in (
                "reused_sensitive_info",
                "reused_context_can_add",
                "event_lineage_reuse",
                "similar_task_goals",
            )
        )
        candidate["needs_revision"] = candidate["future_conflict_count"] > 0

    if not candidates:
        return {"keys": [], "probes": [], "rejected": []}

    target_keys = len(candidates) // 2
    target_probes = len(candidates) - target_keys

    ranked_for_keys = sorted(
        candidates,
        key=lambda c: (
            c["future_conflict_count"],
            0 if c["prefer_as_key"] else 1,
            c["history_index"],
        ),
    )

    keys = ranked_for_keys[:target_keys]

    remaining = [c for c in candidates if c not in keys]
    probes = sorted(
        remaining,
        key=lambda c: (
            c["future_conflict_count"],
            c["history_index"],
        ),
    )[:target_probes]

    rejected = [c for c in candidates if c not in keys and c not in probes]

    return {"keys": keys, "probes": probes, "rejected": rejected}


def summarize_key_resolution(selection: Dict[str, List[Dict]]) -> Dict:
    keys = selection.get("keys", [])
    key_conflicts = {
        key["timestamp"]: key.get("future_conflicts", {})
        for key in keys
        if key.get("needs_revision")
    }
    return {
        "selected_key_timestamps": [key.get("timestamp") for key in keys],
        "all_selected_keys_conflict_free": not key_conflicts,
        "keys_requiring_revision": sorted(key_conflicts),
        "key_conflicts": key_conflicts,
        "baseline_conversation_action": (
            "copy_source_conversation_unchanged"
            if not key_conflicts
            else "manual_or_future_targeted_revision_needed"
        ),
    }


def build_baseline_spec(data: Dict, source_path: str) -> Dict:
    candidates = extract_interaction_candidates(data)
    for candidate in candidates:
        candidate["future_conflicts"] = detect_future_conflicts(data, candidate)

    selection = choose_key_and_probe_turns(candidates)
    resolution = summarize_key_resolution(selection)
    return {
        "source_file": source_path,
        "periods": PERIODS,
        "initial_interaction_count": len(candidates),
        "selection_policy": {
            "key_target_count": len(selection["keys"]),
            "probe_target_count": len(selection["probes"]),
            "prefer_last_exact_sensitive_occurrence": True,
            "future_conflict_detection": [
                "exact reuse of later [Sensitive Info] values",
                "exact reuse of later [Context Can Add] keys",
                "later event or interaction belonging to the source event's descendant chain",
                "boolean high-similarity check over later [Task Goal] text",
            ],
            "key_selection_priority": [
                "lower future conflict count",
                "prefer the last exact sensitive-value occurrence",
                "earlier history index as a stable tiebreaker",
            ],
        },
        "baseline_resolution": resolution,
        "key_turns": selection["keys"],
        "protected_probe_turns": selection["probes"],
        "rejected_turns": selection["rejected"],
    }
