import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import PERIODS, conversation_stage_keys, stage_id_to_conversation_key


TARGET_INSTRUCTION_PERIODS = [
    "Conversation Early Stage",
    "Conversation Intermediate Stage",
    "Conversation Late Stage",
]


TEMPLATES = json.loads(
    Path(__file__).with_name("templates.json").read_text(encoding="utf-8")
)

MEMORY_CONTROL_METADATA_KEY = "_memory_control_metadata"
MEMORY_CONTROL_TRANSFORM_VERSION = "stage_all_v2"
MEMORY_CONTROL_STAGE_TRANSFORM_VERSION = "stage_local_v2"
DEFAULT_FORGET_MIN_SPACING_USER_TURNS = 6
DEFAULT_FORGET_MIN_FINAL_GAP_USER_TURNS = 5


def _find_user_turn_indices(lines: List[str], user_turn: str) -> Optional[Dict[str, int]]:
    needle = " ".join(str(user_turn or "").strip().split())
    if not needle:
        return None
    for idx, line in enumerate(lines):
        if not isinstance(line, str) or not line.startswith("User:"):
            continue
        content = " ".join(line[len("User:"):].strip().split())
        if content == needle or needle in content or content in needle:
            end = min(len(lines), idx + 2)
            return {"start": idx, "end": end}
    return None


def _append_sentence(line: str, sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return line
    if line.endswith((".", "!", "?")):
        return f"{line} {sentence}"
    return f"{line}. {sentence}"


def _merge_assistant_ack(line: str, ack: str) -> str:
    if line.startswith("Assistant:"):
        content = line[len("Assistant:"):].strip()
        return f"Assistant: {ack} {content}".strip()
    return f"Assistant: {ack}"


def _pick(pool: List[str], template_index: Optional[int]) -> str:
    if not pool:
        return ""
    if template_index is None:
        return random.choice(pool)
    return pool[template_index % len(pool)]


def _fill_template(text: str, *, target_reference: str) -> str:
    return text.format(target_reference=target_reference).strip()


def _stable_seed(*parts: str) -> int:
    joined = "\n".join(str(part or "") for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _conversation_periods(data: Dict) -> List[str]:
    return conversation_stage_keys(data) or [period for period in PERIODS if period in data]


def _normalized_turn(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _collect_user_positions(data: Dict) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    global_idx = 0
    for period in _conversation_periods(data):
        lines = data.get(period, [])
        if not isinstance(lines, list):
            continue
        for line_idx, line in enumerate(lines):
            if isinstance(line, str) and line.startswith("User:"):
                positions.append(
                    {
                        "stage": period,
                        "line_index": line_idx,
                        "global_user_turn_index": global_idx,
                        "user_turn": line[len("User:"):].strip(),
                    }
                )
                global_idx += 1
    return positions


def _find_key_position(data: Dict, turn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    user_turn = _normalized_turn(str((turn or {}).get("user_turn", "")))
    stage = stage_id_to_conversation_key(str((turn or {}).get("stage_id", "")))

    positions = _collect_user_positions(data)
    if stage and user_turn:
        for pos in positions:
            if pos["stage"] == stage and _normalized_turn(pos["user_turn"]) == user_turn:
                return pos
    if user_turn:
        for pos in positions:
            candidate = _normalized_turn(pos["user_turn"])
            if candidate == user_turn or user_turn in candidate or candidate in user_turn:
                return pos

    return None


def _choose_distributed_forget_position(
    *,
    candidates: List[Dict[str, Any]],
    rng: random.Random,
    key_index: int,
    key_count: int,
    planned: List[Dict[str, Any]],
    total_original_user_turns: int,
    min_spacing_user_turns: int,
    min_final_gap_user_turns: int,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must be non-empty")

    def with_final_gap(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            pos for pos in items
            if total_original_user_turns - pos["global_user_turn_index"] - 1
            >= min_final_gap_user_turns
        ]

    def with_spacing(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            pos for pos in items
            if all(
                abs(
                    pos["global_user_turn_index"]
                    - prior["insertion_before_global_user_turn_index"]
                ) >= min_spacing_user_turns
                for prior in planned
            )
        ]

    pools = [
        with_spacing(with_final_gap(candidates)),
        with_final_gap(candidates),
        with_spacing(candidates),
        candidates,
    ]
    pool = next((items for items in pools if items), candidates)
    pool = sorted(pool, key=lambda pos: pos["global_user_turn_index"])

    # Spread the selected forget turns across each key's available future
    # window. Random jitter keeps the placement non-mechanical while the target
    # quantile gives the dataset a useful range of reveal->forget gaps.
    denom = max(1, key_count)
    center = (key_index + 0.5) / denom
    jitter = rng.uniform(-0.35 / denom, 0.35 / denom)
    quantile = min(0.95, max(0.05, center + jitter))
    chosen_idx = round(quantile * (len(pool) - 1))
    return pool[max(0, min(chosen_idx, len(pool) - 1))]


def _condense_task_goal(task_goal: str) -> str:
    goal = " ".join(str(task_goal or "").strip().split())
    if not goal:
        return "that earlier request"
    goal = goal.rstrip(".")
    patterns = [
        r"^Kenji seeks assistance in\s+",
        r"^Kenji wants assistance in\s+",
        r"^Kenji wants help with\s+",
        r"^Kenji asks for help with\s+",
        r"^Kenji needs help with\s+",
        r"^Kenji is looking for help with\s+",
        r"^Kenji requests help with\s+",
    ]
    for pattern in patterns:
        goal = re.sub(pattern, "", goal, flags=re.IGNORECASE)
    goal = re.sub(r"^Kenji\s+", "", goal, flags=re.IGNORECASE).strip()
    if not goal:
        return "that earlier request"
    goal = goal[0].lower() + goal[1:] if len(goal) > 1 else goal.lower()
    return goal


def _join_references(refs: List[str]) -> str:
    refs = [ref for ref in refs if ref]
    if not refs:
        return "that earlier request"
    if len(refs) == 1:
        return refs[0]
    if len(refs) == 2:
        return f"{refs[0]} and {refs[1]}"
    return f"{', '.join(refs[:-1])}, and {refs[-1]}"


def _rephrase_label_reference(label: str) -> str:
    label = " ".join(str(label or "").strip().split())
    if not label:
        return ""
    label = label[0].lower() + label[1:] if len(label) > 1 else label.lower()
    return f"that {label}"


def build_key_reference(turns: List[Dict], label_map: Optional[Dict[str, str]] = None) -> str:
    condensed = []
    for turn in turns or []:
        timestamp = str(turn.get("timestamp", "")).strip()
        phrase = ""
        if label_map and timestamp:
            phrase = _rephrase_label_reference(str(label_map.get(timestamp, "")).strip())
        if not phrase:
            phrase = _condense_task_goal(turn.get("task_goal", ""))
        if phrase and phrase not in condensed:
            condensed.append(phrase)
    if not condensed:
        return "that earlier request"
    if len(condensed) == 1:
        return condensed[0]
    sample = condensed[:3]
    return _join_references(sample)

def apply_no_store(
    data: Dict,
    period: str,
    key_timestamp: str,
    user_turn: str = "",
    template_index: Optional[int] = None,
    placement: str = "suffix",
) -> Dict:
    out = copy.deepcopy(data)
    lines = out.get(period, [])
    if not isinstance(lines, list):
        return out
    block = _find_user_turn_indices(lines, user_turn)
    if not block:
        return out

    group = TEMPLATES.get("no_store", {})
    user_key = "user_prefix" if placement == "prefix" else "user_suffix"
    if template_index is None:
        rng = random.Random(_stable_seed("no_store", key_timestamp, user_turn, placement))
        user_pool = group.get(user_key, [])
        assistant_pool = group.get("assistant_ack", [])
        user_text = rng.choice(user_pool) if user_pool else ""
        assistant_ack = rng.choice(assistant_pool) if assistant_pool else ""
    else:
        user_text = _pick(group.get(user_key, []), template_index)
        assistant_ack = _pick(group.get("assistant_ack", []), template_index)

    for idx in range(block["start"], block["end"]):
        if lines[idx].startswith("User:"):
            if placement == "prefix":
                if lines[idx].startswith("User:"):
                    content = lines[idx][len("User:"):].strip()
                    lines[idx] = f"User: {user_text} {content}".strip()
            else:
                lines[idx] = _append_sentence(lines[idx], user_text)
            break
    for idx in range(block["start"], block["end"]):
        if lines[idx].startswith("Assistant:"):
            lines[idx] = _merge_assistant_ack(lines[idx], assistant_ack)
            break
    return out


def append_instruction_turn(
    data: Dict,
    period: str,
    user_line: str,
    assistant_line: str,
) -> Dict:
    out = copy.deepcopy(data)
    lines = out.get(period, [])
    if not isinstance(lines, list):
        return out
    lines.extend([f"User: {user_line}", f"Assistant: {assistant_line}"])
    out[period] = lines
    return out


def apply_forget(
    data: Dict,
    instruction_period: str = "Conversation Early Stage",
    target_reference: str = "that earlier request",
    template_index: Optional[int] = None,
) -> Dict:
    group = TEMPLATES.get("forget", {})
    user_line = _fill_template(_pick(group.get("user", []), template_index), target_reference=target_reference)
    assistant_line = _fill_template(
        _pick(group.get("assistant", []), template_index), target_reference=target_reference
    )
    return append_instruction_turn(data, instruction_period, user_line, assistant_line)


def apply_no_use(
    data: Dict,
    restrict_period: str = "Conversation Early Stage",
    release_period: Optional[str] = None,
    template_index: Optional[int] = None,
) -> Dict:
    group = TEMPLATES.get("no_use", {})
    out = append_instruction_turn(
        data,
        restrict_period,
        _pick(group.get("restrict_user", []), template_index),
        _pick(group.get("restrict_assistant", []), template_index),
    )
    if release_period:
        out = append_instruction_turn(
            out,
            release_period,
            _pick(group.get("release_user", []), template_index),
            _pick(group.get("release_assistant", []), template_index),
        )
    return out


def apply_staged_forget(
    data: Dict,
    target_references: List[str],
    instruction_periods: Optional[List[str]] = None,
    template_index: Optional[int] = None,
) -> Dict:
    periods = instruction_periods or TARGET_INSTRUCTION_PERIODS
    out = copy.deepcopy(data)
    for period, target_reference in zip(periods, target_references):
        out = apply_forget(
            out,
            instruction_period=period,
            target_reference=target_reference,
            template_index=template_index,
        )
    return out


def apply_randomized_forget(
    data: Dict,
    key_turns: List[Dict[str, Any]],
    target_references: List[str],
    template_index: Optional[int] = None,
    min_spacing_user_turns: int = DEFAULT_FORGET_MIN_SPACING_USER_TURNS,
    min_final_gap_user_turns: int = DEFAULT_FORGET_MIN_FINAL_GAP_USER_TURNS,
) -> Dict:
    out = copy.deepcopy(data)
    user_positions = _collect_user_positions(out)
    total_original_user_turns = len(user_positions)
    period_order = {period: idx for idx, period in enumerate(_conversation_periods(out))}

    planned: List[Dict[str, Any]] = []
    key_reference_pairs = list(zip(key_turns or [], target_references or []))
    key_count = len(key_reference_pairs)
    for idx, (turn, target_reference) in enumerate(key_reference_pairs):
        timestamp = str((turn or {}).get("timestamp", "")).strip()
        key_pos = _find_key_position(out, turn)
        if not key_pos:
            continue

        later_positions = [
            pos for pos in user_positions
            if pos["global_user_turn_index"] > key_pos["global_user_turn_index"]
        ]
        rng = random.Random(_stable_seed(timestamp, target_reference, str(idx)))
        if later_positions:
            chosen = _choose_distributed_forget_position(
                candidates=later_positions,
                rng=rng,
                key_index=idx,
                key_count=key_count,
                planned=planned,
                total_original_user_turns=total_original_user_turns,
                min_spacing_user_turns=min_spacing_user_turns,
                min_final_gap_user_turns=min_final_gap_user_turns,
            )
            insert_stage = chosen["stage"]
            insert_line_index = chosen["line_index"]
            insertion_mode = "before_later_user_turn"
            insertion_before_global_user_turn_index = chosen["global_user_turn_index"]
        else:
            insert_stage = key_pos["stage"]
            insert_line_index = len(out.get(insert_stage, []))
            insertion_mode = "append_to_end"
            insertion_before_global_user_turn_index = total_original_user_turns

        group = TEMPLATES.get("forget", {})
        user_pool = group.get("user", [])
        assistant_pool = group.get("assistant", [])
        if template_index is None:
            user_template = rng.choice(user_pool) if user_pool else ""
            assistant_template = rng.choice(assistant_pool) if assistant_pool else ""
        else:
            user_template = _pick(user_pool, template_index)
            assistant_template = _pick(assistant_pool, template_index)
        user_line = _fill_template(user_template, target_reference=target_reference)
        assistant_line = _fill_template(assistant_template, target_reference=target_reference)
        planned.append(
            {
                "key_timestamp": timestamp,
                "key_reference": target_reference,
                "key_stage": key_pos["stage"],
                "key_line_index": key_pos["line_index"],
                "key_global_user_turn_index": key_pos["global_user_turn_index"],
                "forget_stage": insert_stage,
                "original_insert_line_index": insert_line_index,
                "insertion_mode": insertion_mode,
                "insertion_before_global_user_turn_index": insertion_before_global_user_turn_index,
                "forget_user_line": user_line,
                "forget_assistant_line": assistant_line,
            }
        )

    # Apply in conversation order so final line/global indices reflect the
    # transformed history exactly.
    planned.sort(
        key=lambda item: (
            period_order.get(item["forget_stage"], 10_000),
            item["original_insert_line_index"],
            item["key_timestamp"],
        )
    )
    stage_offsets: Dict[str, int] = {}
    inserted_before_original_global = 0
    insertions: List[Dict[str, Any]] = []
    for item in planned:
        stage = item["forget_stage"]
        lines = out.get(stage, [])
        if not isinstance(lines, list):
            continue
        offset = stage_offsets.get(stage, 0)
        final_line_index = item["original_insert_line_index"] + offset
        final_line_index = max(0, min(final_line_index, len(lines)))
        lines[final_line_index:final_line_index] = [
            f"User: {item['forget_user_line']}",
            f"Assistant: {item['forget_assistant_line']}",
        ]
        out[stage] = lines
        stage_offsets[stage] = offset + 2

        forget_global_user_turn_index = (
            item["insertion_before_global_user_turn_index"] + inserted_before_original_global
        )
        final_user_turn_count = total_original_user_turns + len(planned)
        inserted_before_original_global += 1
        insertions.append(
            {
                "key_timestamp": item["key_timestamp"],
                "key_reference": item["key_reference"],
                "key_stage": item["key_stage"],
                "key_line_index": item["key_line_index"],
                "key_global_user_turn_index": item["key_global_user_turn_index"],
                "forget_stage": stage,
                "forget_user_line_index": final_line_index,
                "forget_assistant_line_index": final_line_index + 1,
                "forget_global_user_turn_index": forget_global_user_turn_index,
                "insertion_mode": item["insertion_mode"],
                "insertion_before_original_global_user_turn_index": item[
                    "insertion_before_global_user_turn_index"
                ],
                "reveal_to_forget_user_turn_gap": (
                    forget_global_user_turn_index - item["key_global_user_turn_index"]
                ),
                "forget_to_final_ask_user_turn_gap": (
                    final_user_turn_count - forget_global_user_turn_index - 1
                ),
                "forget_user_line": item["forget_user_line"],
                "forget_assistant_line": item["forget_assistant_line"],
            }
        )

    metadata = dict(out.get(MEMORY_CONTROL_METADATA_KEY, {}))
    metadata["transform_version"] = MEMORY_CONTROL_TRANSFORM_VERSION
    metadata["forget_insertions"] = insertions
    metadata["forget_insertion_policy"] = {
        "placement": "deterministic_random_after_key_turn",
        "distribution": "key_index_quantile_with_jitter",
        "ask_position": "end_of_all_stages",
        "preferred_min_spacing_user_turns": min_spacing_user_turns,
        "preferred_min_final_gap_user_turns": min_final_gap_user_turns,
        "total_original_user_turns": total_original_user_turns,
        "total_transformed_user_turns": total_original_user_turns + len(insertions),
    }
    out[MEMORY_CONTROL_METADATA_KEY] = metadata
    return out


def apply_stage_local_forget(
    data: Dict,
    key_turns: List[Dict[str, Any]],
    target_references: List[str],
    stage_id: str,
    template_index: Optional[int] = None,
    min_gap_after_key_user_turns: int = 2,
) -> Dict:
    """Insert forget instructions after target turns, constrained to one stage."""
    out = copy.deepcopy(data)
    stage_key = stage_id_to_conversation_key(stage_id)
    lines = out.get(stage_key, [])
    if not stage_key or not isinstance(lines, list):
        metadata = dict(out.get(MEMORY_CONTROL_METADATA_KEY, {}))
        metadata["transform_version"] = MEMORY_CONTROL_STAGE_TRANSFORM_VERSION
        metadata["forget_insertions"] = []
        metadata["forget_insertion_policy"] = {
            "placement": "deterministic_after_key_turn_same_stage",
            "ask_position": "end_of_stage",
            "stage_id": stage_id,
        }
        out[MEMORY_CONTROL_METADATA_KEY] = metadata
        return out

    stage_user_positions = [
        {"line_index": line_idx, "stage_user_turn_index": user_idx}
        for user_idx, line_idx in enumerate(
            idx for idx, line in enumerate(lines)
            if isinstance(line, str) and line.startswith("User:")
        )
    ]
    if not stage_user_positions:
        return out

    planned: List[Dict[str, Any]] = []
    key_reference_pairs = [
        (turn, ref)
        for turn, ref in zip(key_turns or [], target_references or [])
        if str((turn or {}).get("stage_id", "")).strip() == stage_id
    ]
    for idx, (turn, target_reference) in enumerate(key_reference_pairs):
        timestamp = str((turn or {}).get("timestamp", "")).strip()
        key_pos = _find_key_position(out, turn)
        if not key_pos or key_pos.get("stage") != stage_key:
            continue
        key_stage_user_index = sum(
            1 for pos in stage_user_positions
            if pos["line_index"] <= key_pos["line_index"]
        ) - 1
        candidates = [
            pos for pos in stage_user_positions
            if pos["stage_user_turn_index"] >= key_stage_user_index + min_gap_after_key_user_turns
        ]
        if candidates:
            rng = random.Random(_stable_seed("stage_local_forget", stage_id, timestamp, target_reference, str(idx)))
            chosen = candidates[min(len(candidates) - 1, rng.randrange(len(candidates)))]
            insert_line_index = chosen["line_index"]
            insertion_mode = "before_later_user_turn_same_stage"
            insertion_before_stage_user_turn_index = chosen["stage_user_turn_index"]
        else:
            insert_line_index = len(lines)
            insertion_mode = "append_to_stage_end"
            insertion_before_stage_user_turn_index = len(stage_user_positions)

        group = TEMPLATES.get("forget", {})
        user_pool = group.get("user", [])
        assistant_pool = group.get("assistant", [])
        if template_index is None:
            rng = random.Random(_stable_seed("stage_local_forget_template", timestamp, target_reference, str(idx)))
            user_template = rng.choice(user_pool) if user_pool else ""
            assistant_template = rng.choice(assistant_pool) if assistant_pool else ""
        else:
            user_template = _pick(user_pool, template_index)
            assistant_template = _pick(assistant_pool, template_index)
        planned.append(
            {
                "key_timestamp": timestamp,
                "key_reference": target_reference,
                "key_stage": stage_key,
                "key_line_index": key_pos["line_index"],
                "key_stage_user_turn_index": key_stage_user_index,
                "forget_stage": stage_key,
                "original_insert_line_index": insert_line_index,
                "insertion_mode": insertion_mode,
                "insertion_before_stage_user_turn_index": insertion_before_stage_user_turn_index,
                "forget_user_line": _fill_template(user_template, target_reference=target_reference),
                "forget_assistant_line": _fill_template(assistant_template, target_reference=target_reference),
            }
        )

    planned.sort(key=lambda item: (item["original_insert_line_index"], item["key_timestamp"]))
    offset = 0
    insertions: List[Dict[str, Any]] = []
    for item in planned:
        final_line_index = max(0, min(item["original_insert_line_index"] + offset, len(lines)))
        lines[final_line_index:final_line_index] = [
            f"User: {item['forget_user_line']}",
            f"Assistant: {item['forget_assistant_line']}",
        ]
        offset += 2
        insertions.append(
            {
                "key_timestamp": item["key_timestamp"],
                "key_reference": item["key_reference"],
                "key_stage": item["key_stage"],
                "key_line_index": item["key_line_index"],
                "key_stage_user_turn_index": item["key_stage_user_turn_index"],
                "forget_stage": stage_key,
                "forget_user_line_index": final_line_index,
                "forget_assistant_line_index": final_line_index + 1,
                "insertion_mode": item["insertion_mode"],
                "insertion_before_stage_user_turn_index": item["insertion_before_stage_user_turn_index"],
                "reveal_to_forget_user_turn_gap": (
                    item["insertion_before_stage_user_turn_index"] - item["key_stage_user_turn_index"]
                ),
                "forget_to_final_ask_user_turn_gap": (
                    len(stage_user_positions) + len(planned)
                    - item["insertion_before_stage_user_turn_index"]
                    - 1
                ),
                "forget_user_line": item["forget_user_line"],
                "forget_assistant_line": item["forget_assistant_line"],
            }
        )
    out[stage_key] = lines

    metadata = dict(out.get(MEMORY_CONTROL_METADATA_KEY, {}))
    metadata["transform_version"] = MEMORY_CONTROL_STAGE_TRANSFORM_VERSION
    metadata["forget_insertions"] = insertions
    metadata["forget_insertion_policy"] = {
        "placement": "deterministic_after_key_turn_same_stage",
        "ask_position": "end_of_stage",
        "stage_id": stage_id,
        "stage_key": stage_key,
        "preferred_min_gap_after_key_user_turns": min_gap_after_key_user_turns,
        "total_stage_original_user_turns": len(stage_user_positions),
        "total_stage_transformed_user_turns": len(stage_user_positions) + len(insertions),
    }
    out[MEMORY_CONTROL_METADATA_KEY] = metadata
    return out


def _stage_user_positions_for_lines(lines: List[str]) -> List[Dict[str, Any]]:
    return [
        {"line_index": line_idx, "stage_user_turn_index": user_idx}
        for user_idx, line_idx in enumerate(
            idx for idx, line in enumerate(lines)
            if isinstance(line, str) and line.startswith("User:")
        )
    ]


def _choose_stage_boundary(
    positions: List[Dict[str, Any]],
    *,
    start_user_index: int,
    seed_parts: List[str],
) -> Dict[str, Any]:
    candidates = [
        pos for pos in positions
        if int(pos["stage_user_turn_index"]) >= start_user_index
    ]
    if not candidates:
        return {
            "line_index": -1,
            "stage_user_turn_index": len(positions),
            "insertion_mode": "append_to_stage_end",
        }
    rng = random.Random(_stable_seed(*seed_parts))
    chosen = candidates[rng.randrange(len(candidates))]
    return {
        "line_index": chosen["line_index"],
        "stage_user_turn_index": chosen["stage_user_turn_index"],
        "insertion_mode": "before_later_user_turn_same_stage",
    }


def apply_stage_local_no_use(
    data: Dict,
    key_turns: List[Dict[str, Any]],
    probe_turns: List[Dict[str, Any]],
    stage_id: str,
    *,
    release: bool = False,
    template_index: Optional[int] = None,
    min_gap_after_key_user_turns: int = 1,
) -> Dict:
    """Insert one stage-local no-use instruction after a target key turn."""
    out = copy.deepcopy(data)
    stage_key = stage_id_to_conversation_key(stage_id)
    lines = out.get(stage_key, [])
    if not stage_key or not isinstance(lines, list):
        metadata = dict(out.get(MEMORY_CONTROL_METADATA_KEY, {}))
        metadata["transform_version"] = MEMORY_CONTROL_STAGE_TRANSFORM_VERSION
        metadata["no_use_insertions"] = []
        out[MEMORY_CONTROL_METADATA_KEY] = metadata
        return out

    stage_user_positions = _stage_user_positions_for_lines(lines)
    stage_key_turns = [
        turn for turn in key_turns or []
        if str((turn or {}).get("stage_id", "")).strip() == stage_id
    ]
    if not stage_user_positions or not stage_key_turns:
        metadata = dict(out.get(MEMORY_CONTROL_METADATA_KEY, {}))
        metadata["transform_version"] = MEMORY_CONTROL_STAGE_TRANSFORM_VERSION
        metadata["no_use_insertions"] = []
        out[MEMORY_CONTROL_METADATA_KEY] = metadata
        return out

    keyed_positions: List[Dict[str, Any]] = []
    for turn in stage_key_turns:
        key_pos = _find_key_position(out, turn)
        if not key_pos or key_pos.get("stage") != stage_key:
            continue
        key_stage_user_index = sum(
            1 for pos in stage_user_positions
            if pos["line_index"] <= key_pos["line_index"]
        ) - 1
        keyed_positions.append(
            {
                "turn": turn,
                "key_pos": key_pos,
                "key_stage_user_turn_index": key_stage_user_index,
            }
        )
    if not keyed_positions:
        metadata = dict(out.get(MEMORY_CONTROL_METADATA_KEY, {}))
        metadata["transform_version"] = MEMORY_CONTROL_STAGE_TRANSFORM_VERSION
        metadata["no_use_insertions"] = []
        out[MEMORY_CONTROL_METADATA_KEY] = metadata
        return out

    # no_use re-partitions turns purely by POSITION relative to the instruction:
    # turns before -> not_remember (memory_control), turns after -> remember (utility).
    # Original key/probe roles do NOT decide the no_use class; position does. We place
    # the instruction at the MEDIAN of the merged key+probe MCQ turns so each stage
    # yields both classes (ceil(N/2) before, floor(N/2) after; N=1 -> the lone turn is
    # not_remember). slot_recall reuses this same placement because it is driven by the
    # shared sidecar key/probe turns + seed.
    # NOTE: no_use utility = MCQ turns after the instruction (expected remember). It is
    # NOT aligned turn-by-turn with the baseline world -- only per-world scores are
    # comparable (same MCQ set, repartitioned by instruction position). If a strict
    # no_use-specific baseline is needed later, add a separate baseline test instead of
    # reusing the shared one.
    def _stage_user_index_of(turn):
        pos = _find_key_position(out, turn)
        if not pos or pos.get("stage") != stage_key:
            return None
        return sum(1 for p in stage_user_positions if p["line_index"] <= pos["line_index"]) - 1

    mcq_turns = []
    for turn in list(key_turns or []) + list(probe_turns or []):
        if str((turn or {}).get("stage_id", "")).strip() != stage_id:
            continue
        ts = str((turn or {}).get("timestamp", "")).strip()
        idx = _stage_user_index_of(turn)
        if ts and idx is not None:
            mcq_turns.append((idx, ts))
    mcq_turns = sorted(set(mcq_turns))
    n_mcq = len(mcq_turns)
    total_user = len(stage_user_positions)
    split = (n_mcq + 1) // 2  # ceil(N/2) MCQ turns fall before the instruction
    s_split = mcq_turns[split - 1][0] if split >= 1 else -1
    s_next = mcq_turns[split][0] if split < n_mcq else None
    target_timestamp = mcq_turns[split - 1][1] if split >= 1 else ""
    key_stage_user_index = s_split

    # Instruction goes after the split-th MCQ turn (seeded random gap 1-3 user turns) but
    # never past the (split+1)-th MCQ turn, so the before/after counts stay ceil/floor.
    gap = random.Random(_stable_seed("stage_local_no_use_place", stage_id)).randint(1, 3)
    restrict_user_index = (s_split if s_split >= 0 else 0) + gap
    if s_next is not None:
        restrict_user_index = min(restrict_user_index, s_next)
    restrict_user_index = min(restrict_user_index, total_user)
    pos_by_user_index = {
        int(p["stage_user_turn_index"]): int(p["line_index"]) for p in stage_user_positions
    }
    if restrict_user_index >= total_user:
        restrict_line_index = len(lines)
        restrict_mode = "append_to_stage_end"
    else:
        restrict_line_index = pos_by_user_index[restrict_user_index]
        restrict_mode = "before_later_user_turn_same_stage"
    restrict = {
        "line_index": restrict_line_index,
        "stage_user_turn_index": restrict_user_index,
        "insertion_mode": restrict_mode,
    }

    group = TEMPLATES.get("no_use", {})
    if template_index is None:
        rng = random.Random(_stable_seed("stage_local_no_use_template", stage_id, target_timestamp))
        restrict_user = rng.choice(group.get("restrict_user", []) or [""])
        restrict_assistant = rng.choice(group.get("restrict_assistant", []) or [""])
        release_user = rng.choice(group.get("release_user", []) or [""])
        release_assistant = rng.choice(group.get("release_assistant", []) or [""])
    else:
        restrict_user = _pick(group.get("restrict_user", []), template_index)
        restrict_assistant = _pick(group.get("restrict_assistant", []), template_index)
        release_user = _pick(group.get("release_user", []), template_index)
        release_assistant = _pick(group.get("release_assistant", []), template_index)

    active_target_timestamps: List[str] = []
    for turn in list(key_turns or []) + list(probe_turns or []):
        if str((turn or {}).get("stage_id", "")).strip() != stage_id:
            continue
        pos = _find_key_position(out, turn)
        if not pos or pos.get("stage") != stage_key:
            continue
        turn_stage_user_index = sum(
            1 for stage_pos in stage_user_positions
            if stage_pos["line_index"] <= pos["line_index"]
        ) - 1
        timestamp = str((turn or {}).get("timestamp", "")).strip()
        if timestamp and turn_stage_user_index < int(restrict["stage_user_turn_index"]):
            active_target_timestamps.append(timestamp)

    lines[restrict_line_index:restrict_line_index] = [
        f"User: {restrict_user}",
        f"Assistant: {restrict_assistant}",
    ]

    release_info: Dict[str, Any] = {}
    if release:
        transformed_positions = _stage_user_positions_for_lines(lines)
        total_t = len(transformed_positions)
        # active -> release: at least 1 turn between (R >= restrict_index + 2);
        # release -> end-of-stage: more than 1 turn after release (R <= total_t - 2).
        lo = int(restrict["stage_user_turn_index"]) + 2
        hi = total_t - 2
        release_user_index = None
        if lo <= hi:
            release_user_index = random.Random(
                _stable_seed("stage_local_no_use_release", stage_id, target_timestamp)
            ).randint(lo, hi)
        if release_user_index is not None:
            release_line_index = next(
                int(p["line_index"]) for p in transformed_positions
                if int(p["stage_user_turn_index"]) == release_user_index
            )
            lines[release_line_index:release_line_index] = [
                f"User: {release_user}",
                f"Assistant: {release_assistant}",
            ]
            release_info = {
                "release_user_line_index": release_line_index,
                "release_assistant_line_index": release_line_index + 1,
                "release_insertion_mode": "before_later_user_turn_same_stage",
                "release_before_stage_user_turn_index": release_user_index,
                "release_user_line": release_user,
                "release_assistant_line": release_assistant,
            }
        else:
            release_info = {"release_skipped_reason": "insufficient_room_for_release_gaps"}

    out[stage_key] = lines
    insertion = {
        "key_timestamp": target_timestamp,
        "key_stage": stage_key,
        "key_stage_user_turn_index": key_stage_user_index,
        "restrict_user_line_index": restrict_line_index,
        "restrict_assistant_line_index": restrict_line_index + 1,
        "restrict_insertion_mode": restrict["insertion_mode"],
        "restrict_before_stage_user_turn_index": restrict["stage_user_turn_index"],
        "restrict_user_line": restrict_user,
        "restrict_assistant_line": restrict_assistant,
        "active_target_timestamps": sorted(set(active_target_timestamps)),
    }
    insertion.update(release_info)
    metadata = dict(out.get(MEMORY_CONTROL_METADATA_KEY, {}))
    metadata["transform_version"] = MEMORY_CONTROL_STAGE_TRANSFORM_VERSION
    metadata["no_use_insertions"] = [insertion]
    metadata["no_use_policy"] = {
        "placement": "median_of_key_plus_probe_turns",
        "release": release,
        "ask_position": "end_of_stage",
        "stage_id": stage_id,
        "stage_key": stage_key,
        "active_gap_after_split_user_turns": "random_1_3",
        "release_gap_min_between_active_and_release_user_turns": 1,
        "release_gap_min_after_release_user_turns": 2,
        "n_mcq_turns": n_mcq,
        "n_before_instruction": split,
        "total_stage_original_user_turns": len(stage_user_positions),
    }
    out[MEMORY_CONTROL_METADATA_KEY] = metadata
    return out


def apply_staged_no_use(
    data: Dict,
    restrict_periods: Optional[List[str]] = None,
    template_index: Optional[int] = None,
) -> Dict:
    periods = restrict_periods or TARGET_INSTRUCTION_PERIODS
    out = copy.deepcopy(data)
    for period in periods:
        out = apply_no_use(
            out,
            restrict_period=period,
            release_period=None,
            template_index=template_index,
        )
    return out


def build_context_messages(data: Dict, ask_period: str, stage_id: str = "") -> List[Dict]:
    periods = conversation_stage_keys(data)
    if stage_id:
        stage_key = stage_id_to_conversation_key(stage_id)
        periods = [stage_key] if stage_key in data else []
    if not periods:
        if ask_period not in PERIODS:
            return []
        end_idx = PERIODS.index(ask_period)
        periods = PERIODS[: end_idx + 1]
    messages: List[Dict] = []
    for period in periods:
        lines = data.get(period, [])
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, str):
                continue
            if line.startswith("User:"):
                messages.append({"role": "user", "content": line[len("User:"):].strip()})
            elif line.startswith("Assistant:"):
                messages.append({"role": "assistant", "content": line[len("Assistant:"):].strip()})
            elif line.strip():
                messages.append({"role": "user", "content": line.strip()})
    return messages
