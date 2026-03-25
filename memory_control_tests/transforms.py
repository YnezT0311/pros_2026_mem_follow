import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .common import PERIODS, parse_side_note


TEMPLATES = json.loads(
    Path(__file__).with_name("templates.json").read_text(encoding="utf-8")
)


def _find_block_indices(lines: List[str], timestamp: str) -> Optional[Dict[str, int]]:
    current_start = None
    current_timestamp = None
    for idx, line in enumerate(lines):
        parsed = parse_side_note(line)
        if parsed:
            if current_timestamp == timestamp:
                return {"start": current_start, "end": idx}
            current_start = idx
            current_timestamp = parsed[1]
    if current_timestamp == timestamp:
        return {"start": current_start, "end": len(lines)}
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


def _template_group(world_type: str, template_style: str) -> Dict[str, List[str]]:
    group = TEMPLATES.get(world_type, {})
    if isinstance(group, dict) and template_style in group:
        return group[template_style]
    if isinstance(group, dict):
        return group
    return {}


def apply_no_store(
    data: Dict,
    period: str,
    key_timestamp: str,
    template_index: Optional[int] = None,
    template_style: str = "explicit",
    placement: str = "suffix",
) -> Dict:
    out = copy.deepcopy(data)
    lines = out.get(period, [])
    if not isinstance(lines, list):
        return out
    block = _find_block_indices(lines, key_timestamp)
    if not block:
        return out

    group = _template_group("no_store", template_style)
    user_key = "user_prefix" if placement == "prefix" else "user_suffix"
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
    template_index: Optional[int] = None,
    template_style: str = "explicit",
) -> Dict:
    group = _template_group("forget", template_style)
    user_line = _pick(group.get("user", []), template_index)
    assistant_line = _pick(group.get("assistant", []), template_index)
    return append_instruction_turn(data, instruction_period, user_line, assistant_line)


def apply_no_use(
    data: Dict,
    restrict_period: str = "Conversation Early Stage",
    release_period: Optional[str] = None,
    template_index: Optional[int] = None,
    template_style: str = "explicit",
) -> Dict:
    group = _template_group("no_use", template_style)
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


def build_context_messages(data: Dict, ask_period: str) -> List[Dict]:
    if ask_period not in PERIODS:
        return []
    end_idx = PERIODS.index(ask_period)
    messages: List[Dict] = []
    for period in PERIODS[: end_idx + 1]:
        lines = data.get(period, [])
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, str) or line.startswith("Side_Note"):
                continue
            if line.startswith("User:"):
                messages.append({"role": "user", "content": line[len("User:"):].strip()})
            elif line.startswith("Assistant:"):
                messages.append({"role": "assistant", "content": line[len("Assistant:"):].strip()})
            elif line.strip():
                messages.append({"role": "user", "content": line.strip()})
    return messages
