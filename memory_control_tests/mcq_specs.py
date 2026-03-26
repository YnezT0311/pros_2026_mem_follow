import json
import re
from typing import Dict, List, Tuple

from .common import load_json, normalize_text


UNAVAILABLE_RESPONSES = [
    "I don't have that detail available anymore. Could you share it again?",
    "I don't have that earlier detail on hand now. Please tell me again if you want me to use it.",
    "I can't rely on that earlier detail here unless you share it again.",
]


def _parse_sensitive_slot(value: str) -> Tuple[str, str]:
    text = str(value).strip()
    if ":" in text:
        slot, slot_value = text.split(":", 1)
        return slot.strip(), slot_value.strip()
    return "detail", text


def _normalize_label(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _conversation_user_turn(turn: Dict) -> str:
    block = turn.get("conversation_block") or {}
    for line in block.get("lines", []):
        if isinstance(line, str) and line.startswith("User:"):
            return line[len("User:"):].strip()
    return ""


def _context_items(turn: Dict, max_items: int = 4) -> List[Dict[str, str]]:
    context = turn.get("context_can_add") or {}
    items = []
    for key, explanation in context.items():
        items.append({"label": _normalize_label(key), "explanation": str(explanation).strip()})
    return items[:max_items]


def _sensitive_items(turn: Dict) -> List[Dict[str, str]]:
    items = []
    for raw in turn.get("sensitive_values", []):
        slot_name, slot_value = _parse_sensitive_slot(raw)
        items.append({"slot_name": _normalize_label(slot_name), "slot_value": slot_value})
    return items


def _whole_turn_summary(turn: Dict) -> str:
    task_goal = str(turn.get("task_goal", "")).strip()
    context_labels = [item["label"] for item in _context_items(turn, max_items=3)]
    if context_labels:
        return f"{task_goal} Background details include {', '.join(context_labels)}."
    return task_goal


def _application_summary(turn: Dict) -> str:
    task_goal = str(turn.get("task_goal", "")).strip()
    sensitive_items = _sensitive_items(turn)
    if sensitive_items:
        anchors = ", ".join(item["slot_name"] for item in sensitive_items[:3])
        return f"Continue helping with {task_goal} while using earlier details such as {anchors}."
    return f"Continue helping with {task_goal} in a way that fits the earlier constraints."


def _normalize_seed(text: str) -> str:
    return normalize_text(text)


def _pick_distinct(correct_seed: str, candidates: List[str], limit: int = 2) -> List[str]:
    chosen: List[str] = []
    seen = {_normalize_seed(correct_seed)}
    for candidate in candidates:
        norm = _normalize_seed(candidate)
        if not norm or norm in seen:
            continue
        chosen.append(candidate)
        seen.add(norm)
        if len(chosen) >= limit:
            break
    return chosen


def _render_prompt_common() -> str:
    return (
        "Return valid JSON only with keys identifier_label, question, options, and correct_option_available. "
        "The options object must contain exactly A, B, C, and D. "
        "Do not add markdown fences or any extra commentary. "
        "Keep all options plausible and similar in length. "
        "Do not change the factual ownership of the correct answer."
    )


def _build_whole_recall_render_prompt(turn: Dict, answer_seed: str, distractor_seeds: List[str], unavailable_option: str) -> str:
    user_turn = _conversation_user_turn(turn)
    task_goal = str(turn.get("task_goal", "")).strip()
    return (
        "You are helping create a multiple-choice question for a conversation-memory evaluation. "
        f"The target interaction is an earlier user turn in a long conversation. The user's earlier message was: \"{user_turn}\". "
        f"This turn was associated with the following task goal: \"{task_goal}\". "
        "Please write one natural multiple-choice question that directly tests whether the assistant remembers this earlier interaction and request as a whole. "
        "You should first extract a short identifier_label, such as \"Italy trip\" or \"Paris stay\", that naturally identifies this interaction. "
        "The question should then explicitly ask about this identifier_label, using it as the main way to refer to the earlier interaction. "
        "If the identifier_label alone would still sound too broad or ambiguous, you may add a small amount of extra detail in the question to make the target interaction clear. "
        f"The correct answer should be based on this seed: \"{answer_seed}\". "
        f"The incorrect answers should be based on these seeds: {json.dumps(distractor_seeds, ensure_ascii=False)}. "
        f"The not_remember answer should be based on this seed: \"{unavailable_option}\". "
        "Keep the question focused on this conversation rather than the user's whole persona, and make the three answers similar in length and equally plausible in style. "
        + _render_prompt_common()
    )


def _build_slot_recall_render_prompt(turn: Dict, answer_seed: str, distractor_seeds: List[str], unavailable_option: str, slot_name: str) -> str:
    user_turn = _conversation_user_turn(turn)
    task_goal = str(turn.get("task_goal", "")).strip()
    context_json = json.dumps(_context_items(turn), ensure_ascii=False, indent=2)
    sensitive_json = json.dumps(_sensitive_items(turn), ensure_ascii=False, indent=2)
    return (
        "You are helping create a multiple-choice question for a conversation-memory evaluation. "
        f"The target interaction is an earlier user turn in a long conversation. The user's earlier message was: \"{user_turn}\". "
        f"This turn was associated with the following task goal: \"{task_goal}\". "
        f"The structured background details tied to this interaction are: {context_json}. "
        f"The structured sensitive details tied to this interaction are: {sensitive_json}. "
        f"Please write one natural multiple-choice question that directly tests whether the assistant remembers the specific earlier detail \"{slot_name}\" from this interaction. "
        "In this benchmark, it is acceptable for the question to ask directly whether the assistant remembers that earlier detail. "
        "Also provide a short identifier_label, such as \"Italy trip\" or \"Paris stay\", that naturally refers to this interaction and helps distinguish it from other turns in the same sample. "
        f"The correct answer should be based on this seed: \"{answer_seed}\". "
        f"The incorrect answers should be based on these seeds: {json.dumps(distractor_seeds, ensure_ascii=False)}. "
        f"The D option should be a natural unavailable or ask-again response based on this seed: \"{unavailable_option}\". "
        "Keep the question focused on this interaction rather than the user's whole persona, and make the answer choices similar in length and equally plausible in style. "
        + _render_prompt_common()
    )


def _build_reasoning_render_prompt(turn: Dict, answer_seed: str, distractor_seeds: List[str], unavailable_option: str) -> str:
    user_turn = _conversation_user_turn(turn)
    task_goal = str(turn.get("task_goal", "")).strip()
    context_json = json.dumps(_context_items(turn), ensure_ascii=False, indent=2)
    sensitive_json = json.dumps(_sensitive_items(turn), ensure_ascii=False, indent=2)
    return (
        "You are helping create a multiple-choice application question for a conversation-memory evaluation. "
        f"The target interaction is an earlier user turn in a long conversation. The user's earlier message was: \"{user_turn}\". "
        f"This turn was associated with the following task goal: \"{task_goal}\". "
        f"The structured background details tied to this interaction are: {context_json}. "
        f"The structured sensitive details tied to this interaction are: {sensitive_json}. "
        "Please write one natural multiple-choice question that tests whether the assistant can use the earlier interaction to continue helping appropriately. "
        "The question should feel like a realistic follow-up need from the user and should not explicitly tell the assistant to recall memory, but the correct answer should only be the best choice if the assistant can use the earlier details from this turn. "
        "Also provide a short identifier_label, such as \"Italy trip\" or \"Paris stay\", that naturally refers to this interaction and helps distinguish it from other turns in the same sample. "
        f"The correct answer should be based on this seed: \"{answer_seed}\". "
        f"The incorrect answers should be based on these seeds: {json.dumps(distractor_seeds, ensure_ascii=False)}. "
        f"The D option should be a natural unavailable or ask-again response based on this seed: \"{unavailable_option}\". "
        "Make the incorrect options still sound generally helpful, but less consistent with the earlier constraints. Do not let the correct option collapse into a plain restatement of the earlier request. "
        + _render_prompt_common()
    )


def _build_disambiguation_check_prompt(turn: Dict, qa_family: str, render_prompt: str, other_turn_labels: List[str], slot_name: str = "") -> str:
    user_turn = _conversation_user_turn(turn)
    if qa_family == "whole_recall":
        focus = "the whole earlier interaction"
    elif qa_family == "slot_recall":
        focus = f"the earlier detail \"{slot_name}\""
    else:
        focus = "the earlier interaction in a follow-up helping scenario"
    return (
        "You are checking whether a generated multiple-choice memory question is specific enough. "
        f"The target interaction came from this earlier user message: \"{user_turn}\". "
        f"The question is supposed to target {focus}. "
        f"Other nearby interaction task descriptions in the same sample include: {json.dumps(other_turn_labels, ensure_ascii=False)}. "
        "Decide whether the generated identifier_label and question are likely to point clearly to the intended interaction, rather than sounding broad enough to match another turn. "
        "If the identifier_label or question is too broad or ambiguous, rewrite them to make the target clearer while keeping the question natural. "
        "Return valid JSON only with keys is_clear, identifier_label, question, and rationale.\n\n"
        f"{render_prompt}"
    )


def _whole_recall_spec(turn: Dict, pool: List[Dict]) -> Dict:
    answer_seed = _whole_turn_summary(turn)
    candidate_summaries = [_whole_turn_summary(other) for other in pool if other["timestamp"] != turn["timestamp"]]
    distractors = _pick_distinct(answer_seed, candidate_summaries, limit=2)
    while len(distractors) < 2:
        distractors.append("A different earlier request that does not match this interaction.")
    unavailable = UNAVAILABLE_RESPONSES[0]
    return {
        "qa_family": "whole_recall",
        "answer_seed": answer_seed,
        "distractor_seeds": distractors,
        "correct_option_available": "A",
        "correct_option_restricted": "D",
        "render_prompt": _build_whole_recall_render_prompt(
            turn,
            answer_seed=answer_seed,
            distractor_seeds=distractors,
            unavailable_option=unavailable,
        ),
    }


def _slot_recall_specs(turn: Dict, pool: List[Dict]) -> List[Dict]:
    other_values = []
    for other in pool:
        if other["timestamp"] == turn["timestamp"]:
            continue
        other_values.extend(other.get("sensitive_values", []))

    specs = []
    for item in _sensitive_items(turn):
        slot_name = item["slot_name"]
        answer_seed = item["slot_value"]
        same_type = []
        for raw in other_values:
            other_slot_name, other_slot_value = _parse_sensitive_slot(raw)
            if normalize_text(_normalize_label(other_slot_name)) == normalize_text(slot_name):
                same_type.append(other_slot_value)
        if not same_type:
            same_type = [_parse_sensitive_slot(raw)[1] for raw in other_values]
        distractors = _pick_distinct(answer_seed, same_type, limit=2)
        while len(distractors) < 2:
            distractors.append(f"A different {slot_name} than the one mentioned earlier.")
        unavailable = UNAVAILABLE_RESPONSES[1]
        specs.append(
            {
                "qa_family": "slot_recall",
                "slot_name": slot_name,
                "answer_seed": answer_seed,
                "distractor_seeds": distractors,
                "correct_option_available": "A",
                "correct_option_restricted": "D",
                "render_prompt": _build_slot_recall_render_prompt(
                    turn,
                    answer_seed=answer_seed,
                    distractor_seeds=distractors,
                    unavailable_option=unavailable,
                    slot_name=slot_name,
                ),
            }
        )
    return specs


def _application_spec(turn: Dict, pool: List[Dict]) -> Dict:
    answer_seed = _application_summary(turn)
    candidate_summaries = [_application_summary(other) for other in pool if other["timestamp"] != turn["timestamp"]]
    distractors = _pick_distinct(answer_seed, candidate_summaries, limit=2)
    while len(distractors) < 2:
        distractors.append("Continue with a follow-up that sounds helpful but does not fit the earlier constraints.")
    unavailable = UNAVAILABLE_RESPONSES[2]
    return {
        "qa_family": "application",
        "answer_seed": answer_seed,
        "distractor_seeds": distractors,
        "correct_option_available": "A",
        "correct_option_restricted": "D",
        "render_prompt": _build_reasoning_render_prompt(
            turn,
            answer_seed=answer_seed,
            distractor_seeds=distractors,
            unavailable_option=unavailable,
        ),
    }


def _turn_bundle(turn: Dict, pool: List[Dict], turn_role: str) -> Dict:
    bundle = {
        "timestamp": turn["timestamp"],
        "turn_role": turn_role,
        "event_id": turn.get("event_id"),
        "source_event_id": turn.get("source_event_id"),
        "task_goal": turn.get("task_goal", ""),
        "context_can_add": turn.get("context_can_add", {}),
        "sensitive_info": turn.get("sensitive_info", {}),
        "user_turn": _conversation_user_turn(turn),
        "whole_recall": _whole_recall_spec(turn, pool),
        "slot_recall": _slot_recall_specs(turn, pool),
        "application": _application_spec(turn, pool),
    }
    other_turn_labels = [str(other.get("task_goal", "")).strip() for other in pool if other["timestamp"] != turn["timestamp"]]
    bundle["whole_recall"]["disambiguation_check_prompt"] = _build_disambiguation_check_prompt(
        turn,
        qa_family="whole_recall",
        render_prompt=bundle["whole_recall"]["render_prompt"],
        other_turn_labels=other_turn_labels,
    )
    for spec in bundle["slot_recall"]:
        spec["disambiguation_check_prompt"] = _build_disambiguation_check_prompt(
            turn,
            qa_family="slot_recall",
            render_prompt=spec["render_prompt"],
            other_turn_labels=other_turn_labels,
            slot_name=spec["slot_name"],
        )
    bundle["application"]["disambiguation_check_prompt"] = _build_disambiguation_check_prompt(
        turn,
        qa_family="application",
        render_prompt=bundle["application"]["render_prompt"],
        other_turn_labels=other_turn_labels,
    )
    return bundle


def build_mcq_spec_dict(sidecar_path: str) -> Dict:
    sidecar = load_json(sidecar_path)
    key_turns = sidecar.get("key_turns", [])
    probe_turns = sidecar.get("protected_probe_turns", [])
    turn_pool = key_turns + probe_turns
    return {
        "source_sidecar": sidecar_path,
        "source_conversation": sidecar.get("source_file", ""),
        "baseline_resolution": sidecar.get("baseline_resolution", {}),
        "mcq_design_version": "v2_prompt_ready",
        "key_turns": [_turn_bundle(turn, turn_pool, "key") for turn in key_turns],
        "probe_turns": [_turn_bundle(turn, turn_pool, "probe") for turn in probe_turns],
    }


def build_mcq_spec_bundle(sidecar_path: str) -> str:
    return json.dumps(build_mcq_spec_dict(sidecar_path), ensure_ascii=False, indent=2)
