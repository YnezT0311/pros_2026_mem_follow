import argparse
import ast
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from ..common import classify_slot_type
from ..mcq_specs import (
    _build_slot_recall_render_prompt,
    build_disambiguation_check_prompt,
    build_mcq_spec_dict,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TITLE = "MemoryCtrl"
MODEL_ALIASES = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5-mini": "openai/gpt-5-mini",
}


def _resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _load_client(api_key_file: str = "openrouter_key.txt") -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key and Path(api_key_file).exists():
        api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        legacy_key = os.getenv("OPENAI_API_KEY", "").strip()
        legacy_path = Path("openai_key.txt")
        if not legacy_key and legacy_path.exists():
            legacy_key = legacy_path.read_text(encoding="utf-8").strip()
        api_key = legacy_key
    if not api_key:
        raise FileNotFoundError("No API key found. Set OPENROUTER_API_KEY or provide openrouter_key.txt.")
    base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or OPENROUTER_BASE_URL
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"X-OpenRouter-Title": OPENROUTER_TITLE},
    )


def _extract_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    output = getattr(resp, "output", None)
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()

    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
    return ""


def _request_text(client: OpenAI, model: str, prompt: str) -> str:
    model = _resolve_model_name(model)
    try:
        resp = client.responses.create(model=model, input=prompt)
        text = _extract_text(resp)
        if text:
            return text
    except Exception:
        pass

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content.strip()


def _extract_json_candidate(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    object_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if object_match:
        return object_match.group(1).strip()
    return text


def _parse_json_text(text: str) -> Dict[str, Any]:
    text = _extract_json_candidate(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    repaired = text
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"(?<!\\)'", '"', repaired)
    repaired = re.sub(r'(\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1 "\2":', repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    raise


def _request_json(client: OpenAI, model: str, prompt: str) -> Dict[str, Any]:
    raw = _request_text(client, model, prompt)
    try:
        return _parse_json_text(raw)
    except Exception:
        repair_prompt = (
            "Rewrite the following content as valid JSON only. Do not change its meaning. "
            "Do not add markdown fences or commentary.\n\n"
            f"{raw}"
        )
        repaired = _request_text(client, model, repair_prompt)
        return _parse_json_text(repaired)


WHOLE_SLOT_LEAK_RE = re.compile(
    r"\b(budget|date|dates|email|contact|nightly|schedule|arrival|departure|price|cost)\b",
    re.IGNORECASE,
)
REQUIRED_ANSWER_TYPES = {"remember_correct", "distractor_irrelevant", "not_remember"}


def _whole_recall_needs_repair(rendered_whole: Dict[str, Any]) -> bool:
    question = str(rendered_whole.get("question", ""))
    normalized_answers = _normalize_answer_list(rendered_whole.get("answers", []))
    answer_types = {answer.get("answer_type", "") for answer in normalized_answers}
    missing_core_types = REQUIRED_ANSWER_TYPES - answer_types
    return bool(WHOLE_SLOT_LEAK_RE.search(question)) or len(normalized_answers) < 3 or bool(missing_core_types)


def _build_whole_recall_repair_prompt(
    turn: Dict[str, Any],
    rendered_whole: Dict[str, Any],
    *,
    require_complete_answers: bool = False,
) -> str:
    user_turn = turn["user_turn"]
    task_goal = turn["task_goal"]
    return (
        "You are revising a multiple-choice recall question for a conversation-memory evaluation. "
        f"The earlier user message was: \"{user_turn}\". "
        f"The task goal was: \"{task_goal}\". "
        f"The current identifier_label is: \"{rendered_whole['identifier_label']}\". "
        f"The current question is: \"{rendered_whole['question']}\". "
        "Please rewrite this whole-recall question so that it asks only about the overall purpose of the earlier request. "
        "Do not ask about specific slot-level details such as dates, budgets, email addresses, schedules, contact information, or other sensitive details. "
        "The remember_correct answer should also stay at the level of the overall request rather than listing slot-level facts unless a brief mention is absolutely necessary for clarity. "
        "Make sure the output includes exactly three answers with the answer types remember_correct, distractor_irrelevant, and not_remember. "
        "Keep the same JSON format with identifier_label, question, and answers. "
        "Keep the same three answer types: remember_correct, distractor_irrelevant, and not_remember. "
        "\"not_remember\" should be a natural answer as if you are a helpful assistant who no longer remembers the earlier detail. "
        + (
            "The previous draft was invalid because one or more answer candidates were missing, so all three answer types must be present this time. "
            if require_complete_answers
            else ""
        )
        + "Return valid JSON only and no other words."
    )


def _find_turn(spec_dict: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
    for section in ("key_turns", "probe_turns"):
        for turn in spec_dict.get(section, []):
            if turn.get("timestamp") == timestamp:
                return turn
    raise KeyError(f"Turn not found for timestamp={timestamp}")


def _shuffle_answer_bank(answers: List[Dict[str, str]], seed: str) -> Dict[str, Any]:
    shuffled = list(answers)
    random.Random(seed).shuffle(shuffled)
    labels = ["A", "B", "C"]
    choices = {label: answer["text"] for label, answer in zip(labels, shuffled)}
    answer_type_to_choice = {
        answer["answer_type"]: label for label, answer in zip(labels, shuffled)
    }
    choice_to_answer_type = {
        label: answer["answer_type"] for label, answer in zip(labels, shuffled)
    }
    return {
        "choices": choices,
        "choice_order": labels,
        "answer_type_to_choice": answer_type_to_choice,
        "choice_to_answer_type": choice_to_answer_type,
        "remember_correct_choice": answer_type_to_choice.get("remember_correct", ""),
        "distractor_irrelevant_choice": answer_type_to_choice.get("distractor_irrelevant", ""),
        "not_remember_choice": answer_type_to_choice.get("not_remember", ""),
        "shuffled_answers": [
            {
                "choice": label,
                "answer_type": answer["answer_type"],
                "text": answer["text"],
            }
            for label, answer in zip(labels, shuffled)
        ],
    }


def _normalize_answer_list(answers: Any) -> List[Dict[str, str]]:
    if not isinstance(answers, list):
        return []
    normalized: List[Dict[str, str]] = []
    default_types = ["remember_correct", "distractor_irrelevant", "not_remember"]
    for idx, answer in enumerate(answers):
        if isinstance(answer, dict):
            answer_type = str(answer.get("answer_type", "")).strip() or default_types[min(idx, 2)]
            text = str(answer.get("text", "")).strip()
            if text:
                normalized.append({"answer_type": answer_type, "text": text})
        elif isinstance(answer, str):
            text = answer.strip()
            if text:
                normalized.append(
                    {
                        "answer_type": default_types[min(idx, 2)],
                        "text": text,
                    }
                )
    return normalized


def _validate_answer_bank(normalized_answers: List[Dict[str, str]], *, qa_family: str) -> None:
    if len(normalized_answers) != 3:
        raise ValueError(f"{qa_family} item must contain exactly three answer candidates.")
    answer_types = [answer.get("answer_type", "") for answer in normalized_answers]
    missing = REQUIRED_ANSWER_TYPES - set(answer_types)
    if missing:
        raise ValueError(f"{qa_family} item is missing required answer types: {sorted(missing)}")
    if len(set(answer_types)) != 3:
        raise ValueError(f"{qa_family} item contains duplicated answer types: {answer_types}")


def _finalize_whole_render(rendered_whole: Dict[str, Any], seed: str) -> Dict[str, Any]:
    normalized_answers = _normalize_answer_list(rendered_whole.get("answers", []))
    _validate_answer_bank(normalized_answers, qa_family="whole_recall")
    shuffled = _shuffle_answer_bank(normalized_answers, seed)
    return {
        "identifier_label": rendered_whole["identifier_label"],
        "question": rendered_whole["question"],
        "choices": shuffled["choices"],
        "choice_order": shuffled["choice_order"],
        "answer_type_to_choice": shuffled["answer_type_to_choice"],
        "choice_to_answer_type": shuffled["choice_to_answer_type"],
        "remember_correct_choice": shuffled["remember_correct_choice"],
        "distractor_irrelevant_choice": shuffled["distractor_irrelevant_choice"],
        "not_remember_choice": shuffled["not_remember_choice"],
        "answers_unshuffled": normalized_answers,
        "answers_shuffled": shuffled["shuffled_answers"],
    }


def _repair_whole_recall_until_valid(
    client: OpenAI,
    model: str,
    turn: Dict[str, Any],
    raw_whole: Dict[str, Any],
) -> Dict[str, Any]:
    candidate = raw_whole
    for attempt in range(3):
        if not _whole_recall_needs_repair(candidate):
            return candidate
        candidate = _request_json(
            client,
            model,
            _build_whole_recall_repair_prompt(
                turn,
                candidate,
                require_complete_answers=(attempt > 0),
            ),
        )
    return candidate


def _finalize_slot_render(rendered_slot: Dict[str, Any], seed_prefix: str) -> Dict[str, Any]:
    finalized_items = []
    for idx, item in enumerate(rendered_slot.get("items", [])):
        normalized_answers = _normalize_answer_list(item.get("answers", []))
        _validate_answer_bank(normalized_answers, qa_family="slot_recall")
        shuffled = _shuffle_answer_bank(normalized_answers, f"{seed_prefix}:{idx}")
        finalized_items.append(
            {
                "sensitive_key": item["sensitive_key"],
                "sensitive_value": item["sensitive_value"],
                "slot_type": classify_slot_type(
                    item["sensitive_key"],
                    item["sensitive_value"],
                    item.get("question", ""),
                ),
                "identifier_label": item["identifier_label"],
                "question": item["question"],
                "choices": shuffled["choices"],
                "choice_order": shuffled["choice_order"],
                "answer_type_to_choice": shuffled["answer_type_to_choice"],
                "choice_to_answer_type": shuffled["choice_to_answer_type"],
                "remember_correct_choice": shuffled["remember_correct_choice"],
                "distractor_irrelevant_choice": shuffled["distractor_irrelevant_choice"],
                "not_remember_choice": shuffled["not_remember_choice"],
                "answers_unshuffled": normalized_answers,
                "answers_shuffled": shuffled["shuffled_answers"],
            }
        )
    return {"items": finalized_items}


def _validate_rendered_output(rendered: Dict[str, Any]) -> None:
    for item in rendered.get("whole_recall_set", []):
        payload = item.get("rendered", {})
        if not payload.get("choices") or not payload.get("choice_to_answer_type") or not payload.get("remember_correct_choice"):
            raise ValueError(f"Invalid whole-recall MCQ for timestamp {item.get('timestamp')}")
        if set(payload.get("choice_to_answer_type", {}).values()) != REQUIRED_ANSWER_TYPES:
            raise ValueError(f"Incomplete whole-recall answer typing for timestamp {item.get('timestamp')}")

    for item in rendered.get("slot_recall_set", []):
        for slot_item in item.get("rendered", {}).get("items", []):
            if not slot_item.get("choices") or not slot_item.get("choice_to_answer_type") or not slot_item.get("remember_correct_choice"):
                raise ValueError(
                    f"Invalid slot-recall MCQ for timestamp {item.get('timestamp')} key {slot_item.get('sensitive_key')}"
                )
            if set(slot_item.get("choice_to_answer_type", {}).values()) != REQUIRED_ANSWER_TYPES:
                raise ValueError(
                    f"Incomplete slot-recall answer typing for timestamp {item.get('timestamp')} key {slot_item.get('sensitive_key')}"
                )


def _load_existing_rendered(output_path: Path) -> Dict[str, Any]:
    if not output_path.exists():
        raise FileNotFoundError(
            f"{output_path} does not exist. Create a full rendered file first or run with --qa_family all."
        )
    rendered = json.loads(output_path.read_text(encoding="utf-8"))
    _validate_rendered_output(rendered)
    return rendered


def _index_turns(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in items:
        timestamp = str(item.get("timestamp", "")).strip()
        if timestamp:
            indexed[timestamp] = item
    return indexed


def _render_whole_recall(
    client: OpenAI,
    model: str,
    turn: Dict[str, Any],
    pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_whole = _request_json(client, model, turn["whole_recall"]["render_prompt"])
    raw_whole = _repair_whole_recall_until_valid(client, model, turn, raw_whole)
    rendered_whole = _finalize_whole_render(raw_whole, seed=f"{turn['timestamp']}:whole")

    identifier_label = rendered_whole["identifier_label"]
    disambig_prompt = build_disambiguation_check_prompt(turn, pool, identifier_label)
    disambig = _request_json(client, model, disambig_prompt)

    return {
        "timestamp": turn["timestamp"],
        "turn_role": turn["turn_role"],
        "identifier_label": identifier_label,
        "user_turn": turn["user_turn"],
        "task_goal": turn["task_goal"],
        "rendered": rendered_whole,
        "disambiguation": disambig,
        "is_identifier_unique_to_target": sorted(disambig.get("matched_timestamps", []))
        == [turn["timestamp"]],
    }


def _render_slot_recall(
    client: OpenAI,
    model: str,
    turn: Dict[str, Any],
    identifier_label: str,
) -> Dict[str, Any]:
    raw_slot = _request_json(
        client,
        model,
        _build_slot_recall_render_prompt(
            turn,
            identifier_label=identifier_label,
            distractor_seeds=turn["slot_recall"]["distractor_seeds"],
        ),
    )
    rendered_slot = _finalize_slot_render(raw_slot, seed_prefix=f"{turn['timestamp']}:slot")
    for item in rendered_slot.get("items", []):
        item["identifier_label"] = identifier_label

    return {
        "timestamp": turn["timestamp"],
        "turn_role": turn["turn_role"],
        "identifier_label": identifier_label,
        "user_turn": turn["user_turn"],
        "task_goal": turn["task_goal"],
        "rendered": rendered_slot,
    }


def render_file(
    sidecar_path: str,
    model: str = "gpt-5-mini",
    output_path: str = "",
    api_key_file: str = "openrouter_key.txt",
    qa_family: str = "all",
) -> Path:
    spec_dict = build_mcq_spec_dict(sidecar_path)
    resolved_output_path = Path(output_path or sidecar_path.replace(".memory_control.json", ".recall_rendered.json"))
    client = _load_client(api_key_file)

    pool = spec_dict["key_turns"] + spec_dict["probe_turns"]
    if qa_family == "all":
        rendered = {
            "source_sidecar": spec_dict["source_sidecar"],
            "source_conversation": spec_dict["source_conversation"],
            "model": model,
            "whole_recall_set": [],
            "slot_recall_set": [],
        }
        existing_whole: Dict[str, Dict[str, Any]] = {}
        existing_slot: Dict[str, Dict[str, Any]] = {}
    else:
        rendered = _load_existing_rendered(resolved_output_path)
        rendered["model"] = model
        existing_whole = _index_turns(rendered.get("whole_recall_set", []))
        existing_slot = _index_turns(rendered.get("slot_recall_set", []))
        rendered["whole_recall_set"] = []
        rendered["slot_recall_set"] = []

    for section in ("key_turns", "probe_turns"):
        for turn_stub in spec_dict.get(section, []):
            turn = _find_turn(spec_dict, turn_stub["timestamp"])
            timestamp = turn["timestamp"]

            if qa_family in {"all", "whole"}:
                whole_item = _render_whole_recall(client, model, turn, pool)
            else:
                whole_item = existing_whole.get(timestamp)
                if whole_item is None:
                    raise ValueError(
                        f"Missing existing whole-recall item for timestamp {timestamp}. Run with --qa_family all first."
                    )
            rendered["whole_recall_set"].append(whole_item)

            if qa_family in {"all", "slot"}:
                slot_item = _render_slot_recall(client, model, turn, whole_item["identifier_label"])
            else:
                slot_item = existing_slot.get(timestamp)
                if slot_item is None:
                    raise ValueError(
                        f"Missing existing slot-recall item for timestamp {timestamp}. Run with --qa_family all first."
                    )
            rendered["slot_recall_set"].append(slot_item)

    _validate_rendered_output(rendered)
    resolved_output_path.write_text(json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render recall-only MCQs from baseline MCQ specs.")
    parser.add_argument(
        "--sidecar",
        default="data/test/travelPlanning/specs/conversation_travelPlanning_persona0_sample0.memory_control.json",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--output", default="")
    parser.add_argument("--api_key_file", default="openrouter_key.txt")
    parser.add_argument("--qa_family", choices=["all", "whole", "slot"], default="all")
    args = parser.parse_args()
    output_path = render_file(
        sidecar_path=args.sidecar,
        model=args.model,
        output_path=args.output,
        api_key_file=args.api_key_file,
        qa_family=args.qa_family,
    )
    print(output_path)


if __name__ == "__main__":
    main()
