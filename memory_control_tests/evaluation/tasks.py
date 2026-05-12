from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

from ..common import PERIODS, TARGET_INSTRUCTION_PERIODS
from .shared import extract_choice


def build_period_messages(data: Dict[str, Any], period: str) -> List[Dict[str, str]]:
    lines = data.get(period, [])
    messages: List[Dict[str, str]] = []
    if not isinstance(lines, list):
        return messages
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


def build_incremental_stage_batches(data: Dict[str, Any], ask_period: str) -> List[Dict[str, Any]]:
    if ask_period not in PERIODS:
        return []
    batches: List[Dict[str, Any]] = []
    for period in PERIODS[: PERIODS.index(ask_period) + 1]:
        period_messages = build_period_messages(data, period)
        if period_messages:
            batches.append({"period": period, "messages": period_messages})
    return batches


def build_forget_eval_targets(sidecar: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    targets: Dict[str, Dict[str, str]] = {}
    key_turns = sidecar.get("key_turns", []) if isinstance(sidecar, dict) else []
    probe_turns = sidecar.get("protected_probe_turns", []) if isinstance(sidecar, dict) else []
    for idx, period in enumerate(TARGET_INSTRUCTION_PERIODS):
        key_timestamp = str(key_turns[idx].get("timestamp", "")).strip() if idx < len(key_turns) else ""
        probe_timestamp = str(probe_turns[idx].get("timestamp", "")).strip() if idx < len(probe_turns) else ""
        targets[period] = {
            "key_timestamp": key_timestamp,
            "probe_timestamp": probe_timestamp,
        }
    return targets


def should_score_item_for_world(
    *,
    world: str,
    ask_period: str,
    timestamp: str,
    forget_targets: Dict[str, Dict[str, str]],
) -> bool:
    if world != "forget":
        return True
    stage_targets = forget_targets.get(ask_period, {})
    allowed_timestamps = {
        str(stage_targets.get("key_timestamp", "")).strip(),
        str(stage_targets.get("probe_timestamp", "")).strip(),
    }
    allowed_timestamps.discard("")
    if not allowed_timestamps:
        return False
    return timestamp in allowed_timestamps


def build_mcq_tasks(
    *,
    rendered: Dict[str, Any],
    world: str,
    ask_period: str,
    forget_targets: Dict[str, Dict[str, str]],
    forget_stage_map: Dict[str, str],
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[Tuple[int, int], Dict[str, Any]]]]:
    whole_tasks: List[Tuple[int, Dict[str, Any]]] = []
    for whole_idx, item in enumerate(rendered.get("whole_recall_set", [])):
        timestamp = str(item.get("timestamp", "")).strip()
        if not should_score_item_for_world(
            world=world,
            ask_period=ask_period,
            timestamp=timestamp,
            forget_targets=forget_targets,
        ):
            continue
        rendered_item = item.get("rendered", {})
        choices = rendered_item.get("choices", {})
        choice_to_answer_type = rendered_item.get("choice_to_answer_type", {})
        if not choices or not choice_to_answer_type:
            continue
        whole_tasks.append(
            (
                whole_idx,
                {
                    "timestamp": item["timestamp"],
                    "turn_role": item["turn_role"],
                    "identifier_label": item["identifier_label"],
                    "forget_stage": forget_stage_map.get(item["timestamp"], ""),
                    "question": rendered_item["question"],
                    "choices": choices,
                    "choice_to_answer_type": choice_to_answer_type,
                },
            )
        )

    slot_tasks: List[Tuple[Tuple[int, int], Dict[str, Any]]] = []
    for slot_idx, item in enumerate(rendered.get("slot_recall_set", [])):
        timestamp = str(item.get("timestamp", "")).strip()
        if not should_score_item_for_world(
            world=world,
            ask_period=ask_period,
            timestamp=timestamp,
            forget_targets=forget_targets,
        ):
            continue
        for sub_idx, slot_item in enumerate(item.get("rendered", {}).get("items", [])):
            choices = slot_item.get("choices", {})
            choice_to_answer_type = slot_item.get("choice_to_answer_type", {})
            if not choices or not choice_to_answer_type:
                continue
            slot_tasks.append(
                (
                    (slot_idx, sub_idx),
                    {
                        "timestamp": item["timestamp"],
                        "turn_role": item["turn_role"],
                        "identifier_label": item["identifier_label"],
                        "forget_stage": forget_stage_map.get(item["timestamp"], ""),
                        "sensitive_key": slot_item["sensitive_key"],
                        "sensitive_value": slot_item["sensitive_value"],
                        "question": slot_item["question"],
                        "choices": choices,
                        "choice_to_answer_type": choice_to_answer_type,
                    },
                )
            )
    return whole_tasks, slot_tasks


def decode_answer(model_response: str, choices: Dict[str, str], choice_to_answer_type: Dict[str, str]) -> Tuple[str, str]:
    labels = list(choices.keys())
    predicted_choice = extract_choice(model_response, labels)
    predicted_answer_type = choice_to_answer_type.get(predicted_choice, "")
    return predicted_choice, predicted_answer_type


def run_mcq_tasks(
    *,
    adapter: Any,
    whole_tasks: List[Tuple[int, Dict[str, Any]]],
    slot_tasks: List[Tuple[Tuple[int, int], Dict[str, Any]]],
    run_whole: Callable[[Dict[str, Any]], Dict[str, Any]],
    run_slot: Callable[[Dict[str, Any]], Dict[str, Any]],
    workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parallel = bool(getattr(adapter, "supports_parallel_mcq", False)) and workers > 1
    if not parallel:
        whole_results = [run_whole(payload) for _, payload in whole_tasks]
        slot_results = [run_slot(payload) for _, payload in slot_tasks]
        return whole_results, slot_results

    whole_results_by_idx: Dict[int, Dict[str, Any]] = {}
    slot_results_by_idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        whole_futures = {executor.submit(run_whole, payload): idx for idx, payload in whole_tasks}
        slot_futures = {executor.submit(run_slot, payload): idx for idx, payload in slot_tasks}
        for future in as_completed(list(whole_futures.keys()) + list(slot_futures.keys())):
            if future in whole_futures:
                whole_results_by_idx[whole_futures[future]] = future.result()
            else:
                slot_results_by_idx[slot_futures[future]] = future.result()
    whole_results = [whole_results_by_idx[idx] for idx, _ in whole_tasks]
    slot_results = [slot_results_by_idx[idx] for idx, _ in slot_tasks]
    return whole_results, slot_results
